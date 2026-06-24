import os
import random
from glob import glob

import torch
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset
from torchvision.io import read_image

from .datasets import register


def _normalize_hw(size):
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Invalid spatial size: {size}")


def _list_numbered_pngs(seq_dir, strict_contiguous=True):
    frame_items = []
    for file_name in os.listdir(seq_dir):
        file_path = os.path.join(seq_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        stem, ext = os.path.splitext(file_name)
        if ext.lower() != ".png" or not stem.isdigit():
            continue
        frame_items.append((int(stem), file_path))

    if not frame_items:
        return None

    frame_items.sort(key=lambda item: item[0])
    frame_ids = [item[0] for item in frame_items]
    if frame_ids[0] != 1:
        return None

    if strict_contiguous:
        expected = list(range(1, len(frame_ids) + 1))
        if frame_ids != expected:
            return None

    return frame_items


def _resize_clip(frames, target_size):
    if target_size is None:
        return frames
    return [TF.resize(frame, target_size, antialias=True) for frame in frames]


def _random_crop_clip(frames, crop_size):
    crop_h, crop_w = crop_size
    h, w = frames[0].shape[-2:]

    if h < crop_h or w < crop_w:
        resize_h = max(h, crop_h)
        resize_w = max(w, crop_w)
        frames = _resize_clip(frames, (resize_h, resize_w))
        h, w = frames[0].shape[-2:]

    top = 0 if h == crop_h else random.randint(0, h - crop_h)
    left = 0 if w == crop_w else random.randint(0, w - crop_w)
    return [frame[:, top:top + crop_h, left:left + crop_w] for frame in frames]


@register("video_sequence_folder")
class VideoSequenceFolder(Dataset):
    """
    Read one dense sample directory, then expand it into sparse stride-(interp_num + 1)
    subsequences.

    For interp_num = n, a dense sequence is split into n + 1 phase subsequences:
    1, 1 + n + 1, 1 + 2(n + 1), ...
    2, 2 + n + 1, 2 + 2(n + 1), ...
    ...

    Each dataset item is one adjacent sparse pair sampled from one such phase
    subsequence. The target frame can be chosen in two ways controlled by
    enumerate_all_targets:
    - false: each pair picks one middle frame on the fly
    - true: each pair contributes all middle targets 1..interp_num as separate
      dataset items

    The support window is still built inside that sparse subsequence, e.g. for
    window=4 the support frames are [prev, img0, img1, next] in sparse time.
    Near sequence borders, missing support slots are filled with a clamped real
    frame but marked invalid by frame_mask, which acts as the "virtual frame"
    indicator.

    Returned tensor shapes for one dataset item:
    - img0 / imgt / img1: [C, H, W]
    - time_step: [1, 1, 1]
    - clip: [T, C, H, W], sparse support frames for the context branch
    - frame_mask: [T], 1 for valid support frames and 0 for virtual support slots
    - frame_indices: [T], original dense frame indices of the support frames
    - anchor_indices: [2]

    The custom collate_fn stacks these pair-centric samples into the standard batch
    format expected by the rest of the project.
    """

    def __init__(
        self,
        root_path,
        interp_num,
        split="train",
        patch_size=224,
        clip_length=None,
        pad_to_clip_length=None,
        val_ratio=0.05,
        val_test_patch_mode="resize",
        eval_size=None,
        min_seq_len=2,
        strict_contiguous=True,
        random_horizontal_flip=True,
        random_vertical_flip=False,
        output_pair_fields=True,
        enumerate_all_targets=True,
        target_sampling_mode=None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = root_path
        self.mode = split
        self.interp_num = int(interp_num)
        self.patch_size = _normalize_hw(patch_size)
        self.eval_size = _normalize_hw(eval_size) if eval_size is not None else self.patch_size
        self.clip_length = None if clip_length is None else int(clip_length)
        self.min_seq_len = int(min_seq_len)
        self.val_ratio = float(val_ratio)
        self.val_test_patch_mode = val_test_patch_mode
        self.strict_contiguous = strict_contiguous
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.output_pair_fields = bool(output_pair_fields)
        self.pad_to_clip_length = True if pad_to_clip_length is None else bool(pad_to_clip_length)
        if enumerate_all_targets is None and target_sampling_mode is None:
            enumerate_all_targets = True

        if enumerate_all_targets is not None:
            self.enumerate_all_targets = bool(enumerate_all_targets)
        else:
            legacy_mode = str(target_sampling_mode).lower()
            if legacy_mode == "enumerate_all":
                self.enumerate_all_targets = True
            elif legacy_mode == "random_one":
                self.enumerate_all_targets = False
            else:
                raise ValueError(
                    "target_sampling_mode must be one of {'random_one', 'enumerate_all'}, "
                    f"but got {target_sampling_mode!r}"
                )

        if not self.output_pair_fields:
            raise ValueError("video_sequence_folder expects output_pair_fields=True in sequence training mode.")

        self.anchor_gap = self.interp_num + 1
        self.required_seq_len = max(self.min_seq_len, self.anchor_gap + 1)
        self.context_length = 4 if self.clip_length is None else self.clip_length

        if self.interp_num < 1:
            raise ValueError("interp_num must be >= 1 when output_pair_fields=True")
        if self.context_length < 2:
            raise ValueError(
                f"clip_length/context_length ({self.context_length}) must be >= 2 for pair-centric support windows"
            )
        if not self.pad_to_clip_length:
            raise ValueError("video_sequence_folder requires pad_to_clip_length=True for pair batching.")
        if self.mode not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported split: {self.mode}")

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No valid video sequence samples found in: {self.data_root}")

    def _split_valid_sequences(self):
        first_frame_list = glob(os.path.join(self.data_root, "**", "1.png"), recursive=True)
        sample_dirs = sorted(set(os.path.dirname(path) for path in first_frame_list))

        valid_sequences = []
        for sample_dir in sample_dirs:
            frame_items = _list_numbered_pngs(sample_dir, strict_contiguous=self.strict_contiguous)
            if frame_items is None or len(frame_items) < self.required_seq_len:
                continue

            rel_dir = os.path.relpath(sample_dir, self.data_root)
            valid_sequences.append(
                {
                    "sample_dir": sample_dir,
                    "sample_name": rel_dir,
                    "frame_items": frame_items,
                }
            )

        if not valid_sequences:
            return []

        if self.mode == "test":
            return valid_sequences

        if len(valid_sequences) == 1:
            return valid_sequences

        split_idx = max(1, int(len(valid_sequences) * (1.0 - self.val_ratio)))
        if split_idx >= len(valid_sequences):
            split_idx = len(valid_sequences) - 1

        if self.mode == "train":
            return valid_sequences[:split_idx]
        return valid_sequences[split_idx:]

    def _build_samples(self):
        base_sequences = self._split_valid_sequences()
        samples = []

        for sequence in base_sequences:
            dense_length = len(sequence["frame_items"])
            for phase_index in range(self.anchor_gap):
                sparse_anchor_positions = list(range(phase_index, dense_length, self.anchor_gap))
                if len(sparse_anchor_positions) < 2:
                    continue
                for pair_idx in range(len(sparse_anchor_positions) - 1):
                    if self.enumerate_all_targets:
                        target_offsets = range(1, self.interp_num + 1)
                    else:
                        target_offsets = [None]
                    for target_offset in target_offsets:
                        samples.append(
                            {
                                "sample_dir": sequence["sample_dir"],
                                "sample_name": sequence["sample_name"],
                                "frame_items": sequence["frame_items"],
                                "phase_index": phase_index,
                                "pair_index": pair_idx,
                                "target_offset": target_offset,
                                "sparse_anchor_positions": sparse_anchor_positions,
                            }
                        )

        return samples

    def __len__(self):
        return len(self.samples)

    def _augment_train_clip(self, frames):
        if self.patch_size is not None:
            frames = _random_crop_clip(frames, self.patch_size)
        if self.random_horizontal_flip and random.random() > 0.5:
            frames = [TF.horizontal_flip(frame) for frame in frames]
        if self.random_vertical_flip and random.random() > 0.5:
            frames = [TF.vertical_flip(frame) for frame in frames]
        return frames

    def _process_eval_clip(self, frames):
        if self.val_test_patch_mode == "resize":
            return _resize_clip(frames, self.eval_size)
        if self.val_test_patch_mode == "keep":
            return frames
        raise ValueError(f"Unsupported val_test_patch_mode: {self.val_test_patch_mode}")

    def _build_sparse_support_window(self, sparse_anchor_positions, pair_idx):
        left_extra = (self.context_length - 2) // 2
        right_extra = self.context_length - 2 - left_extra
        desired_sparse_indices = list(range(pair_idx - left_extra, pair_idx + 2 + right_extra))

        selected_positions = []
        frame_mask = []
        for sparse_idx in desired_sparse_indices:
            if 0 <= sparse_idx < len(sparse_anchor_positions):
                selected_positions.append(sparse_anchor_positions[sparse_idx])
                frame_mask.append(1.0)
            else:
                clamped_idx = min(max(sparse_idx, 0), len(sparse_anchor_positions) - 1)
                selected_positions.append(sparse_anchor_positions[clamped_idx])
                frame_mask.append(0.0)

        anchor_indices = torch.tensor([left_extra, left_extra + 1], dtype=torch.long)
        frame_mask = torch.tensor(frame_mask, dtype=torch.float32)
        valid_num_frames = int(frame_mask.sum().item())
        return selected_positions, frame_mask, valid_num_frames, anchor_indices

    def _select_target_offset(self):
        if self.mode == "train":
            return random.randint(1, self.interp_num)
        return (self.interp_num + 1) // 2

    def _load_required_frames(self, frame_items, required_positions):
        unique_positions = sorted(set(int(pos) for pos in required_positions))
        frames = [read_image(frame_items[pos][1]) for pos in unique_positions]

        if self.mode == "train":
            frames = self._augment_train_clip(frames)
        else:
            frames = self._process_eval_clip(frames)

        frames = [TF.to_dtype(frame, torch.float32, scale=True) for frame in frames]
        return {pos: frame for pos, frame in zip(unique_positions, frames)}

    def __getitem__(self, index):
        sample = self.samples[index]
        frame_items = sample["frame_items"]
        sparse_anchor_positions = sample["sparse_anchor_positions"]
        orig_num_frames = len(frame_items)
        pair_idx = int(sample["pair_index"])
        anchor_start = sparse_anchor_positions[pair_idx]
        anchor_end = sparse_anchor_positions[pair_idx + 1]
        if sample["target_offset"] is None:
            target_offset = self._select_target_offset()
        else:
            target_offset = int(sample["target_offset"])
        target_dense_index = anchor_start + target_offset

        selected_positions, frame_mask, valid_num_frames, anchor_indices = self._build_sparse_support_window(
            sparse_anchor_positions=sparse_anchor_positions,
            pair_idx=pair_idx,
        )
        loaded_frames = self._load_required_frames(
            frame_items,
            selected_positions + [anchor_start, target_dense_index, anchor_end],
        )

        clip = torch.stack([loaded_frames[pos] for pos in selected_positions], dim=0)
        frame_indices = torch.tensor([frame_items[pos][0] for pos in selected_positions], dtype=torch.long)
        clip_scene_names = [os.path.relpath(frame_items[pos][1], self.data_root) for pos in selected_positions]
        scene_names = [
            os.path.relpath(frame_items[anchor_start][1], self.data_root),
            os.path.relpath(frame_items[target_dense_index][1], self.data_root),
            os.path.relpath(frame_items[anchor_end][1], self.data_root),
        ]

        if self.interp_num > 0:
            interp_time_steps = torch.arange(
                1, self.interp_num + 1, dtype=torch.float32
            ) / float(self.interp_num + 1)
        else:
            interp_time_steps = torch.empty(0, dtype=torch.float32)

        return {
            "img0": loaded_frames[anchor_start],
            "imgt": loaded_frames[target_dense_index],
            "img1": loaded_frames[anchor_end],
            "time_step": torch.tensor(
                [float(target_offset) / float(self.anchor_gap)],
                dtype=torch.float32,
            ).reshape(1, 1, 1),
            "clip": clip,
            "frame_mask": frame_mask,
            "frame_indices": frame_indices,
            "num_frames": torch.tensor(valid_num_frames, dtype=torch.long),
            "orig_num_frames": torch.tensor(orig_num_frames, dtype=torch.long),
            "interp_num": torch.tensor(self.interp_num, dtype=torch.long),
            "interp_time_steps": interp_time_steps,
            "anchor_indices": anchor_indices,
            "scene_names": scene_names,
            "clip_scene_names": clip_scene_names,
            "sample_name": sample["sample_name"],
            "phase_index": torch.tensor(sample["phase_index"], dtype=torch.long),
            "pair_index": torch.tensor(pair_idx, dtype=torch.long),
            "target_offset": torch.tensor(target_offset, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        if not batch:
            return {}

        tensor_stack_keys = [
            "img0",
            "imgt",
            "img1",
            "time_step",
            "clip",
            "frame_mask",
            "frame_indices",
            "num_frames",
            "orig_num_frames",
            "anchor_indices",
            "phase_index",
            "pair_index",
            "target_offset",
        ]
        collated = {}

        for key in tensor_stack_keys:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch], dim=0)

        collated["interp_num"] = batch[0]["interp_num"]
        collated["interp_time_steps"] = batch[0]["interp_time_steps"]

        scene_names = [[], [], []]
        clip_scene_names = []
        sample_names = []

        for item in batch:
            scene_names[0].append(item["scene_names"][0])
            scene_names[1].append(item["scene_names"][1])
            scene_names[2].append(item["scene_names"][2])
            clip_scene_names.append(item["clip_scene_names"])
            sample_names.append(item["sample_name"])

        collated["scene_names"] = scene_names
        collated["clip_scene_names"] = clip_scene_names
        collated["sample_name"] = sample_names
        return collated
