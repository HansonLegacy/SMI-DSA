import os

import torch

from .datasets import register
from .video_sequence_folder import VideoSequenceFolder


@register("video_sequence_global_context_folder")
class VideoSequenceGlobalContextFolder(VideoSequenceFolder):
    """
    Pair-centric VFI dataset with an additional variable-length global context.

    The inherited pair fields stay the same as VideoSequenceFolder:
    - img0 / imgt / img1: [C, H, W]
    - time_step: [1, 1, 1]
    - clip / frame_mask / frame_indices / anchor_indices: local sparse support window

    Extra fields for the global context branch:
    - global_clip: [G, C, H, W], variable-length observed sequence for this sample
    - global_frame_mask: [G], 1 for valid observed frames before batch padding
    - global_frame_indices: [G], original 1-based frame ids from file names
    - global_frame_positions: [G], normalized global positions, e.g. frame 3 of 7 -> 3/7
    - global_anchor_indices: [2], img0/img1 positions inside global_clip
    - global_anchor_positions: [2], normalized global positions of img0/img1
    - target_global_position: [1], normalized global position of imgt

    The local context aliases describe the short target-centered window used by
    the local fine extractor. The sparse_sequence aliases describe the full
    sparse input sequence used by the global memory compressor.

    By default, global_clip is the whole sparse anchor sequence for the current
    phase. This gives the interpolation model bidirectional context without
    leaking the dense middle-frame target into the global memory.
    """

    _VALID_GLOBAL_SOURCES = {"sparse_anchors", "dense_without_target", "dense_all"}
    _VALID_POSITION_NORM_MODES = {"one_based", "zero_to_one"}

    def __init__(
        self,
        *args,
        global_context_source="sparse_anchors",
        global_context_max_frames=None,
        position_norm_mode="one_based",
        output_local_context_fields=True,
        output_sparse_sequence_fields=True,
        **kwargs,
    ):
        self.global_context_source = str(global_context_source)
        if self.global_context_source not in self._VALID_GLOBAL_SOURCES:
            raise ValueError(
                "global_context_source must be one of "
                f"{sorted(self._VALID_GLOBAL_SOURCES)}, but got {global_context_source!r}"
            )
        self.global_context_max_frames = (
            None if global_context_max_frames is None else int(global_context_max_frames)
        )
        if self.global_context_max_frames is not None and self.global_context_max_frames < 2:
            raise ValueError("global_context_max_frames must be None or >= 2")
        self.position_norm_mode = str(position_norm_mode)
        if self.position_norm_mode not in self._VALID_POSITION_NORM_MODES:
            raise ValueError(
                "position_norm_mode must be one of "
                f"{sorted(self._VALID_POSITION_NORM_MODES)}, but got {position_norm_mode!r}"
            )
        self.output_local_context_fields = bool(output_local_context_fields)
        self.output_sparse_sequence_fields = bool(output_sparse_sequence_fields)
        super().__init__(*args, **kwargs)

    def _select_global_context_positions(
        self,
        frame_items,
        sparse_anchor_positions,
        anchor_start,
        anchor_end,
        target_dense_index,
    ):
        if self.global_context_source == "sparse_anchors":
            positions = list(sparse_anchor_positions)
        elif self.global_context_source == "dense_without_target":
            positions = [pos for pos in range(len(frame_items)) if pos != target_dense_index]
        elif self.global_context_source == "dense_all":
            positions = list(range(len(frame_items)))
        else:
            raise ValueError(f"Unsupported global_context_source: {self.global_context_source}")

        positions = sorted(set(int(pos) for pos in positions) | {int(anchor_start), int(anchor_end)})
        if self.global_context_source != "dense_all" and target_dense_index in positions:
            positions.remove(target_dense_index)
        positions = sorted(positions)

        if self.global_context_max_frames is not None and len(positions) > self.global_context_max_frames:
            positions = self._limit_global_context_positions(
                positions,
                anchor_start=anchor_start,
                anchor_end=anchor_end,
                target_dense_index=target_dense_index,
            )

        return positions

    def _limit_global_context_positions(self, positions, anchor_start, anchor_end, target_dense_index):
        keep = {positions[0], positions[-1], int(anchor_start), int(anchor_end)}
        ranked_positions = sorted(
            positions,
            key=lambda pos: (abs(int(pos) - int(target_dense_index)), int(pos)),
        )
        for pos in ranked_positions:
            if len(keep) >= self.global_context_max_frames:
                break
            keep.add(int(pos))
        return sorted(keep)

    def _normalized_positions(self, frame_items, positions):
        first_frame_id = int(frame_items[0][0])
        max_frame_id = max(1, int(frame_items[-1][0]))
        if self.position_norm_mode == "one_based":
            return torch.tensor(
                [float(frame_items[int(pos)][0]) / float(max_frame_id) for pos in positions],
                dtype=torch.float32,
            )

        denom = max(1, max_frame_id - first_frame_id)
        return torch.tensor(
            [(float(frame_items[int(pos)][0]) - float(first_frame_id)) / float(denom) for pos in positions],
            dtype=torch.float32,
        )

    def _local_relative_positions(self, positions, anchor_start):
        return torch.tensor(
            [float(int(pos) - int(anchor_start)) / float(self.anchor_gap) for pos in positions],
            dtype=torch.float32,
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        frame_items = sample["frame_items"]
        sparse_anchor_positions = sample["sparse_anchor_positions"]
        orig_num_frames = len(frame_items)
        phase_index = int(sample["phase_index"])
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
        global_positions = self._select_global_context_positions(
            frame_items=frame_items,
            sparse_anchor_positions=sparse_anchor_positions,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
            target_dense_index=target_dense_index,
        )

        required_positions = selected_positions + global_positions + [
            anchor_start,
            target_dense_index,
            anchor_end,
        ]
        loaded_frames = self._load_required_frames(frame_items, required_positions)

        clip = torch.stack([loaded_frames[pos] for pos in selected_positions], dim=0)
        global_clip = torch.stack([loaded_frames[pos] for pos in global_positions], dim=0)

        frame_indices = torch.tensor([frame_items[pos][0] for pos in selected_positions], dtype=torch.long)
        local_frame_positions = self._normalized_positions(frame_items, selected_positions)
        local_frame_relative_positions = self._local_relative_positions(selected_positions, anchor_start)
        global_frame_indices = torch.tensor([frame_items[pos][0] for pos in global_positions], dtype=torch.long)
        global_dense_indices = torch.tensor(global_positions, dtype=torch.long)
        global_frame_positions = self._normalized_positions(frame_items, global_positions)

        global_anchor_indices = torch.tensor(
            [global_positions.index(anchor_start), global_positions.index(anchor_end)],
            dtype=torch.long,
        )
        global_anchor_positions = self._normalized_positions(frame_items, [anchor_start, anchor_end])
        target_global_position = self._normalized_positions(frame_items, [target_dense_index])
        target_local_position = torch.tensor(
            [float(target_offset) / float(self.anchor_gap)],
            dtype=torch.float32,
        )
        anchor_frame_indices = torch.tensor(
            [frame_items[anchor_start][0], frame_items[anchor_end][0]],
            dtype=torch.long,
        )
        anchor_dense_indices = torch.tensor([anchor_start, anchor_end], dtype=torch.long)
        target_frame_index = torch.tensor(frame_items[target_dense_index][0], dtype=torch.long)
        target_dense_index_tensor = torch.tensor(target_dense_index, dtype=torch.long)
        target_frame_id = int(frame_items[target_dense_index][0])
        save_subdir = os.path.join(sample["sample_name"], f"phase_{phase_index:02d}")
        save_stem = f"t{target_frame_id:04d}_off{target_offset:02d}"

        clip_scene_names = [os.path.relpath(frame_items[pos][1], self.data_root) for pos in selected_positions]
        global_scene_names = [os.path.relpath(frame_items[pos][1], self.data_root) for pos in global_positions]
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

        result = {
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
            "global_clip": global_clip,
            "global_frame_mask": torch.ones(len(global_positions), dtype=torch.float32),
            "global_frame_indices": global_frame_indices,
            "global_dense_indices": global_dense_indices,
            "global_frame_positions": global_frame_positions,
            "global_num_frames": torch.tensor(len(global_positions), dtype=torch.long),
            "global_anchor_indices": global_anchor_indices,
            "global_anchor_positions": global_anchor_positions,
            "target_global_position": target_global_position,
            "anchor_frame_indices": anchor_frame_indices,
            "anchor_dense_indices": anchor_dense_indices,
            "target_frame_index": target_frame_index,
            "target_dense_index": target_dense_index_tensor,
            "scene_names": scene_names,
            "clip_scene_names": clip_scene_names,
            "global_scene_names": global_scene_names,
            "save_subdir": save_subdir,
            "save_stem": save_stem,
            "sample_name": sample["sample_name"],
            "phase_index": torch.tensor(phase_index, dtype=torch.long),
            "pair_index": torch.tensor(pair_idx, dtype=torch.long),
            "target_offset": torch.tensor(target_offset, dtype=torch.long),
        }

        if self.output_local_context_fields:
            result.update(
                {
                    "local_clip": clip,
                    "local_frame_mask": frame_mask,
                    "local_frame_indices": frame_indices,
                    "local_frame_positions": local_frame_positions,
                    "local_frame_relative_positions": local_frame_relative_positions,
                    "local_num_frames": torch.tensor(valid_num_frames, dtype=torch.long),
                    "local_anchor_indices": anchor_indices,
                    "local_anchor_positions": global_anchor_positions,
                    "target_local_position": target_local_position,
                }
            )

        if self.output_sparse_sequence_fields and self.global_context_source == "sparse_anchors":
            result.update(
                {
                    "sparse_sequence_clip": global_clip,
                    "sparse_sequence_frame_mask": result["global_frame_mask"],
                    "sparse_sequence_frame_indices": global_frame_indices,
                    "sparse_sequence_dense_indices": global_dense_indices,
                    "sparse_sequence_frame_positions": global_frame_positions,
                    "sparse_sequence_num_frames": result["global_num_frames"],
                    "sparse_sequence_anchor_indices": global_anchor_indices,
                    "sparse_sequence_anchor_positions": global_anchor_positions,
                }
            )

        return result

    @staticmethod
    def _pad_global_tensors(batch):
        batch_size = len(batch)
        max_len = max(item["global_clip"].shape[0] for item in batch)
        first_clip = batch[0]["global_clip"]
        channels, height, width = first_clip.shape[1:]

        global_clip = first_clip.new_zeros((batch_size, max_len, channels, height, width))
        global_frame_mask = first_clip.new_zeros((batch_size, max_len))
        global_frame_indices = torch.zeros((batch_size, max_len), dtype=torch.long)
        global_dense_indices = torch.full((batch_size, max_len), -1, dtype=torch.long)
        global_frame_positions = first_clip.new_zeros((batch_size, max_len))

        for batch_idx, item in enumerate(batch):
            seq_len = item["global_clip"].shape[0]
            global_clip[batch_idx, :seq_len] = item["global_clip"]
            global_frame_mask[batch_idx, :seq_len] = item["global_frame_mask"]
            global_frame_indices[batch_idx, :seq_len] = item["global_frame_indices"]
            global_dense_indices[batch_idx, :seq_len] = item["global_dense_indices"]
            global_frame_positions[batch_idx, :seq_len] = item["global_frame_positions"]

        return {
            "global_clip": global_clip,
            "global_frame_mask": global_frame_mask,
            "global_frame_indices": global_frame_indices,
            "global_dense_indices": global_dense_indices,
            "global_frame_positions": global_frame_positions,
        }

    @staticmethod
    def collate_fn(batch):
        if not batch:
            return {}

        collated = VideoSequenceFolder.collate_fn(batch)
        collated.update(VideoSequenceGlobalContextFolder._pad_global_tensors(batch))

        tensor_stack_keys = [
            "global_num_frames",
            "global_anchor_indices",
            "global_anchor_positions",
            "target_global_position",
            "anchor_frame_indices",
            "anchor_dense_indices",
            "target_frame_index",
            "target_dense_index",
        ]
        for key in tensor_stack_keys:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)

        if "local_clip" in batch[0]:
            collated["local_clip"] = collated["clip"]
            collated["local_frame_mask"] = collated["frame_mask"]
            collated["local_frame_indices"] = collated["frame_indices"]
            collated["local_num_frames"] = collated["num_frames"]
            collated["local_anchor_indices"] = collated["anchor_indices"]
            for key in [
                "local_frame_positions",
                "local_frame_relative_positions",
                "local_anchor_positions",
                "target_local_position",
            ]:
                collated[key] = torch.stack([item[key] for item in batch], dim=0)

        if "sparse_sequence_clip" in batch[0]:
            collated["sparse_sequence_clip"] = collated["global_clip"]
            collated["sparse_sequence_frame_mask"] = collated["global_frame_mask"]
            collated["sparse_sequence_frame_indices"] = collated["global_frame_indices"]
            collated["sparse_sequence_dense_indices"] = collated["global_dense_indices"]
            collated["sparse_sequence_frame_positions"] = collated["global_frame_positions"]
            collated["sparse_sequence_num_frames"] = collated["global_num_frames"]
            collated["sparse_sequence_anchor_indices"] = collated["global_anchor_indices"]
            collated["sparse_sequence_anchor_positions"] = collated["global_anchor_positions"]

        collated["global_scene_names"] = [item["global_scene_names"] for item in batch]
        collated["save_subdir"] = [item["save_subdir"] for item in batch]
        collated["save_stem"] = [item["save_stem"] for item in batch]
        return collated
