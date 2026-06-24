import argparse
import json
import re
import shutil
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from modules.components import make_components


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a sparse anchor sequence from dense frames and interpolate the missing frames."
    )
    parser.add_argument("--cfg", default="cfgs/bim_vfi_bimprior_only.yaml")
    parser.add_argument("--ckpt", default=None, help="Override checkpoint path. Defaults to cfg.resume or cfg.pretrained.")
    parser.add_argument("--input_dir", required=True, help="Dense input sequence directory, e.g. contains 1.png, 2.png, ...")
    parser.add_argument("--name", required=True, help="Output folder name under save/seq/<name>.")
    parser.add_argument("--interp_num", type=int, required=True, help="Number of frames to interpolate between sparse anchors.")
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=None,
        help="Anchor spacing in the dense sequence. Defaults to interp_num + 1.",
    )
    parser.add_argument(
        "--no_include_last",
        action="store_true",
        help="Do not force the last dense frame into the sparse anchor sequence.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use autocast fp16 on CUDA.")
    return parser.parse_args()


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    key.append(path.suffix.lower())
    return key


def list_frames(input_dir: Path):
    frames = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    frames.sort(key=natural_key)
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 image frames in {input_dir}, but found {len(frames)}.")
    return frames


def load_cfg(cfg_path: Path):
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config {cfg_path} did not load into a dict.")
    if "model" not in cfg or "name" not in cfg["model"]:
        raise KeyError(f"Config {cfg_path} is missing model.name.")
    cfg.setdefault("model", {})
    cfg["model"].setdefault("args", {})
    return cfg


def resolve_path(path_str, cfg_path: Path):
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_file():
        return path.resolve()
    cfg_relative = (cfg_path.parent / path).resolve()
    if cfg_relative.is_file():
        return cfg_relative
    cwd_relative = (Path.cwd() / path).resolve()
    if cwd_relative.is_file():
        return cwd_relative
    return path.resolve()


def resolve_ckpt_path(cfg, cfg_path: Path, override_ckpt):
    ckpt_path = override_ckpt or cfg.get("resume") or cfg.get("pretrained")
    if ckpt_path is None:
        raise ValueError("No checkpoint path found. Please pass --ckpt or set cfg.resume/cfg.pretrained.")
    resolved = resolve_path(ckpt_path, cfg_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_checkpoint(model, ckpt_path: Path):
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net", "params"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}.")
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] ckpt={ckpt_path}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[LOAD] first_missing={missing[:10]}")
    if unexpected:
        print(f"[LOAD] first_unexpected={unexpected[:10]}")


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if image.ndim == 2:
        mode = "gray"
        rgb = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        mode = "gray"
        rgb = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 3:
        mode = "rgb"
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        mode = "rgb"
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")

    if np.issubdtype(image.dtype, np.integer):
        value_scale = float(np.iinfo(image.dtype).max)
    elif np.issubdtype(image.dtype, np.floating):
        value_scale = 1.0
    else:
        raise TypeError(f"Unsupported image dtype {image.dtype} for {path}")

    return {
        "path": path,
        "image": image,
        "rgb": rgb,
        "mode": mode,
        "dtype": image.dtype,
        "value_scale": value_scale,
    }


def image_to_tensor(image_info, device):
    rgb = image_info["rgb"].astype(np.float32) / image_info["value_scale"]
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    return tensor


def save_tensor_as_image(tensor, out_path: Path, save_mode, output_dtype, value_scale):
    arr = tensor.detach().clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
    if save_mode == "gray":
        arr = arr.mean(axis=2)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if np.issubdtype(output_dtype, np.integer):
        arr = np.clip(arr * value_scale + 0.5, 0, value_scale).astype(output_dtype)
    else:
        arr = arr.astype(output_dtype)

    cv2.imwrite(str(out_path), arr)


def build_sparse_positions(num_frames, sample_stride, include_last):
    positions = list(range(0, num_frames, sample_stride))
    if include_last and positions[-1] != num_frames - 1:
        positions.append(num_frames - 1)
    positions = sorted(set(positions))
    if len(positions) < 2:
        raise ValueError("Sparse anchor sequence must contain at least 2 frames.")
    return positions


def write_metadata(meta_path: Path, payload):
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def main():
    args = parse_args()

    if args.interp_num < 0:
        raise ValueError("--interp_num must be >= 0.")

    sample_stride = args.sample_stride if args.sample_stride is not None else args.interp_num + 1
    if sample_stride < 1:
        raise ValueError("--sample_stride must be >= 1.")

    cfg_path = Path(args.cfg).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = (Path.cwd() / "save" / "seq" / args.name).resolve()
    sparse_dir = output_dir / "_sparse_input"

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    cfg = load_cfg(cfg_path)
    ckpt_path = resolve_ckpt_path(cfg, cfg_path, args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] cfg={cfg_path}")
    print(f"[INFO] ckpt={ckpt_path}")

    model = make_components(cfg["model"])
    model.to(device)
    model.eval()
    load_checkpoint(model, ckpt_path)

    frames = list_frames(input_dir)
    include_last = not args.no_include_last
    sparse_positions = build_sparse_positions(len(frames), sample_stride, include_last)
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    save_reference = read_image(frames[0])

    metadata = {
        "cfg": str(cfg_path),
        "ckpt": str(ckpt_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "interp_num": args.interp_num,
        "sample_stride": sample_stride,
        "include_last": include_last,
        "dense_frames": [frame.name for frame in frames],
        "sparse_positions": sparse_positions,
        "sparse_frames": [frames[idx].name for idx in sparse_positions],
        "pairs": [],
    }

    print(f"[INFO] dense_frames={len(frames)}")
    print(f"[INFO] sparse_anchors={len(sparse_positions)}")
    print(f"[INFO] save_dir={output_dir}")

    def autocast_context():
        if args.fp16 and device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    last_saved_name = None
    for pair_index, (left_pos, right_pos) in enumerate(zip(sparse_positions[:-1], sparse_positions[1:]), start=1):
        left_path = frames[left_pos]
        right_path = frames[right_pos]
        gap_count = right_pos - left_pos - 1

        shutil.copy2(left_path, sparse_dir / left_path.name)
        if pair_index == len(sparse_positions) - 1:
            shutil.copy2(right_path, sparse_dir / right_path.name)

        if last_saved_name != left_path.name:
            shutil.copy2(left_path, output_dir / left_path.name)
            last_saved_name = left_path.name
            print(f"[SAVE] anchor {left_path.name}")

        left_image = read_image(left_path)
        right_image = read_image(right_path)
        img0 = image_to_tensor(left_image, device)
        img1 = image_to_tensor(right_image, device)

        print(f"[PAIR {pair_index}] {left_path.name} -> {right_path.name}, fill={gap_count}")
        pair_meta = {
            "left": left_path.name,
            "right": right_path.name,
            "fill_count": gap_count,
            "generated": [],
        }

        with autocast_context():
            for missing_offset in range(1, gap_count + 1):
                target_index = left_pos + missing_offset
                target_name = frames[target_index].name
                t = missing_offset / (gap_count + 1)
                pred = model(img0=img0, img1=img1, time_step=float(t), run_with_gt=False)["imgt_pred"]
                save_tensor_as_image(
                    pred,
                    output_dir / target_name,
                    save_mode=save_reference["mode"],
                    output_dtype=save_reference["dtype"],
                    value_scale=save_reference["value_scale"],
                )
                pair_meta["generated"].append({"name": target_name, "time_step": t})
                last_saved_name = target_name
                print(f"[SAVE] pred {target_name} t={t:.6f}")

        shutil.copy2(right_path, output_dir / right_path.name)
        last_saved_name = right_path.name
        print(f"[SAVE] anchor {right_path.name}")
        metadata["pairs"].append(pair_meta)

    write_metadata(output_dir / "meta.json", metadata)
    print(f"[DONE] sequence saved to {output_dir}")


if __name__ == "__main__":
    main()
