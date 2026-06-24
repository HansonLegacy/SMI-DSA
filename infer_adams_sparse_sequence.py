import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate one numbered sparse image sequence with the "
            "ADAMS/DMSE/ASD BiM-VFI model."
        )
    )
    parser.add_argument("--cfg", default="cfgs/infer_adams_dmse_asd_sparse_sequence.yaml")
    parser.add_argument("--model_cfg", default=None, help="Override the model/training yaml.")
    parser.add_argument("--ckpt", default=None, help="Override checkpoint path.")
    parser.add_argument("--input_dir", default=None, help="Override sparse sequence directory.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--interp_num", type=int, default=None, help="Override inserted frames per sparse pair.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA autocast fp16.")
    return parser.parse_args()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML must load into a dict: {path}")
    return data


def resolve_path(path_value, *, cfg_path: Path, must_exist=False):
    if path_value is None:
        return None

    path = Path(str(path_value)).expanduser()
    candidates = [path] if path.is_absolute() else [cfg_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if not must_exist or candidate.exists():
            return candidate

    resolved = candidates[-1].resolve()
    if must_exist:
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved


def resolve_output_dir(output_dir, input_dir: Path, output_subdir: str):
    if output_dir is None:
        return (input_dir / output_subdir).resolve()
    path = Path(str(output_dir)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def select_device(device_name):
    if device_name is None or str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(device_name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is not available.")
    return device


def numeric_frame_key(path: Path):
    if not path.stem.isdigit():
        raise ValueError(f"Frame name must have a numeric stem, got: {path.name}")
    return int(path.stem)


def list_numbered_frames(input_dir: Path, strict_numbering: bool):
    frames = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.isdigit()
    ]
    frames.sort(key=numeric_frame_key)

    if not frames:
        raise RuntimeError(
            f"No numbered image frames found in {input_dir}. Expected names like 1.png, 2.png, ..."
        )

    ids = [numeric_frame_key(frame) for frame in frames]
    if strict_numbering:
        expected = list(range(1, len(frames) + 1))
        if ids != expected:
            raise ValueError(
                "Sparse input frames must be contiguous and start at 1 when "
                f"strict_numbering=true. Found ids={ids[:20]}..."
            )

    return frames


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
        "height": int(rgb.shape[0]),
        "width": int(rgb.shape[1]),
    }


def validate_compatible_frames(image_infos):
    first = image_infos[0]
    for info in image_infos[1:]:
        if info["height"] != first["height"] or info["width"] != first["width"]:
            raise ValueError(
                "All sparse frames must have the same spatial size. "
                f"{first['path'].name} is {(first['height'], first['width'])}, "
                f"but {info['path'].name} is {(info['height'], info['width'])}."
            )


def image_info_to_tensor(image_info, device):
    rgb = image_info["rgb"].astype(np.float32) / image_info["value_scale"]
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def save_float_rgb(rgb_float, out_path: Path, save_mode, output_dtype, value_scale):
    rgb_float = np.clip(rgb_float, 0.0, 1.0)
    if save_mode == "gray":
        arr = rgb_float.mean(axis=2)
    else:
        arr = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2BGR)

    if np.issubdtype(output_dtype, np.integer):
        arr = np.clip(arr * value_scale + 0.5, 0, value_scale).astype(output_dtype)
    else:
        arr = arr.astype(output_dtype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), arr):
        raise RuntimeError(f"Failed to write image: {out_path}")


def save_image_info(image_info, out_path: Path, save_mode, output_dtype, value_scale):
    rgb_float = image_info["rgb"].astype(np.float32) / image_info["value_scale"]
    save_float_rgb(rgb_float, out_path, save_mode, output_dtype, value_scale)


def save_tensor_as_image(tensor, out_path: Path, save_mode, output_dtype, value_scale):
    rgb_float = tensor.detach().clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
    save_float_rgb(rgb_float, out_path, save_mode, output_dtype, value_scale)


def output_name(index: int, output_ext: str):
    if not output_ext.startswith("."):
        output_ext = "." + output_ext
    return f"{index}{output_ext.lower()}"


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Set overwrite: true in the inference yaml if you want to replace it."
            )
        for child in output_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                import shutil

                shutil.rmtree(child)
    output_dir.mkdir(parents=True, exist_ok=True)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format.")


def key_candidates(key: str):
    prefixes = ("model.module.", "model_without_ddp.", "module.", "model.")
    yield key
    for prefix in prefixes:
        if key.startswith(prefix):
            yield key[len(prefix):]


def load_checkpoint(model, ckpt_path: Path):
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()

    matched = {}
    unexpected = []
    mismatched = []
    used_target_keys = set()

    for key, value in state.items():
        if not torch.is_tensor(value):
            unexpected.append(key)
            continue

        target_key = None
        for candidate in key_candidates(key):
            if candidate in model_state:
                target_key = candidate
                break

        if target_key is None:
            unexpected.append(key)
            continue

        if tuple(model_state[target_key].shape) != tuple(value.shape):
            mismatched.append(
                f"{key}: ckpt{tuple(value.shape)} != model{tuple(model_state[target_key].shape)}"
            )
            continue

        matched[target_key] = value
        used_target_keys.add(target_key)

    if not matched:
        raise RuntimeError(f"No checkpoint tensors matched the model: {ckpt_path}")

    result = model.load_state_dict(matched, strict=False)
    print(f"[LOAD] ckpt={ckpt_path}")
    print(
        "[LOAD] matched={} missing={} unexpected={} mismatched={}".format(
            len(matched), len(result.missing_keys), len(unexpected), len(mismatched)
        )
    )
    if result.missing_keys:
        print(f"[LOAD] first_missing={list(result.missing_keys)[:10]}")
    if unexpected:
        print(f"[LOAD] first_unexpected={unexpected[:10]}")
    if mismatched:
        print(f"[LOAD] first_mismatched={mismatched[:10]}")

    return {
        "matched": len(matched),
        "missing": list(result.missing_keys),
        "unexpected": unexpected,
        "mismatched": mismatched,
        "used_target_keys": sorted(used_target_keys),
    }


def resolve_checkpoint(infer_cfg, model_cfg, infer_cfg_path: Path, model_cfg_path: Path, override_ckpt):
    ckpt_value = override_ckpt or infer_cfg.get("ckpt") or model_cfg.get("resume") or model_cfg.get("pretrained")
    if ckpt_value is None:
        raise ValueError("No checkpoint path found. Set ckpt in the inference yaml or pass --ckpt.")

    ckpt_path = Path(str(ckpt_value)).expanduser()
    if ckpt_path.is_absolute():
        resolved = ckpt_path
    else:
        candidates = [
            infer_cfg_path.parent / ckpt_path,
            model_cfg_path.parent / ckpt_path,
            Path.cwd() / ckpt_path,
        ]
        resolved = next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())

    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def build_global_context(image_infos, device):
    clip = torch.stack([image_info_to_tensor(info, device).squeeze(0) for info in image_infos], dim=0)
    clip = clip.unsqueeze(0)
    seq_len = clip.shape[1]
    frame_mask = torch.ones((1, seq_len), device=device, dtype=torch.float32)
    frame_positions = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32).view(1, seq_len)
    frame_positions = frame_positions / float(max(seq_len, 1))
    return clip, frame_mask, frame_positions


def build_model_inputs(
    image_infos,
    global_clip,
    global_frame_mask,
    global_frame_positions,
    left_index,
    right_index,
    t,
    device,
):
    img0 = image_info_to_tensor(image_infos[left_index], device)
    img1 = image_info_to_tensor(image_infos[right_index], device)
    time_step = torch.tensor([[[float(t)]]], device=device, dtype=torch.float32)
    target_local_position = torch.tensor([[float(t)]], device=device, dtype=torch.float32)
    global_anchor_indices = torch.tensor([[left_index, right_index]], device=device, dtype=torch.long)
    target_global_position = torch.tensor(
        [[(float(left_index) + 1.0 + float(t)) / float(max(global_clip.shape[1], 1))]],
        device=device,
        dtype=torch.float32,
    )

    return {
        "img0": img0,
        "img1": img1,
        "time_step": time_step,
        "run_with_gt": False,
        "global_clip": global_clip,
        "global_frame_mask": global_frame_mask,
        "global_frame_positions": global_frame_positions,
        "global_anchor_indices": global_anchor_indices,
        "target_global_position": target_global_position,
        "target_local_position": target_local_position,
        "sparse_sequence_clip": global_clip,
        "sparse_sequence_frame_mask": global_frame_mask,
        "sparse_sequence_frame_positions": global_frame_positions,
        "sparse_sequence_anchor_indices": global_anchor_indices,
    }


def to_gif_frame(path: Path, max_size=None):
    info = read_image(path)
    rgb = info["rgb"]
    if np.issubdtype(rgb.dtype, np.integer):
        max_value = float(np.iinfo(rgb.dtype).max)
        frame = np.clip(rgb.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)
    elif np.issubdtype(rgb.dtype, np.floating):
        frame = np.clip(rgb, 0.0, 1.0)
        frame = (frame * 255.0 + 0.5).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image dtype for GIF: {rgb.dtype}")

    if max_size is not None:
        max_size = int(max_size)
        h, w = frame.shape[:2]
        scale = min(1.0, float(max_size) / float(max(h, w)))
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def save_gif(frame_paths, gif_path: Path, fps: float, max_size=None):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[WARN] imageio is not available, skip GIF: {gif_path} ({exc})")
        return False

    if fps <= 0:
        raise ValueError("gif_fps must be > 0.")

    frames = [to_gif_frame(path, max_size=max_size) for path in frame_paths]
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(gif_path), frames, duration=1.0 / float(fps))
    print(f"[SAVE] gif={gif_path}")
    return True


def load_model_config(infer_cfg, infer_cfg_path: Path, override_model_cfg):
    model_cfg_value = override_model_cfg or infer_cfg.get("model_cfg")
    if model_cfg_value is None:
        if "model" not in infer_cfg:
            raise ValueError("Inference yaml must contain model_cfg or an inline model block.")
        return infer_cfg, infer_cfg_path

    model_cfg_path = resolve_path(model_cfg_value, cfg_path=infer_cfg_path, must_exist=True)
    return load_yaml(model_cfg_path), model_cfg_path


def apply_cli_overrides(infer_cfg, args):
    if args.input_dir is not None:
        infer_cfg["input_dir"] = args.input_dir
    if args.output_dir is not None:
        infer_cfg["output_dir"] = args.output_dir
    if args.interp_num is not None:
        infer_cfg["interp_num"] = args.interp_num
    if args.device is not None:
        infer_cfg["device"] = args.device
    if args.fp16:
        infer_cfg["fp16"] = True
    return infer_cfg


def autocast_context(fp16: bool, device):
    if fp16 and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def ensure_cuda_for_bim_vfi(device):
    if device.type == "cuda" and torch.cuda.is_available():
        return
    raise RuntimeError(
        "This BiM-VFI/ADAMS inference path uses a CUDA/CuPy cost-volume kernel. "
        "Please run with an NVIDIA GPU and a CUDA-enabled PyTorch build. "
        "You can check the current environment with: "
        "python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\""
    )


@torch.no_grad()
def main():
    args = parse_args()
    infer_cfg_path = Path(args.cfg).resolve()
    infer_cfg = apply_cli_overrides(load_yaml(infer_cfg_path), args)

    input_dir = resolve_path(infer_cfg.get("input_dir"), cfg_path=infer_cfg_path, must_exist=True)
    if input_dir is None or not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir does not exist or is not a directory: {input_dir}")

    interp_num = int(infer_cfg.get("interp_num", 0))
    if interp_num < 0:
        raise ValueError("interp_num must be >= 0.")

    output_subdir = str(infer_cfg.get("output_subdir", "full_seq"))
    output_dir = resolve_output_dir(infer_cfg.get("output_dir"), input_dir, output_subdir)
    output_ext = str(infer_cfg.get("output_ext", ".png"))
    overwrite = bool(infer_cfg.get("overwrite", False))
    strict_numbering = bool(infer_cfg.get("strict_numbering", True))
    fp16 = bool(infer_cfg.get("fp16", False))
    device = select_device(infer_cfg.get("device", "auto"))

    frames = list_numbered_frames(input_dir, strict_numbering=strict_numbering)
    if interp_num > 0 and len(frames) < 2:
        raise ValueError("Need at least two sparse frames when interp_num > 0.")

    image_infos = [read_image(path) for path in frames]
    validate_compatible_frames(image_infos)

    prepare_output_dir(output_dir, overwrite=overwrite)
    print(f"[INFO] cfg={infer_cfg_path}")
    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] sparse_frames={len(frames)} interp_num={interp_num}")
    print(f"[INFO] device={device} fp16={fp16}")

    save_reference = image_infos[0]
    save_mode = save_reference["mode"]
    output_dtype = save_reference["dtype"]
    value_scale = save_reference["value_scale"]

    model = None
    model_cfg_path = None
    ckpt_path = None
    load_info = None
    global_clip = global_frame_mask = global_frame_positions = None

    if interp_num > 0:
        ensure_cuda_for_bim_vfi(device)
        from modules.components import make_components

        model_cfg, model_cfg_path = load_model_config(infer_cfg, infer_cfg_path, args.model_cfg)
        ckpt_path = resolve_checkpoint(infer_cfg, model_cfg, infer_cfg_path, model_cfg_path, args.ckpt)

        model = make_components(model_cfg["model"])
        model.to(device)
        model.eval()
        load_info = load_checkpoint(model, ckpt_path)
        global_clip, global_frame_mask, global_frame_positions = build_global_context(image_infos, device)

    metadata = {
        "cfg": str(infer_cfg_path),
        "model_cfg": None if model_cfg_path is None else str(model_cfg_path),
        "ckpt": None if ckpt_path is None else str(ckpt_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sparse_frames": [path.name for path in frames],
        "interp_num": interp_num,
        "output_ext": output_ext,
        "load_info": load_info,
        "frames": [],
    }

    full_frame_paths = []
    output_index = 1

    first_out = output_dir / output_name(output_index, output_ext)
    save_image_info(image_infos[0], first_out, save_mode, output_dtype, value_scale)
    full_frame_paths.append(first_out)
    metadata["frames"].append(
        {
            "output": first_out.name,
            "kind": "anchor",
            "source": frames[0].name,
            "sparse_index": 1,
        }
    )
    print(f"[SAVE] anchor {frames[0].name} -> {first_out.name}")
    output_index += 1

    for left_index in range(len(frames) - 1):
        right_index = left_index + 1

        for offset in range(1, interp_num + 1):
            t = float(offset) / float(interp_num + 1)
            out_path = output_dir / output_name(output_index, output_ext)
            model_inputs = build_model_inputs(
                image_infos=image_infos,
                global_clip=global_clip,
                global_frame_mask=global_frame_mask,
                global_frame_positions=global_frame_positions,
                left_index=left_index,
                right_index=right_index,
                t=t,
                device=device,
            )
            with autocast_context(fp16, device):
                pred = model(**model_inputs)["imgt_pred"]
            save_tensor_as_image(pred, out_path, save_mode, output_dtype, value_scale)
            full_frame_paths.append(out_path)
            metadata["frames"].append(
                {
                    "output": out_path.name,
                    "kind": "interpolated",
                    "left_source": frames[left_index].name,
                    "right_source": frames[right_index].name,
                    "time_step": t,
                }
            )
            print(
                "[SAVE] pred {} -> {} t={:.6f}".format(
                    f"{frames[left_index].name}/{frames[right_index].name}",
                    out_path.name,
                    t,
                )
            )
            output_index += 1

        anchor_out = output_dir / output_name(output_index, output_ext)
        save_image_info(image_infos[right_index], anchor_out, save_mode, output_dtype, value_scale)
        full_frame_paths.append(anchor_out)
        metadata["frames"].append(
            {
                "output": anchor_out.name,
                "kind": "anchor",
                "source": frames[right_index].name,
                "sparse_index": right_index + 1,
            }
        )
        print(f"[SAVE] anchor {frames[right_index].name} -> {anchor_out.name}")
        output_index += 1

    metadata_path = output_dir / str(infer_cfg.get("metadata_name", "meta.json"))
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] metadata={metadata_path}")

    gif_fps = float(infer_cfg.get("gif_fps", 8))
    gif_max_size = infer_cfg.get("gif_max_size", None)
    if bool(infer_cfg.get("save_sparse_gif", True)):
        save_gif(frames, output_dir / str(infer_cfg.get("sparse_gif_name", "sparse_seq.gif")), gif_fps, gif_max_size)
    if bool(infer_cfg.get("save_full_gif", True)):
        save_gif(full_frame_paths, output_dir / str(infer_cfg.get("full_gif_name", "full_seq.gif")), gif_fps, gif_max_size)

    print(f"[DONE] total_frames={len(full_frame_paths)} save_dir={output_dir}")


if __name__ == "__main__":
    main()
