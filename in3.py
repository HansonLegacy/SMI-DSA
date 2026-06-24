import os
import json
import math
import argparse
import importlib
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import contextlib

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Strict SSIM (your code)
# ----------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().to(device)
    return window

def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1):
    # here we enforce val_range=1 (0..1 tensors)
    L = val_range

    padd = 0
    (_, _, height, width) = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)

    # treat C as depth -> [N,1,C,H,W]
    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1) - mu2_sq
    sigma12  = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# ----------------------------
# LPIPS (alex + vgg)
# ----------------------------
try:
    import lpips  # pip install lpips
    _HAS_LPIPS = True
except Exception:
    print("[WARN] lpips import failed. Install with: pip install lpips")
    _HAS_LPIPS = False

_LPIPS_NETS = {}  # cache by name: {'alex': net, 'vgg': net}

def get_lpips_net(net="alex"):
    if not _HAS_LPIPS:
        return None
    if net not in _LPIPS_NETS:
        _LPIPS_NETS[net] = lpips.LPIPS(net=net).to(device).eval()
    return _LPIPS_NETS[net]

@torch.no_grad()
def compute_lpips_0_1(gt_t, pred_t, net="alex"):
    """
    gt_t, pred_t: [1,C,H,W], range [0,1]
    returns float (lower is better) or None
    """
    lpips_net = get_lpips_net(net)
    if lpips_net is None:
        return None

    # LPIPS expects 3ch RGB in [-1,1]
    if gt_t.shape[1] == 1:
        gt_t = gt_t.repeat(1, 3, 1, 1)
    if pred_t.shape[1] == 1:
        pred_t = pred_t.repeat(1, 3, 1, 1)

    gt_in = gt_t * 2.0 - 1.0
    pred_in = pred_t * 2.0 - 1.0

    d = lpips_net(gt_in, pred_in)
    return float(d.mean().detach().cpu().item())

# ----------------------------
# Utils (I/O)
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_image(path, force_gray=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if force_gray and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def img_to_tensor_0_1(img_np):
    # [H,W] or [H,W,3] -> [1,C,H,W] float 0..1
    x = torch.from_numpy(img_np).to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0)  # [1,H,W]
    else:
        x = x.permute(2, 0, 1)  # [C,H,W]  (NOTE: cv2 is BGR)
        # if you want true RGB for LPIPS, uncomment next line:
        # x = x[[2,1,0], ...]
    x = x.float() / 255.0
    return x.unsqueeze(0)

def tensor_to_u8_img(t):
    # t: [1,C,H,W] 0..1
    t = t.detach().clamp(0, 1).cpu()[0]
    if t.shape[0] == 1:
        img = (t[0].numpy() * 255.0 + 0.5).astype(np.uint8)
    else:
        img = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        # if your network outputs RGB, convert to BGR for cv2 save:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def pad_to_32(x):
    _, _, h, w = x.shape
    ph = (32 - (h % 32)) % 32
    pw = (32 - (w % 32)) % 32
    if ph == 0 and pw == 0:
        return x, (0, 0, 0, 0)
    xpad = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return xpad, (0, pw, 0, ph)

def unpad(x, pad):
    l, r, t, b = pad
    if r == 0 and b == 0:
        return x
    return x[..., : x.shape[-2] - b, : x.shape[-1] - r]

# ----------------------------
# Wrapper-safe helpers (fix your NameError)
# ----------------------------
def find_inner_module(obj):
    if isinstance(obj, nn.Module):
        return obj
    for name in ["net", "model", "module", "network", "generator"]:
        if hasattr(obj, name) and isinstance(getattr(obj, name), nn.Module):
            return getattr(obj, name)
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            if isinstance(v, nn.Module):
                return v
    return None

def safe_to(obj, dev):
    m = find_inner_module(obj)
    if m is not None:
        m.to(dev)
        return m
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        obj.to(dev)
        return obj
    return obj

def safe_eval(obj):
    m = find_inner_module(obj)
    if m is not None and hasattr(m, "eval"):
        m.eval()
        return m
    if hasattr(obj, "eval") and callable(getattr(obj, "eval")):
        obj.eval()
        return obj
    return obj

def load_ckpt_into(obj, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "params"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    m = find_inner_module(obj)
    target = m if m is not None else obj
    if not hasattr(target, "load_state_dict"):
        raise RuntimeError("No load_state_dict() found on model/wrapper")

    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(f"[OK] loaded ckpt (strict=False), missing={len(missing)}, unexpected={len(unexpected)}")

def pick_tensor(out):
    # BiMVFI.forward common returns: (flow, occ, interp_img, extra, teacher)
    if isinstance(out, (list, tuple)) and len(out) >= 3 and torch.is_tensor(out[2]):
        return out[2]
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for v in out:
            if torch.is_tensor(v):
                return v
            if isinstance(v, (list, tuple, dict)):
                vv = pick_tensor(v)
                if torch.is_tensor(vv):
                    return vv
    if isinstance(out, dict):
        for k in ["pred", "output", "frame", "img", "It", "y"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v
            if isinstance(v, (list, tuple, dict)):
                vv = pick_tensor(v)
                if torch.is_tensor(vv):
                    return vv
    raise RuntimeError(f"Cannot pick tensor from output type={type(out)}")

def get_call_target(obj):
    if callable(obj):
        return obj
    m = find_inner_module(obj)
    if m is not None and callable(m):
        return m
    if hasattr(obj, "forward") and callable(getattr(obj, "forward")):
        return obj.forward
    if m is not None and hasattr(m, "forward") and callable(getattr(m, "forward")):
        return m.forward
    raise RuntimeError(f"No callable/forward found for type={type(obj)}")

def run_model_call(net_or_wrapper, I0, I1, t):
    target = get_call_target(net_or_wrapper)
    candidates = [
        lambda: target(I0, I1, time_step=t),
        lambda: target(I0, I1, t=t),
        lambda: target(I0, I1, timestep=t),
        lambda: target(I0, I1, t),
    ]
    last_e = None
    for fn in candidates:
        try:
            out = fn()
            return pick_tensor(out)
        except Exception as e:
            last_e = e
    raise last_e

# ----------------------------
# Metrics (PSNR/SSIM/LPIPS + residual map)
# ----------------------------
def compute_abs_residual_x10_and_metrics(gt_u8, pred_u8, out_dir, force_gray=False):
    ensure_dir(out_dir)

    if force_gray:
        if gt_u8.ndim == 3:
            gt_u8 = cv2.cvtColor(gt_u8, cv2.COLOR_BGR2GRAY)
        if pred_u8.ndim == 3:
            pred_u8 = cv2.cvtColor(pred_u8, cv2.COLOR_BGR2GRAY)

    if gt_u8.shape != pred_u8.shape:
        raise ValueError(f"Shape mismatch: gt{gt_u8.shape} vs pred{pred_u8.shape}")

    gt_f = gt_u8.astype(np.float32)
    pred_f = pred_u8.astype(np.float32)
    abs_res = np.abs(gt_f - pred_f)

    abs_x10 = np.clip(abs_res * 10.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "abs_residual_x10.png"), abs_x10)

    # strict metrics on uint8->tensor 0..1
    gt_t = img_to_tensor_0_1(gt_u8)
    pred_t = img_to_tensor_0_1(pred_u8)

    diff = gt_t - pred_t
    mse = (diff * diff).mean().detach().cpu().item()
    psnr = float("inf") if mse <= 1e-20 else (-10.0 * math.log10(mse))
    ssimv = float(ssim_matlab(gt_t, pred_t).detach().cpu().item())
    mae = float(diff.abs().mean().detach().cpu().item())

    lpips_alex = compute_lpips_0_1(gt_t, pred_t, net="alex")
    lpips_vgg = compute_lpips_0_1(gt_t, pred_t, net="vgg")

    metrics = {
        "metrics_strict_0_1": {
            "mae": mae,
            "mse": float(mse),
            "psnr": float(psnr),
            "ssim_matlab": float(ssimv),
            "lpips_alex": None if lpips_alex is None else float(lpips_alex),
            "lpips_vgg": None if lpips_vgg is None else float(lpips_vgg),
        }
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"PSNR={psnr}\n")
        f.write(f"SSIM={ssimv}\n")
        f.write(f"LPIPS(alex)={lpips_alex}\n")
        f.write(f"LPIPS(vgg)={lpips_vgg}\n")
        f.write(f"MAE={mae}\n")
        f.write(f"MSE={mse}\n")

    return psnr, ssimv, lpips_alex, lpips_vgg

# ----------------------------
# Main
# ----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/bim_vfi_demo.yaml")
    ap.add_argument("--ckpt", default="pretrained/model_ep1_xieheliu2d_123loss_lr5e5.pth")
    ap.add_argument("--in1", default="testimg/2D2/frame_1.png")
    ap.add_argument("--in2", default="testimg/2D2/frame_5.png")
    ap.add_argument("--out_dir", default="testimg/2D2/model_ep1_xieheliu2d_123loss_lr5e5_test")
    ap.add_argument("--force_gray", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    print("device:", device)
    print("cfg:", args.cfg)
    print("ckpt:", args.ckpt)

    cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise RuntimeError("YAML cfg must load into a dict")

    cfg.setdefault("distributed", False)

    model_factory = importlib.import_module("modules.models.models")
    wrapper = model_factory.make(cfg)

    safe_to(wrapper, device)
    safe_eval(wrapper)

    if args.ckpt and os.path.isfile(args.ckpt):
        load_ckpt_into(wrapper, args.ckpt)
    else:
        print("[WARN] ckpt not found, skip loading:", args.ckpt)

    img1 = read_image(args.in1, force_gray=args.force_gray)
    img2 = read_image(args.in2, force_gray=args.force_gray)

    I0 = img_to_tensor_0_1(img1)
    I1 = img_to_tensor_0_1(img2)

    # model expects 3ch
    if I0.shape[1] == 1:
        I0 = I0.repeat(1, 3, 1, 1)
    if I1.shape[1] == 1:
        I1 = I1.repeat(1, 3, 1, 1)

    I0p, pad = pad_to_32(I0)
    I1p, _ = pad_to_32(I1)

    ts = [0.25, 0.5, 0.75]
    preds = []
    ensure_dir(args.out_dir)

    autocast_ctx = torch.cuda.amp.autocast if (args.fp16 and device.type == "cuda") else contextlib.nullcontext

    with autocast_ctx():
        for t in ts:
            It = run_model_call(wrapper, I0p, I1p, t)
            It = unpad(It, pad)
            preds.append(It)

    pred_paths = []
    for idx, (t, It) in enumerate(zip(ts, preds), start=2):
        out_path = os.path.join(args.out_dir, f"frame_{idx}_pred.png")
        out_img = tensor_to_u8_img(It)
        cv2.imwrite(out_path, out_img)
        pred_paths.append(out_path)
        print("[SAVE]", out_path)

    # eval vs frame_2_gt.png / frame_3_gt.png / frame_4_gt.png
    gt_names = ["frame_2_gt.png", "frame_3_gt.png", "frame_4_gt.png"]
    test_dirs = ["test1", "test2", "test3"]

    for i in range(3):
        gt_path = os.path.join(os.path.dirname(args.in1), gt_names[i])
        pred_path = pred_paths[i]
        out_eval_dir = os.path.join(args.out_dir, test_dirs[i])
        ensure_dir(out_eval_dir)

        gt_u8 = read_image(gt_path, force_gray=args.force_gray)
        pred_u8 = read_image(pred_path, force_gray=args.force_gray)

        cv2.imwrite(os.path.join(out_eval_dir, "gt.png"), gt_u8)
        cv2.imwrite(os.path.join(out_eval_dir, "pred.png"), pred_u8)

        psnr, ssimv, lp_a, lp_v = compute_abs_residual_x10_and_metrics(
            gt_u8=gt_u8,
            pred_u8=pred_u8,
            out_dir=out_eval_dir,
            force_gray=args.force_gray
        )
        print(f"[EVAL] {test_dirs[i]}  PSNR={psnr:.4f} SSIM={ssimv:.6f} LPIPS(alex)={lp_a} LPIPS(vgg)={lp_v} -> {out_eval_dir}")

if __name__ == "__main__":
    main()
