# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMI-DSA is a PyTorch video frame interpolation model for sparse-sampling Digital Subtraction Angiography (DSA). It builds on BiM-VFI (CVPR 2025) and adds sequence-context modulation via four novel components: ADAMS encoder, DMSE motion-state estimator, BiMFN optical-flow corrector, and ASD anatomical decoder. The codebase also includes baseline models (BiM-VFI, MoSt-DSA, GaraMoSt) for comparison.

## Environment & Commands

```bash
# Environment setup
conda create -n smi_dsa python=3.11
conda activate smi_dsa
pip install basicsr-fixed Ipython torchsummary wandb moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch==2.4.1 torchvision==0.19.1
conda install cupy -c conda-forge
```

```bash
# Single-GPU training
python main.py --cfg cfgs/train_and_test_adams_dmse_asd.yaml

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/train_and_test_adams_dmse_asd.yaml

# With wandb logging (requires wandb.yaml with api_key, project, entity)
python main.py --cfg cfgs/train_and_test_adams_dmse_asd.yaml -w

# Benchmark evaluation
python main.py --cfg cfgs/train_and_test_adams_dmse_asd_test.yaml

# Sparse DSA sequence inference
python infer_adams_sparse_sequence.py --cfg cfgs/infer_seq.yaml

# Demo on custom videos
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

**CLI arguments:** `--cfg <path>` (required), `--name/-n` (exp name override), `--tag` (suffix), `--load-root` (data root, default `data`), `--save-root` (output root, default `save`), `--wandb-upload/-w`, `--port-offset/-p`, `--cudnn`.

## Architecture

### Entry & Config Flow

```
main.py → parse_args() → make_cfg() → init_experiment() → init_distributed_mode() → Trainer(cfgs)
```

Configs are YAML files under `cfgs/`. `$load_root$` in paths is substituted from `--load-root`. The config `mode` field (`train`/`validate`/`benchmark`/`demo`) determines which `Trainer` method runs. Config names encode the experiment: `train_and_test_adams_dmse_asd.yaml` is the full SMI-DSA pipeline.

### Registry Pattern

Four registries map string names to classes via `@register('name')`:
- `modules/models/models.py` — top-level model wrappers
- `modules/components/components.py` — the actual neural network (`make_components()`)
- `modules/loss.py` — loss functions
- `datasets/` — dataset classes (implicitly via `__init__.py`)

All YAML `name`/`args` blocks are instantiated through these registries.

### Model Hierarchy (two layers)

1. **`modules/models/bim_vfi.py::BiMVFI(BaseModel)`** — training/validation logic: loss computation, metric logging, checkpointing, DDP wrapping. Has `train_one_step()`, `validate()`, `benchmark()`.
2. **`modules/components/bim_vfi/bim_vfi.py::BiMVFI(nn.Module)`** — the actual neural network. All feature flags (`use_adams_encoder`, `use_dmse`, `use_asd`, `use_tcar`, `use_teacher_branch`, etc.) are constructor args controlled by the YAML `model.args`.

### SMI-DSA Forward Pass (lowest pyramid level)

```
img0, img1, time_step
    │
    ├─ ADAMS encoder (if use_adams_encoder=True)
    │   └─ DASME: dynamic/anatomical context pyramids + global tokens
    │       from the full sparse sequence
    │
    ├─ DMSE (if use_dmse=True, replaces MTMC/BiMPriorPredictor)
    │   └─ Predicts gated (ρ, φ) motion state from dynamic context
    │       rho = (1-gate)·τ + gate·ρ_candidate
    │       phi = Norm((1-gate)·u₀ + gate·u_candidate)
    │
    ├─ BiMFN (always used, the optical-flow corrector / OFC)
    │   └─ Cost volume (9×9 CuPy kernel) → flow+occ encoding →
    │       BiMMConv modulation with rho/phi → 4-channel flow residual
    │
    ├─ CAUN (flow upsampler)
    │   └─ Coarse-to-fine using anatomical features (cfeat0_pyr, cfeat1_pyr)
    │
    └─ ASD / TCAR / SN (decoder, mutually exclusive)
        └─ Warp anchors → fuse anatomical context → predict image residual
           → occlusion-weighted blend → final interpolated frame
```

**Teacher branch (KDVCF):** When `use_teacher_branch=True` and `imgt` is provided, a parallel pass uses the ground-truth target to compute "teacher" BiM values (r_gt, phi_gt) for distillation. Losses compare student vs teacher BiM predictions. This is a key training mechanism from the original BiM-VFI.

### Key Components

| File | Component | Role |
|------|-----------|------|
| `adams_encoder.py` | ADAMS / DASME | Sequence encoder with stepwise multi-scale scheduling; dynamic + anatomical branches; local interval mixing + biased global token attention + Token-to-map FiLM modulation |
| `dmse.py` | DMSE | Gated residual motion-state estimator: predicts (ρ, φ) from anchor features + dynamic context + global tokens. Gate controls deviation from uniform prior. |
| `bimfn.py` | BiMFN / OFC | Cost volume → correlation encoding → flow/occ embedding → BiMMConv(ρ,φ) modulation → 4-ch flow residual. **This is where cost volume adaptation would be integrated.** |
| `costvol.py` | costvol_func | Raw CuPy CUDA kernel for 9×9 correlation cost volume. Fixed symmetric search window. |
| `caun.py` | CAUN | Content-Aware Upsampling Network: upscales low-res bidirectional flow using anatomical anchor features |
| `asd.py` | ASD | Anatomical Structure Decoder: warp anchors → add DASME anatomical context (zero-init adapters) → predict image + occlusion residual |
| `tcar.py` | TCAR | Alternative decoder (mutually exclusive with ASD), can accept global structure tokens |
| `sn.py` | SynthesisNetwork variants | Standard UNet, Deep SN, EAMamba SN, TransUNet SN. Used when ASD/TCAR are off. |
| `backwarp.py` | backwarp | `F.grid_sample`-based differentiable image warping with coordinate caching |
| `resnet_encoder.py` | ResNetPyramid | Shared CNN feature extractor (mfe = dynamic stem, cfe = anatomical stem). ADAMS reuses these stems. |

### Cost Volume (costvol.py) — Important for Modifications

Uses raw CUDA kernels via CuPy's `RawModule`. The kernel template system:
- `{{variable}}` — replaced with tensor dtype, integer values, etc.
- `SIZE_N(tensor)` — replaced with `tensor.size(N)`
- `OFFSET_N(tensor, idx...)` — replaced with linear index expression using tensor strides
- `VALUE_N(tensor, idx...)` — replaced with tensor element access

Two kernels: `costvol_out` (forward — dot product over 9×9 window), `costvol_onegrad` + `costvol_twograd` (backward). The class is a `torch.autograd.Function` with `@torch.cuda.amp.custom_fwd/bwd`.

**To modify the search region:** The current kernel uses fixed `intOy/intOx` loops around `(intY, intX)`. Making it deformable requires either: (a) passing per-pixel offsets and reading `tenTwo` at offset positions with bilinear interpolation, or (b) replacing the CUDA kernel with a PyTorch `grid_sample`-based implementation (see earlier design discussion).

### Config File Structure

Key YAML sections:
- `model.args` — all feature flags and component configs
- `train_dataset` / `test_dataset` — name + args + loader (batch_size, num_workers)
- `loss` — list of `{name, args}` blocks
- `optimizer` / `lr_scheduler` — standard PyTorch names
- `max_epoch`, `validate_every`, `save_every`, `seed`
- `pretrained` — path to checkpoint for weight initialization (partial loading supported)

The dataset `video_sequence_global_context_folder` is the primary dataset for SMI-DSA training. It emits `global_clip`, `global_frame_mask`, `global_anchor_indices` for ADAMS, plus `img0`, `img1`, `imgt`, and `time_step`.

### Distributed Training

Uses PyTorch DDP with `DistributedDataParallel`. The Trainer wraps the model in DDP when `cfgs['distributed']` is True. `model_without_ddp` always points to the unwrapped module. Validation metrics are synced across ranks via file-based sync (`_sync_validation_metrics_by_file`).

### Freeze/Unfreeze Mechanism

YAML supports `freeze_modules` (list of dot-separated paths like `"mfe"`, `"cfe"`, `"bimfn"`) to freeze at startup, and `unfreeze_epoch` (dict of `module_path: epoch`) to unfreeze at specific epochs. Also supports `freeze_param_prefixes` / `unfreeze_param_prefixes` for prefix-based parameter freezing.

## Files You Should Not Modify

- `pretrained/` — model weights, gitignored
- `save/`, `test_outputs/`, `unified_results_n3_p0/` — outputs, gitignored
- `000 article/`, `000sh/` — research notes, gitignored
- `assets/` — demo images, committed but treated as static
