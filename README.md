<div align="center">
<h2>SMI-DSA</h2>
<h3>Sequence-Context Modulation for Robust Frame Interpolation in<br>Sparse-Sampling Digital Subtraction Angiography</h3>

<div>
    <a href='https://github.com/HansonLegacy' target='_blank'>Jiaxuan Li</a><sup>*</sup>&nbsp;
    <a href='https://github.com/HansonLegacy' target='_blank'>Ruiheng Zhang</a><sup>*</sup>&nbsp;
    <a href='https://github.com/HansonLegacy' target='_blank'>Huangxuan Zhao</a><sup>* †</sup>&nbsp;
    <a href='https://github.com/HansonLegacy' target='_blank'>Bo Du</a><sup>†</sup>
</div>
<div>
    <sup>*</sup>Equal contribution&nbsp;&nbsp;&nbsp;<sup>†</sup>Corresponding authors
</div>
<div>
    School of Computer Science, Wuhan University, Wuhan, China
</div>
<div>
    <code>{jiaxuanli, ruihengzhang, huangxuanzhao, dubo}@whu.edu.cn</code>
</div>

<br>

<div>
    <a href="https://github.com/HansonLegacy/SMI-DSA" target='_blank'>
    <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
    </a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/HansonLegacy/SMI-DSA">
    <img alt="License" src="https://img.shields.io/badge/License-Research%20%26%20Education%20Only-lightgrey">
</div>

---

<h4>
    Official PyTorch implementation of "SMI-DSA: Sequence-Context Modulation for
    Robust Frame Interpolation in Sparse-Sampling Digital Subtraction Angiography".
</h4>
</div>

## Abstract

Reconstructing undersampled Digital Subtraction Angiography (DSA) sequences via video frame interpolation can effectively reduce radiation exposure. However, existing DSA interpolation methods mostly rely on local two-frame interpolation. Under sparser sampling, larger temporal gaps amplify errors in the linear-motion approximation underlying two-frame interpolation, leading to inaccurate motion prediction and blurred fine structures. To address this, we propose **SMI-DSA**, a sequence-context-modulated model for sparse DSA frame interpolation. Its encoder adopts a stepwise multi-scale scheduling: target-near frames preserve rich multi-scale features to capture local motion and vascular details, while target-far frames are progressively compressed into compact context features to control memory and computation costs. Based on sequence-level dynamic context, we further design a motion estimator that generates an intermediate-frame dynamic motion-state representation and injects it into optical-flow residual prediction to improve motion estimation. Extensive experiments show that SMI-DSA improves motion prediction accuracy and fine-vessel preservation in sparse-sampling DSA scenarios, and significantly outperforms existing methods when recovering five or seven consecutive intermediate frames, achieving state-of-the-art performance.

## Architecture Overview

SMI-DSA builds upon the BiM-VFI (CVPR 2025) bidirectional motion field framework and extends it with four novel components for sequence-level sparse DSA reconstruction:

| Component | Description |
|-----------|-------------|
| **ADAMS Encoder** (DASME) | Dynamic-Anatomical Stepwise Multi-scale Encoder — encodes the full sparse sequence with resolution scheduling, producing target-time dynamic and anatomical context pyramids |
| **DMSE** | Dynamic Motion-State Estimator — predicts a gated intermediate-frame motion state (ρ, φ) from sequence-level context |
| **BiMFN** (OFC) | Optical-Flow Corrector — injects DMSE motion state into bidirectional flow residual prediction via coarse-to-fine correlation |
| **ASD** | Anatomical Structure Decoder — synthesizes the target frame by warping anchor images and calibrating with anatomical context |

## Contents
- [Contents](#contents)
- [Environment Setting](#environment-setting)
- [Dataset](#dataset)
  - [Download](#download)
  - [Preparation](#preparation)
- [Pretrained Model](#pretrained-model)
- [Evaluation](#evaluation)
- [Training](#training)
- [Demo](#demo)
- [License](#license)
- [Citation](#citation)

## Environment Setting
```bash
conda create -n smi_dsa python=3.11
conda activate smi_dsa
pip install basicsr-fixed Ipython torchsummary wandb moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch==2.4.1 torchvision==0.19.1
conda install cupy -c conda-forge
```

## Dataset
### Download
The model is evaluated on DSA sequences as well as standard VFI benchmarks:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)
> - [SNU-FILM](https://myungsub.github.io/CAIN/)
> - [SNU-FILM-arb](https://drive.google.com/drive/folders/1Kp1JLP9CCSDG-dhj2jZ-nuB0_plXnzSt?usp=drive_link)
> - [X4K1000FPS](https://www.dropbox.com/scl/fo/88aarlg0v72dm8kvvwppe/AHxNqDye4_VMfqACzZNy5rU?rlkey=a2hgw60sv5prq3uaep2metxcn&e=1&dl=0)

### Preparation
For SNU-FILM and SNU-FILM-arb datasets, move `test-[easy, medium, hard, extreme].txt` and `test-arb-[medium, hard, extreme].txt` to `<PATH_TO_SNU_FILM>/eval_modes` directory.

## Pretrained Model
Pre-trained weights are available for download. *(Link to be updated)*

## Evaluation
Configure the desired benchmark in `benchmark_dataset` section of a test config (e.g., `cfgs/train_and_test_adams_dmse_asd_test.yaml`):

```bash
python main.py --cfg cfgs/train_and_test_adams_dmse_asd_test.yaml
```

For sparse DSA sequence inference with the full ADAMS + DMSE + ASD pipeline:
```bash
python infer_adams_sparse_sequence.py --cfg cfgs/infer_seq.yaml
```

## Training
Single GPU:
```bash
python main.py --cfg cfgs/train_and_test_adams_dmse_asd.yaml
```

Multi-GPU (e.g., 4 GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/train_and_test_adams_dmse_asd.yaml
```

## Demo
Custom videos in image sequences or video format can be interpolated as follows.

Set up the demo root directory:
```
demo_root/
├── video1.mp4
├── video2.mp4
├── video3/
│   ├── img0.png
│   ├── img1.png
│   └── ...
└── ...
```

Then configure `cfgs/bim_vfi_demo.yaml` with your data root and run:
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

## License
This project is released for **research and education purposes only**. Any commercial use requires formal permission from the corresponding authors.

## Citation
If you find SMI-DSA useful, please consider citing:
```BibTeX
@article{li2025smidsa,
  title     = {SMI-DSA: Sequence-Context Modulation for Robust Frame Interpolation
               in Sparse-Sampling Digital Subtraction Angiography},
  author    = {Li, Jiaxuan and Zhang, Ruiheng and Zhao, Huangxuan and Du, Bo},
  journal   = {arXiv preprint},
  year      = {2025}
}
```

## Acknowledgement
This codebase builds upon [BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI) (CVPR 2025) by Wonyong Seo, Jihyong Oh, and Munchurl Kim. We thank the authors for their excellent work.
