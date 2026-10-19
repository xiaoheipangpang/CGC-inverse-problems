# [IEEE TIP 2026] Control Generation Co-design of Flow Matching for Inverse Problems of Image Processing

This is the official PyTorch implementation of our paper: **"Control Generation Co-design of Flow Matching for Inverse Problems of Image Processing"**.

## Introduction

The prevailing paradigm for solving image inverse problems with generative models follows a two-stage, decoupled strategy: "train the generative prior first, then impose external guidance." To overcome the limitations of this paradigm, we propose a novel **Control-Generation Co-design (CGC)** paradigm. We prove that straight trajectories correspond to constant-speed geodesics in Wasserstein space, and the resulting quasilinear dynamics dramatically simplify the optimal control formulation. CGC outperforms state-of-the-art methods in both reconstruction quality (PSNR, SSIM, LPIPS, FID) and computational efficiency (inference time and memory footprint).

## Environment Setup

We recommend using Anaconda to manage the environment.
```bash
conda create -n cgc python=3.10
conda activate cgc
pip install -r requirements.txt

## 📦 Pretrained Models & Datasets

Before running the code, please prepare the pretrained models and datasets.

### 1. Pretrained Models

Please download the pretrained models and place them in the `checkpoints/` directory. 

*   **Pixel-space prior (RF++)**: https://pan.baidu.com/s/1ckzRs7lVbzWCyVXKIYWYZg?pwd=smpx
*   **Latent-space prior (SD3)**: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
    *   *Alternatively, you can modify the default path in `large_scale/sd3_sampler.py` to your local model path.*

### 2. Datasets

Please organize your datasets in the `data/` directory. The expected directory structure is as follows:

```text
data/
├── AFHQ512/
├── Celeba512/
├── DIV2K1024/
├── FFHQ1024/
├── AFHQ64/
└── Celeba64/
