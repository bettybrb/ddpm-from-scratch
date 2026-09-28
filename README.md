# DDPM Scheduler from Scratch

### Implementing the diffusion equations directly and validating them with a pretrained noise-prediction network

A PyTorch project exploring the mechanics of **Denoising Diffusion Probabilistic Models (DDPMs)** by implementing the forward-noising and reverse-sampling schedulers directly from the underlying equations.

The neural noise predictor itself is not trained from scratch: the reconstruction experiment uses the pretrained Hugging Face **`google/ddpm-church-256` U-Net**. The custom contribution is the diffusion scheduling logic around that model.

## What I implemented

- forward diffusion / q-sampling;
- linear beta and alpha schedules;
- cumulative alpha products;
- timestep-dependent Gaussian noise addition;
- reverse diffusion / p-sampling;
- posterior variance calculation;
- clean-sample reconstruction from predicted noise;
- posterior mean coefficients;
- stochastic previous-sample generation;
- iterative denoising over 1,000 timesteps;
- comparison with Hugging Face Diffusers `DDPMScheduler`.

## Forward diffusion

The custom `CustomDDPMScheduler_q` implements the DDPM forward process:

```text
x_0 -> x_1 -> x_2 -> ... -> x_T
```

For an arbitrary timestep, the scheduler combines the original sample with Gaussian noise using the cumulative alpha schedule.

The implementation computes:

- `alpha_t = 1 - beta_t`;
- cumulative alpha products;
- `sqrt(alpha_bar_t)`;
- `sqrt(1 - alpha_bar_t)`;
- the noisy sample `x_t`.

## Reverse diffusion

The custom `CustomDDPMScheduler_p` implements the reverse update used to move from `x_t` toward `x_{t-1}`.

At each of the 1,000 reverse timesteps:

1. the pretrained U-Net predicts the noise residual;
2. the scheduler estimates the original clean sample;
3. posterior mean coefficients are calculated;
4. the previous sample is estimated;
5. stochastic variance noise is added where required.

```text
Gaussian noise
      ↓
pretrained U-Net predicts ε
      ↓
custom DDPM scheduler step
      ↓
x_t -> x_(t-1)
      ↓
repeat for 1,000 timesteps
      ↓
reconstructed image
```

## Reference implementation comparison

The same noisy starting sample is also reconstructed using Hugging Face Diffusers `DDPMScheduler`.

This provides a direct reference pipeline for visually comparing the custom reverse-diffusion implementation with a library scheduler while using the same pretrained U-Net.

## Pretrained model

The experiment uses:

```text
google/ddpm-church-256
```

Its U-Net performs the learned noise prediction. This repository focuses on implementing and understanding the **probabilistic scheduling equations**, rather than training the image-generation network itself.

## DDPM vs DDIM

The notebook also discusses the difference between DDPM and DDIM sampling.

DDPM uses a stochastic reverse process and normally traverses the full diffusion schedule. DDIM can follow a deterministic trajectory and skip timesteps, enabling faster inference.

## Repository structure

```text
ddpm-from-scratch/
├── ddpm_from_scratch.ipynb
├── requirements.txt
└── README.md
```

## Reproducibility

The original forward-noising experiment uses an input `tensor.npy` file that is not distributed in the repository.

The pretrained U-Net is downloaded automatically from Hugging Face when the notebook runs. The original implementation assumes a CUDA-capable environment.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tech

**Python · PyTorch · Hugging Face Diffusers · DDPM · diffusion models · probabilistic sampling · CUDA · NumPy**
