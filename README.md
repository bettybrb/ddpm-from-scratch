# DDPM from Scratch

A PyTorch implementation exploring the mathematical foundations of Denoising Diffusion Probabilistic Models (DDPMs), including forward noise addition and reverse diffusion sampling.

The project implements custom diffusion schedulers and uses a pretrained U-Net to reconstruct images progressively from Gaussian noise.

## Project Overview

Diffusion models generate data by learning to reverse a gradual noising process. This project implements the core DDPM scheduling equations directly rather than treating the diffusion scheduler as a black box.

The implementation covers both directions of the diffusion process:

- forward diffusion (`q`) - progressively adding Gaussian noise to an image
- reverse diffusion (`p`) - progressively estimating and removing noise to reconstruct an image

The custom reverse scheduler is compared against the scheduler provided by Hugging Face Diffusers.

## Forward Diffusion

`CustomDDPMScheduler_q` implements the forward diffusion process.

For a selected timestep, cumulative alpha values from the noise schedule are used to combine the original sample with Gaussian noise, producing the corresponding noisy sample `x_t`.

The implementation includes calculations for:

- the cumulative alpha schedule
- `sqrt(alpha_bar_t)`
- `sqrt(1 - alpha_bar_t)`
- timestep-dependent Gaussian noise addition

## Reverse Diffusion

`CustomDDPMScheduler_p` implements DDPM reverse sampling.

At each timestep, a pretrained U-Net predicts the noise contained in the current sample. The scheduler then estimates the original clean sample and calculates the distribution for the previous timestep.

The reverse process implements:

- posterior variance calculation
- alpha and beta schedule calculations
- prediction of the original clean sample
- posterior mean coefficients
- prediction of the previous sample
- stochastic variance noise
- iterative denoising across 1,000 timesteps

## Pretrained U-Net

The reconstruction experiment uses the Hugging Face `google/ddpm-church-256` pretrained diffusion model.

The U-Net predicts the noise residual at each timestep while the custom scheduler performs the mathematical reverse-diffusion update.

## Validation

The notebook also performs the reconstruction using Hugging Face Diffusers' built-in `DDPMScheduler`.

This provides a reference implementation against which the custom reverse scheduler can be compared.

## DDPM vs DDIM

The accompanying investigation also explores the conceptual difference between DDPM and DDIM sampling.

DDPM uses a stochastic Markovian reverse process and normally traverses the diffusion schedule sequentially. DDIM can use a deterministic non-Markovian trajectory and skip timesteps, enabling substantially faster sampling.

## Repository Structure

- `ddpm_from_scratch.ipynb` - forward and reverse DDPM scheduler implementation and reconstruction experiment
- `requirements.txt` - Python dependencies
- `.gitignore` - generated samples, datasets and local environment files

## Requirements

The experiment requires a CUDA-capable environment because the original implementation places the model and diffusion tensors on CUDA.

An input `tensor.npy` sample is also required to reproduce the original forward-noising experiment. This coursework-provided input is not included in the repository.

The pretrained U-Net is downloaded automatically from Hugging Face when the notebook is executed.

## Installation

```bash
pip install -r requirements.txt
```

## Technologies

- Python
- PyTorch
- Hugging Face Diffusers
- NumPy
- Pillow
- CUDA

## Concepts Demonstrated

- Denoising Diffusion Probabilistic Models
- Forward diffusion
- Reverse diffusion
- Gaussian noise schedules
- U-Net noise prediction
- Markov chains
- Probabilistic sampling
- DDPM scheduling equations
- DDPM vs DDIM sampling
- PyTorch tensor operations
- Generative deep learning

## Motivation

Modern diffusion libraries make image generation accessible through high-level APIs, but those APIs can hide the probabilistic process responsible for generation. This project implements the DDPM scheduling equations directly and uses them with a pretrained neural network, providing a practical view of how an image is transformed from noise into a structured sample one timestep at a time.
