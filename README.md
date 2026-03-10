# Spectrum Patcher — ComfyUI Model Patch
<img width="1200" height="720" alt="spectrum-patcher-guide" src="https://github.com/user-attachments/assets/17b6c2f6-a7e7-418f-b117-dc548fbf5532" />

Training-free diffusion acceleration via Chebyshev spectral forecasting (in the style of *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*).

Original Spectrum implementation/research repo:
https://github.com/hanjq17/Spectrum

This custom node is a **MODEL patcher**: it wraps the model UNet/DiT forward via `set_model_unet_function_wrapper(...)` so it can work with normal ComfyUI samplers (Euler, DPM++, etc.).

## Install

Clone into `ComfyUI/custom_nodes`.

Recommended (clone into a correctly-named folder):

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/maximilianwicen/ComfyUI-Node-for-Adaptive-Spectral-Feature-Forecasting-for-Diffusion-Sampling-Acceleration.git ComfyUI-Max-Spectrum
```

Then restart ComfyUI.

## Node

- Menu name: **Spectrum Patcher (Max)**
- Category: `model/patches`
- Internal id: `MaxSpectrumPatcher`

## Compatible Models

Works with any model that uses ComfyUI's standard MODEL patcher interface, including:

- **Standard diffusion** — SD 1.5, SDXL, and similar eps/v-prediction models
- **Rectified-flow / flow-matching** — Z-Image Turbo, Flux, SD3, SD3.5-Large
- **Video DiTs** — HunyuanVideo, Wan2.1 (untested, community feedback welcome)

## Parameters

- `window_size`: run a real model pass every Nth step, forecast others
- `m`: Chebyshev basis count / polynomial degree
- `lam`: ridge regularization strength for coefficient fitting
- `w`: blend factor between forecast and last real output on forecast steps
- `model_type`: `auto` (default) detects the sigma range at runtime; use `flow` to force rectified-flow mode (Z-Image Turbo, Flux, SD3) or `eps` to force standard diffusion mode

## Z-Image Turbo usage

Z-Image Turbo is a 6B-parameter rectified-flow DiT from Alibaba Tongyi Lab. It uses a **linear sigma schedule (1.0 → 0.0)** and is distilled for ~8 sampling steps. Recommended settings:

- `model_type`: `flow` (or leave on `auto` — will be detected automatically)
- `window_size`: `2` (gives ~4 real passes across 8 steps)
- `w`: `0.5–0.8`
- `m`: `3–4`

Use with a standard linear scheduler (e.g. ComfyUI's `linear` schedule) and an Euler or flow-compatible sampler.

## Notes

This implementation forecasts the **model output tensor** (noise/velocity/flow-prediction output) based on a short history of real evaluations. It is conservative by design: if the fit fails or produces NaNs/Infs, it falls back to a real model call.
