# Spectrum Patcher (Max) — ComfyUI Model Patch

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

## Parameters

- `window_size`: run a real model pass every Nth step, forecast others
- `m`: Chebyshev basis count / polynomial degree
- `lam`: ridge regularization strength for coefficient fitting
- `w`: blend factor between forecast and last real output on forecast steps

## Notes

This implementation forecasts the **model output tensor** (noise/eps-style output) based on a short history of real evaluations. It is conservative by design: if the fit fails or produces NaNs/Infs, it falls back to a real model call.
