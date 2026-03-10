import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


def _to_float_timestep(timestep) -> float:
    if torch.is_tensor(timestep):
        return float(timestep.flatten()[0].item())
    return float(timestep)


def _is_flow_schedule(c: dict) -> bool:
    """Return True if the sigma schedule looks like a rectified-flow model (Z-Image Turbo, Flux, SD3, etc.).

    These models use sigmas in [0, 1] with a near-linear schedule, unlike standard
    diffusion models (SD 1.5, SDXL) whose sigmas reach ~14+.
    """
    try:
        t_opts = c.get("transformer_options", None) if isinstance(c, dict) else None
        if not isinstance(t_opts, dict):
            return False
        sigmas = t_opts.get("sample_sigmas", None)
        if sigmas is None:
            sigmas = t_opts.get("sigmas", None)
        if torch.is_tensor(sigmas) and sigmas.numel() > 1:
            s_max = float(sigmas.detach().flatten().float().max().item())
            return s_max <= 1.05
    except Exception:
        pass
    return False


def _schedule_minmax_from_c(c: dict) -> Optional[Tuple[float, float]]:
    try:
        t_opts = c.get("transformer_options", None)
        if not isinstance(t_opts, dict):
            return None

        sigmas = t_opts.get("sample_sigmas", None)
        if sigmas is None:
            sigmas = t_opts.get("sigmas", None)

        if torch.is_tensor(sigmas) and sigmas.numel() > 1:
            s = sigmas.detach().flatten().float()
            return float(s.min().item()), float(s.max().item())
    except Exception:
        return None
    return None


def _normalize_to_unit_interval(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    denom = (t_max - t_min)
    if abs(denom) < 1e-12:
        return torch.zeros_like(t)
    x = (t - t_min) / denom
    return x


def _normalize_to_chebyshev_domain(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    x01 = _normalize_to_unit_interval(t, t_min=t_min, t_max=t_max)
    x = x01 * 2.0 - 1.0
    return x.clamp(-1.0, 1.0)


def _chebyshev_design_matrix(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Return Phi where Phi[i, k] = T_k(x_i), k in [0, degree-1]."""
    if degree <= 0:
        raise ValueError("degree must be >= 1")
    x = x.reshape(-1)
    n = x.shape[0]

    phi = x.new_empty((n, degree))
    phi[:, 0] = 1.0
    if degree == 1:
        return phi

    phi[:, 1] = x
    for k in range(2, degree):
        phi[:, k] = 2.0 * x * phi[:, k - 1] - phi[:, k - 2]
    return phi


def _fit_ridge(Phi: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """Solve ridge regression: argmin_C ||Phi C - Y||^2 + lam ||C||^2.

    Phi: [N, M]
    Y:   [N, D]
    Returns C: [M, D]
    """
    n, m = Phi.shape
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError("Y must be [N, D] with same N as Phi")

    lam_f = float(lam)
    if lam_f < 0:
        lam_f = 0.0

    # Use float32 math for stability even if model runs in fp16/bf16.
    Phi32 = Phi.float()
    Y32 = Y.float()

    # (Phi^T Phi + lam I) C = Phi^T Y
    A = Phi32.transpose(0, 1) @ Phi32
    if lam_f > 0:
        A = A + (lam_f * torch.eye(m, device=A.device, dtype=A.dtype))
    else:
        # Still add a tiny diagonal for numerical robustness.
        A = A + (1e-8 * torch.eye(m, device=A.device, dtype=A.dtype))
    B = Phi32.transpose(0, 1) @ Y32
    C = torch.linalg.solve(A, B)
    return C


def _forecast_from_coeffs(C: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """C: [M, D], x: scalar tensor -> y_hat: [D]"""
    m = C.shape[0]
    phi = _chebyshev_design_matrix(x.reshape(1), degree=m).float()  # [1, M]
    y = (phi @ C).reshape(-1)  # [D]
    return y


@dataclass
class _SpectrumState:
    cache_ts: List[float] = field(default_factory=list)
    cache_y: List[torch.Tensor] = field(default_factory=list)
    coeffs: Optional[torch.Tensor] = None  # [M, D] float32
    step_index: int = 0
    last_t: Optional[float] = None
    last_shape: Optional[Tuple[int, ...]] = None
    last_dtype: Optional[torch.dtype] = None
    last_device: Optional[torch.device] = None

    def reset(self):
        self.cache_ts.clear()
        self.cache_y.clear()
        self.coeffs = None
        self.step_index = 0
        self.last_t = None
        self.last_shape = None
        self.last_dtype = None
        self.last_device = None


class MaxSpectrumPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "m": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "lam": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.001}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
                "model_type": (["auto", "eps", "flow"], {"default": "auto",
                    "tooltip": "Model parameterization. 'flow' covers rectified-flow models "
                               "(Z-Image Turbo, Flux, SD3). 'eps' covers standard diffusion "
                               "(SD 1.5, SDXL). 'auto' detects from the sigma schedule."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches"

    def patch(self, model, w: float, m: int, lam: float, window_size: int, model_type: str = "auto"):
        model_clone = model.clone()

        # Keep separate Spectrum state per (cond_or_uncond) signature.
        states: Dict[Tuple[int, ...], _SpectrumState] = {}

        w_f = float(max(0.0, min(1.0, w)))
        degree = int(max(1, m))
        lam_f = float(max(0.0, lam))
        win = int(max(1, window_size))
        forced_flow = model_type == "flow"
        forced_eps = model_type == "eps"

        def _get_state(key: Tuple[int, ...]) -> _SpectrumState:
            st = states.get(key)
            if st is None:
                st = _SpectrumState()
                states[key] = st
            return st

        def unet_wrapper_function(model_function, kwargs):
            # kwargs usually contains: input, timestep, c, cond_or_uncond
            x_in = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs.get("c", {})

            key_list = kwargs.get("cond_or_uncond", [])
            try:
                key = tuple(int(v) for v in key_list)
            except Exception:
                key = ()

            st = _get_state(key)

            t_val = _to_float_timestep(timestep)
            out_dtype = x_in.dtype if torch.is_tensor(x_in) else None
            out_device = x_in.device if torch.is_tensor(x_in) else None

            # Reset if a new sampling run likely started or if tensor layout changed.
            eps = 1e-7
            if st.last_t is not None and (t_val > st.last_t + eps):
                st.reset()
            if torch.is_tensor(x_in):
                shape = tuple(int(s) for s in x_in.shape)
                if st.last_shape is not None and shape != st.last_shape:
                    st.reset()
                if st.last_dtype is not None and out_dtype != st.last_dtype:
                    st.reset()
                if st.last_device is not None and out_device != st.last_device:
                    st.reset()
                st.last_shape = shape
                st.last_dtype = out_dtype
                st.last_device = out_device

            # Step index handling: prefer schedule-based index if available.
            schedule = None
            try:
                t_opts = c.get("transformer_options", {}) if isinstance(c, dict) else {}
                schedule = t_opts.get("sample_sigmas", None)
                if schedule is None:
                    schedule = t_opts.get("sigmas", None)
            except Exception:
                schedule = None

            # Detect model type for correct sigma-domain handling.
            # Rectified-flow models (Z-Image Turbo, Flux, SD3) use sigmas in [0, 1];
            # standard diffusion models (SD 1.5, SDXL) use much larger sigma ranges.
            if forced_flow:
                is_flow = True
            elif forced_eps:
                is_flow = False
            else:
                is_flow = _is_flow_schedule(c)

            if torch.is_tensor(schedule) and schedule.numel() > 1:
                sigmas = schedule.detach().flatten()
                t0 = float(sigmas[0].item())
                # If we jumped back to the first sigma (or higher), treat as reset.
                if st.last_t is not None and abs(t_val - t0) < 1e-8 and (t_val > st.last_t + eps):
                    st.reset()

                # Find closest index; also handle non-exact matches.
                try:
                    target = torch.tensor([t_val], device=sigmas.device, dtype=sigmas.dtype)
                    diff = (sigmas - target).abs()
                    idx = int(diff.argmin().item())
                except Exception:
                    idx = st.step_index
                st.step_index = idx
            else:
                if st.last_t is None:
                    st.step_index = 0
                else:
                    if abs(t_val - st.last_t) > 1e-8:
                        st.step_index += 1

            st.last_t = t_val

            def _resolve_tminmax(extra_t=None):
                """Return (t_min, t_max) for Chebyshev domain normalization.

                Priority: schedule from context → flow-model canonical [0,1] →
                cache-derived range.
                """
                tminmax = _schedule_minmax_from_c(c)
                if tminmax is not None:
                    return tminmax
                # For flow models (Z-Image Turbo, Flux, SD3) fall back to [0, 1].
                if is_flow:
                    return 0.0, 1.0
                pts = list(st.cache_ts)
                if extra_t is not None:
                    pts = pts + [extra_t]
                if not pts:
                    return 0.0, 1.0
                return float(min(pts)), float(max(pts))

            # Decide whether to do a real model pass.
            must_run_real = (win <= 1) or (st.step_index % win == 0) or (len(st.cache_ts) < 2)
            if must_run_real:
                y = model_function(x_in, timestep, **c)
                if not torch.is_tensor(y):
                    return y

                # Update cache with detached tensor to avoid holding onto any graph.
                st.cache_ts.append(t_val)
                st.cache_y.append(y.detach())

                # Keep only a small history; enough points to fit degree.
                max_cache = max(2, degree + 1)
                if len(st.cache_ts) > max_cache:
                    st.cache_ts = st.cache_ts[-max_cache:]
                    st.cache_y = st.cache_y[-max_cache:]

                # Fit coefficients once we have at least 2 points.
                try:
                    t_min, t_max = _resolve_tminmax()
                    ts = torch.tensor(st.cache_ts, device=y.device, dtype=torch.float32)
                    xs = _normalize_to_chebyshev_domain(ts, t_min=t_min, t_max=t_max)
                    Phi = _chebyshev_design_matrix(xs, degree=degree)

                    Y = torch.stack([v.float().reshape(-1) for v in st.cache_y], dim=0)  # [N, D]
                    st.coeffs = _fit_ridge(Phi, Y, lam=lam_f)
                except Exception:
                    st.coeffs = None

                return y

            # Forecast pass.
            if st.coeffs is None or len(st.cache_y) == 0:
                return model_function(x_in, timestep, **c)

            try:
                last_real = st.cache_y[-1]

                t_min, t_max = _resolve_tminmax(extra_t=t_val)
                x_t = _normalize_to_chebyshev_domain(
                    torch.tensor([t_val], device=last_real.device, dtype=torch.float32),
                    t_min=t_min,
                    t_max=t_max,
                )
                pred_flat = _forecast_from_coeffs(st.coeffs, x_t)
                pred = pred_flat.reshape(last_real.shape).to(dtype=last_real.dtype)

                if not torch.isfinite(pred).all():
                    return model_function(x_in, timestep, **c)

                if w_f >= 1.0:
                    return pred
                if w_f <= 0.0:
                    return last_real
                return (w_f * pred) + ((1.0 - w_f) * last_real)
            except Exception:
                return model_function(x_in, timestep, **c)

        model_clone.set_model_unet_function_wrapper(unet_wrapper_function)
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "MaxSpectrumPatcher": MaxSpectrumPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaxSpectrumPatcher": "Spectrum Patcher (Max)",
}
