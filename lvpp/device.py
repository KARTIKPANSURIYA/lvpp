"""Pick the best available accelerator: CUDA, then Apple Silicon (MPS), then CPU."""
from __future__ import annotations

import os

import torch

# Let the few operators MPS lacks fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# macOS: PyTorch and XGBoost each bring an OpenMP runtime; without this the
# process aborts when both are loaded. Import xgboost before torch as well.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def get_device() -> str:
    override = os.environ.get("LVPP_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str) -> torch.dtype:
    """bfloat16 where supported (CUDA, MPS on macOS 14+), else float32."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        try:
            torch.zeros(1, dtype=torch.bfloat16, device="mps")
            return torch.bfloat16
        except (RuntimeError, TypeError):
            return torch.float16
    return torch.float32


DEVICE = get_device()
