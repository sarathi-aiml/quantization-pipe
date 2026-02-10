"""Blackwell GPU (sm_121) compatibility patches for PyTorch.

The NVIDIA GB10 (Blackwell architecture, compute capability 12.1) requires
NVRTC from CUDA 13.0+ for JIT-compiled reduction kernels. PyTorch nightly
(cu128) bundles NVRTC 12.8 which doesn't know sm_121, causing failures on
integer reduction ops like prod(). This module patches those ops to use CPU
fallback for small integer tensors, which is functionally correct and has
negligible performance impact since the affected tensors are tiny metadata
tensors (e.g., image grid dimensions).
"""

import torch
import logging

logger = logging.getLogger(__name__)

_patched = False


def apply_blackwell_patches() -> None:
    """Apply compatibility patches for Blackwell GPU NVRTC limitations."""
    global _patched
    if _patched:
        return

    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    if capability[0] < 12:
        return  # Not Blackwell, no patches needed

    logger.info(f"Blackwell GPU detected (sm_{capability[0]}{capability[1]}), applying NVRTC compat patches")

    _INT_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8)

    # Patch prod
    _original_prod = torch.Tensor.prod

    def _safe_prod(self, *args, **kwargs):
        if self.is_cuda and self.dtype in _INT_DTYPES:
            return _original_prod(self.cpu(), *args, **kwargs).to(self.device)
        return _original_prod(self, *args, **kwargs)

    torch.Tensor.prod = _safe_prod

    # Patch cumprod (used in some attention implementations)
    _original_cumprod = torch.Tensor.cumprod

    def _safe_cumprod(self, *args, **kwargs):
        if self.is_cuda and self.dtype in _INT_DTYPES:
            return _original_cumprod(self.cpu(), *args, **kwargs).to(self.device)
        return _original_cumprod(self, *args, **kwargs)

    torch.Tensor.cumprod = _safe_cumprod

    _patched = True
    logger.info("Blackwell NVRTC compatibility patches applied")
