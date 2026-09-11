"""Hand-fused GPU kernels used as baselines for the compiled tensor path."""

from .triton_fused import TritonFusedCFR, triton_available

__all__ = ["TritonFusedCFR", "triton_available"]
