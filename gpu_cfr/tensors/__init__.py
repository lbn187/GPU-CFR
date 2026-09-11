"""Tensor layout, aggregation, and payoff block helpers."""

from .aggregation import bucket_sum
from .edges import EdgeArrays, build_edge_arrays
from .layout import CompiledEFG
from .memory import (
    MemoryEstimate,
    PayoffMemoryEstimate,
    estimate_compiled_efg_bytes,
    estimate_payoff_block_bytes,
    estimate_recursive_solver_bytes,
    estimate_solver_state_bytes,
    tensor_nbytes,
)
from .sparse_payoff import COOPayoffBlock, DensePayoffBlock

__all__ = [
    "CompiledEFG",
    "EdgeArrays",
    "MemoryEstimate",
    "PayoffMemoryEstimate",
    "bucket_sum",
    "build_edge_arrays",
    "COOPayoffBlock",
    "DensePayoffBlock",
    "estimate_compiled_efg_bytes",
    "estimate_payoff_block_bytes",
    "estimate_recursive_solver_bytes",
    "estimate_solver_state_bytes",
    "tensor_nbytes",
]
