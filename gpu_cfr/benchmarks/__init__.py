"""Benchmark utilities for project_gpu_cfr."""

from .artifacts import BenchmarkArtifactError, validate_benchmark_record, validate_jsonl_records
from .gates import BenchmarkGateError, require_expected_row_count, require_nonempty_ok, validate_curve_invariants
from .schema import BenchmarkConfig, BenchmarkCurvePoint, BenchmarkResult

__all__ = [
    "BenchmarkArtifactError",
    "BenchmarkConfig",
    "BenchmarkCurvePoint",
    "BenchmarkGateError",
    "BenchmarkResult",
    "require_expected_row_count",
    "require_nonempty_ok",
    "validate_benchmark_record",
    "validate_curve_invariants",
    "validate_jsonl_records",
]
