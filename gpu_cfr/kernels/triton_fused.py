"""Fused per-level Triton kernels driving a stagewise solver's buffers.

This is the "custom kernel" baseline: the same iteration the compiled path
issues as a stream of generic tensor ops, hand-written so that each depth
level of the forward and backward pass is one gather-multiply-scatter kernel
and the regret-matching, regret-update and average-strategy steps are one
kernel each. Per iteration it launches ``2D + 4`` kernels for a tree of depth
``D`` (plus one scalar fill on the eager path) against the roughly ``8D + 20``
tensor ops of the generic path, so the comparison isolates what fusion buys
on top of, or instead of, CUDA-graph replay of the generic ops.

Scope: simultaneous updates, the CFR and CFR+ rules, float32 or float64. The
CFR+ regret floor is applied at the start of the next iteration's matching
step (``_positive_totals_kernel`` with ``CLAMP``), which visits the same
values as the generic path's end-of-iteration clamp; ``step`` floors the
exposed accumulator once more when it returns so the solver state matches.
"""

from __future__ import annotations

import torch

try:  # Triton is only present in the torch>=2 environment.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    _HAVE_TRITON = False

_BLOCK = 1024
_WARMUP_ITERS = 3


def triton_available() -> bool:
    """Whether the Triton kernels can be built in this environment."""

    return _HAVE_TRITON and torch.cuda.is_available()


if _HAVE_TRITON:

    @triton.jit
    def _positive_totals_kernel(regret_ptr, slot_info_ptr, totals_ptr, n, CLAMP: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        regret = tl.load(regret_ptr + offs, mask=mask, other=0.0)
        positive = tl.maximum(regret, 0.0)
        if CLAMP:
            tl.store(regret_ptr + offs, positive, mask=mask)
        info = tl.load(slot_info_ptr + offs, mask=mask, other=0)
        tl.atomic_add(totals_ptr + info, positive, mask=mask)

    @triton.jit
    def _strategy_kernel(regret_ptr, slot_info_ptr, totals_ptr, uniform_ptr, strat_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        positive = tl.maximum(tl.load(regret_ptr + offs, mask=mask, other=0.0), 0.0)
        info = tl.load(slot_info_ptr + offs, mask=mask, other=0)
        total = tl.load(totals_ptr + info, mask=mask, other=0.0)
        uniform = tl.load(uniform_ptr + offs, mask=mask, other=0.0)
        safe = tl.where(total > 0.0, total, 1.0)
        tl.store(strat_ptr + offs, tl.where(total > 0.0, positive / safe, uniform), mask=mask)

    @triton.jit
    def _forward_level_kernel(reach_ptr, strat_ptr, parent_ptr, child_ptr, slot_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        parent = tl.load(parent_ptr + offs, mask=mask, other=0)
        child = tl.load(child_ptr + offs, mask=mask, other=0)
        slot = tl.load(slot_ptr + offs, mask=mask, other=0)
        reach = tl.load(reach_ptr + parent, mask=mask, other=0.0)
        strat = tl.load(strat_ptr + slot, mask=mask, other=0.0)
        tl.store(reach_ptr + child, reach * strat, mask=mask)

    @triton.jit
    def _backward_level_kernel(values_ptr, strat_ptr, parent_ptr, child_ptr, slot_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        parent = tl.load(parent_ptr + offs, mask=mask, other=0)
        child = tl.load(child_ptr + offs, mask=mask, other=0)
        slot = tl.load(slot_ptr + offs, mask=mask, other=0)
        value = tl.load(values_ptr + child, mask=mask, other=0.0)
        strat = tl.load(strat_ptr + slot, mask=mask, other=0.0)
        tl.atomic_add(values_ptr + parent, strat * value, mask=mask)

    @triton.jit
    def _regret_update_kernel(
        values_ptr, reach_ptr, strat_ptr, regret_ptr, stratsum_ptr, weight_ptr,
        parent_ptr, child_ptr, slot_ptr, sign_ptr, opp_ptr, my_ptr, n,
        LINEAR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        parent = tl.load(parent_ptr + offs, mask=mask, other=0)
        child = tl.load(child_ptr + offs, mask=mask, other=0)
        slot = tl.load(slot_ptr + offs, mask=mask, other=0)
        sign = tl.load(sign_ptr + offs, mask=mask, other=0.0)
        opp = tl.load(opp_ptr + offs, mask=mask, other=0)
        my = tl.load(my_ptr + offs, mask=mask, other=0)
        instant = sign * (tl.load(values_ptr + child, mask=mask, other=0.0) - tl.load(values_ptr + parent, mask=mask, other=0.0))
        weighted = tl.load(reach_ptr + opp, mask=mask, other=0.0) * instant
        tl.atomic_add(regret_ptr + slot, weighted, mask=mask)
        contrib = tl.load(reach_ptr + my, mask=mask, other=0.0) * tl.load(strat_ptr + slot, mask=mask, other=0.0)
        if LINEAR:
            contrib = tl.load(weight_ptr) * contrib
        tl.atomic_add(stratsum_ptr + slot, contrib.to(tl.float64), mask=mask)


class TritonFusedCFR:
    """Run a ``StagewiseCFRSolver``'s iteration with fused Triton kernels.

    The wrapper owns no solver state: it reads and writes the solver's own
    persistent buffers and index blocks, so the solver's ``regret_sum``,
    ``strategy_sum``, ``iterations`` and ``average_strategy_flat`` stay valid
    and can be compared against the generic path directly. With
    ``cuda_graph=True`` the fused iteration is itself captured and replayed,
    which separates the fusion effect from the launch-batching effect.
    """

    def __init__(self, solver, *, cuda_graph: bool = False) -> None:
        if not triton_available():
            raise RuntimeError("Triton kernels require triton and a CUDA device")
        if solver.device.type != "cuda":
            raise ValueError("TritonFusedCFR drives CUDA solvers only")
        if solver.config.variant not in ("cfr", "cfr_plus") or solver.config.alternating_updates:
            raise ValueError("TritonFusedCFR supports simultaneous CFR and CFR+ only")
        self.solver = solver
        self.device = solver.device
        self.cuda_graph = cuda_graph
        self._clamp = solver.config.variant == "cfr_plus"
        self._linear = bool(solver.config.linear_averaging)
        self._weight = torch.ones((), dtype=solver.dtype, device=solver.device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._warmup_left = _WARMUP_ITERS
        s = solver
        self._num_slots = int(s.regret_sum.numel())
        self._num_player_edges = int(s._pe_slot.numel())
        self._forward = tuple((blk.parent2, blk.child2, blk.slot2_ext, int(blk.parent2.numel())) for blk in s._forward_blocks)
        self._backward = tuple((blk.parent, blk.child, blk.slot_bwd_ext, int(blk.parent.numel())) for blk in s._backward_blocks)

    @property
    def launches_per_iteration(self) -> int:
        """Kernel launches one fused iteration issues (template copy included)."""

        return len(self._forward) + len(self._backward) + 5

    def _iteration(self) -> None:
        s = self.solver
        grid_slots = (triton.cdiv(self._num_slots, _BLOCK),)
        s._totals_buf.zero_()
        _positive_totals_kernel[grid_slots](
            s.regret_sum, s._slot_infoset, s._totals_buf, self._num_slots, CLAMP=self._clamp, BLOCK=_BLOCK
        )
        _strategy_kernel[grid_slots](
            s.regret_sum, s._slot_infoset, s._totals_buf, s._slot_uniform, s._strategy_ext, self._num_slots, BLOCK=_BLOCK
        )
        for parent, child, slot, n in self._forward:
            _forward_level_kernel[(triton.cdiv(n, _BLOCK),)](s._reach_flat, s._strategy_ext, parent, child, slot, n, BLOCK=_BLOCK)
        s._values_buf.copy_(s._values_template)
        for parent, child, slot, n in self._backward:
            _backward_level_kernel[(triton.cdiv(n, _BLOCK),)](s._values_buf, s._strategy_ext, parent, child, slot, n, BLOCK=_BLOCK)
        _regret_update_kernel[(triton.cdiv(self._num_player_edges, _BLOCK),)](
            s._values_buf, s._reach_flat, s._strategy_ext, s.regret_sum, s.strategy_sum, self._weight,
            s._pe_parent, s._pe_child, s._pe_slot, s._player_edge_sign, s._pe_opp_idx, s._pe_my_idx,
            self._num_player_edges, LINEAR=self._linear, BLOCK=_BLOCK,
        )

    def _step_eager(self, n: int) -> None:
        for _ in range(n):
            self.solver.iterations += 1
            self._weight.fill_(float(self.solver.iterations))
            self._iteration()

    def _step_graph(self, n: int) -> None:
        remaining = n
        if self._graph is None:
            warm = min(self._warmup_left, remaining)
            self._step_eager(warm)
            self._warmup_left -= warm
            remaining -= warm
            if remaining == 0 or self._warmup_left > 0:
                return
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(self.device), torch.cuda.graph(graph):
                self._iteration()
                self._weight.add_(1.0)
            self._graph = graph
        self._weight.fill_(float(self.solver.iterations + 1))
        for _ in range(remaining):
            self.solver.iterations += 1
            self._graph.replay()

    def step(self, num_iterations: int = 1) -> None:
        """Advance the wrapped solver by ``num_iterations`` fused iterations."""

        if num_iterations < 0:
            raise ValueError("num_iterations must be non-negative")
        with torch.no_grad():
            if self.cuda_graph:
                self._step_graph(num_iterations)
            else:
                self._step_eager(num_iterations)
            if self._clamp:
                self.solver.regret_sum.clamp_(min=0.0)
            self.solver._last_root_value = self.solver._values_buf[self.solver.game.root].detach().clone()
