# GPU-CFR

Code for the paper *"GPU-CFR: 80x Faster Counterfactual Regret Minimization by Compiling the Game to Static Dataflow and CUDA Graph Replay"*. 

Author: Boning Li, Longbo Huang

E-mail: li-bn22@mails.tsinghua.edu.cn

## Citation

@article{li2026av,
  title={GPU-CFR: 80x Faster Counterfactual Regret Minimization by Compiling the Game to Static Dataflow and CUDA Graph Replay},
  author={Li, Boning and Huang, Longbo},
  journal={arXiv preprint arXiv:2609.11923},
  year={2026}
}

## Requirements

The extension modules are built for one interpreter ABI; `BUILD_INFO.txt`
lists the build environment.

- CPython **3.9.x**, Linux x86_64 (the `cpython-39-x86_64-linux-gnu` ABI)
- `torch` >= 1.13, `numpy`
- For the GPU paths: an NVIDIA GPU with CUDA
- For the fused-kernel comparison (`gpu_cfr.benchmarks.kernel_bench`): `torch` 2.5 and `triton` 3.1

Additional dependencies for public-game adapters and third-party baselines:

- OpenSpiel: `pip install open_spiel`; also required for GPU-CFR public-game adapters
- LiteEFG: `pip install liteefg open_spiel`
- NoRegret (https://github.com/uoftcprg/noregret, Python >= 3.12): point
  `NOREGRET_PYTHON` at that interpreter. This package currently lacks
  `noregret_driver.py` and `noregret_file_game.py`; setting the interpreter
  alone is not sufficient to run this baseline

Libratus endgames require `LIBRATUS_ENDGAMES` to point at the endgame files.

## Layout

| Path | Contents |
|---|---|
| `gpu_cfr/efg/` | Game specification, EFG file reader, validation, and the compiler from a game tree to static depth-level dataflow. |
| `gpu_cfr/tensors/` | Tensor layout, edge tables, aggregation, memory model, and the sparse payoff block. |
| `gpu_cfr/solvers/` | The CUDA-graph stagewise solver (`stagewise`), the lifecycle API (`lifecycle`: `reset()`, `set_payoffs()`, `set_root_ranges()`), the CFR update rules (CFR, CFR+, DCFR, PCFR+), the scheduler, and the reference iterate the optimized path is verified against (`snapshot_reference`). |
| `gpu_cfr/games/` | Kuhn, HUNL river and turn subgames, Libratus endgame loader, and the OpenSpiel adapter. |
| `gpu_cfr/eval/` | Best response, exploitability, metrics, and strategy export. |
| `gpu_cfr/kernels/triton_fused.py` | The hand-fused Triton kernel baseline. Shipped as text because Triton's JIT compiles kernels from Python source at run time. |
| `gpu_cfr/benchmarks/` | The benchmark harness: cell registry, runners, profiler hooks, record schema, and the command-line drivers listed below. `benchmarks/baselines/` holds our wrappers for OpenSpiel, LiteEFG, and NoRegret. |
| `probes/` | The probe scripts behind the appendix studies: overhead probe, scaling sweep, alternating-update gap, evaluation precision, variant ordering determinism, sparse-operation microbenchmarks, and Libratus endgame feasibility. The scaling sweep and the Libratus probes run on the GPU and take minutes. |
| `project_gpu_cfr/` | Import shim so `project_gpu_cfr.gpu_cfr.*` resolves to `gpu_cfr/`. |

## Running the benchmarks

Run from the package root with `PYTHONPATH=$PWD`. The `gpu_cfr.benchmarks`
drivers accept `--help`; `--dry-run` on the main CLI prints the expanded cell
list without running. Games: `kuhn_poker`, `hunl_river`, `hunl_turn`,
`libratus_subgame3`, `libratus_subgame4`, plus the OpenSpiel games of the suite.

| Command | Measures |
|---|---|
| `python -m gpu_cfr.benchmarks.cli --game kuhn_poker --solvers gpu_cfr_plus --iterations 1000 --device cuda --execution-mode stagewise --output out.jsonl` | End-to-end timing of one solve. Options: `--variant {cfr,cfr_plus,dcfr,pcfr_plus}`, `--dtype`, `--stage-block-size`, `--alternating-updates`, `--suite`. |
| `python -m gpu_cfr.benchmarks.curve_cli ...` | Anytime convergence curves: exploitability against wall-clock at fixed time targets. |
| `python -m gpu_cfr.benchmarks.iter_bench ...` | Steady-state milliseconds per iteration, eager versus CUDA-graph replay. |
| `python -m gpu_cfr.benchmarks.lifecycle_bench ...` | Build, warm-up, capture, reset, and re-target timings of one compiled tree. |
| `python -m gpu_cfr.benchmarks.kernel_bench ...` | Compiled dataflow against the hand-fused Triton kernel. |
| `python -m gpu_cfr.benchmarks.payoff_microbench_cli ...` | Payoff-block microbenchmark. |
| `python probes/overhead_probe.py <game> [<game> ...]` | Build, warm-up, capture, and steady-state cost of one compiled tree on CUDA. |
| `python probes/scaling_sweep.py [K ...]` | Tree-size scaling of the HUNL turn subgame with K hands per player (CUDA). |
| `python probes/probe_*.py` | The remaining probes; `probe_alternating_gap`, `probe_eval_precision`, `probe_libratus_range_ceiling`, `probe_pcfr_alt_factorial`, and the two `probe_variant_order_*` scripts accept `--help`, the two Libratus feasibility probes take positional arguments. |

The packaged `probe_sparse_ops` is not runnable in its required NoRegret
Python 3.12 environment: its extension is compiled for CPython 3.9. Setting
`NOREGRET_PYTHON` does not resolve this ABI mismatch. The gpugt driver and
CLI backend are not included either.

The main and curve CLIs append records to `--output`. Benchmarks such as
`iter_bench`, `lifecycle_bench`, and `kernel_bench`, and the overhead and
scaling probes print results to stdout; redirect stdout to retain them.
Output fields depend on the driver.

## Reproducing the main results

Run from the package root with `export PYTHONPATH=$PWD` and
`export OMP_NUM_THREADS=8`. Install `open_spiel` for the public-game suite.
Use an idle GPU and pin each process to eight available CPU cores with
`taskset -c <cores>` for timing comparisons. Table 10 lists independent-process
counts: 20 for graph, eager and compiled CPU, and 10 for vanilla and external
baselines. The commands below are reduced checks, not that full protocol.
`--reps 3` rebuilds the solver three times within one process. To reproduce
independent-process repetitions, invoke the command separately for each repeat.
Use fresh output paths for each run.

**Table 3, GPU-CFR rows (graph and eager, ms per iteration).** One process
per game and mode; the `graph` row is `--cuda-graph`, the `eager` row is
`--no-cuda-graph`:

```bash
for g in kuhn_poker dark_hex_2x2 leduc_poker goofspiel hunl_river battleship liars_dice hunl_turn; do
  python -m gpu_cfr.benchmarks.iter_bench --game $g --device cuda --variant cfr_plus \
      --iters 1000 --warmup 50 --reps 3 --cuda-graph    >> iterbench_graph.jsonl
  python -m gpu_cfr.benchmarks.iter_bench --game $g --device cuda --variant cfr_plus \
      --iters 1000 --warmup 50 --reps 3 --no-cuda-graph >> iterbench_eager.jsonl
done
```

Read `ms_per_iter` for graph and eager. The `vanilla CFR` row instead uses
the end-to-end CLI, with `--solvers gpu_cfr --variant cfr --device cuda
--execution-mode stagewise --iterations 1000`. Read
`1000 * timing.train_time_sec / iterations`, not `iter_bench` timings.

**Table 3, "Ours CPU (8 threads)" row.** The end-to-end CLI on CPU with eight
threads; ms per iteration is `1000 * timing.train_time_sec / iterations`:

```bash
for g in kuhn_poker dark_hex_2x2 leduc_poker goofspiel hunl_river battleship liars_dice hunl_turn; do
  OMP_NUM_THREADS=8 python -m gpu_cfr.benchmarks.cli --game $g --solvers gpu_cfr_plus \
      --iterations 1000 --device cpu --execution-mode stagewise --output cli_cpu8.jsonl
done
```

**Table 3, baseline rows.** Use the CLI with `--solvers openspiel_cfr
--device cpu` (needs `open_spiel`) or `--solvers liteefg --device cpu`
(needs `liteefg open_spiel`). Set `OMP_NUM_THREADS=1` for these runs and
use 1000 iterations. OpenSpiel does not support the HUNL/Libratus EFG route;
those cells are skipped. LiteEFG supports that route. The NoRegret wrapper
also supports EFG input, but its external driver files are missing from this
package, so the Kim baseline is not currently runnable from this package alone.

**Optimization ladder and setup costs (Fig. 4, Tables 4 and 9).**
`python probes/overhead_probe.py hunl_turn hunl_river` prints build, warm-up,
capture, and steady-state ms per iteration of the compiled tree on CUDA;
`python -m gpu_cfr.benchmarks.lifecycle_bench --game hunl_turn` times build,
capture, reset, and re-target. The eager and graph stages of the ladder are
the `iter_bench` runs above on `hunl_turn`. The reference-stage timing
driver is not included, so these commands do not reproduce the complete
three-stage ladder or its operation counts.

**Tree-size scaling (Fig. 5).** `python probes/scaling_sweep.py 4 8 12 16 20 24`
grows the turn subgame from 4 to 24 hands per player and prints states and
ms per iteration for each size.

**Convergence against wall-clock (Fig. 6, Table 8).** One process per
backend, game, and seed at the six log-spaced time targets:

```bash
python -m gpu_cfr.benchmarks.curve_cli --game hunl_turn --solvers gpu_cfr_plus --device cuda \
    --execution-mode stagewise --time-budgets-sec 0.1,0.3,1,3,10,30 --seeds 0 --output curves.jsonl
```

Repeat with `--seeds 1` to `9` and with `--device cpu`, and with
`--solvers liteefg --device cpu` for LiteEFG. NoRegret requires the missing
driver files noted above. The full curves and time-to-threshold comparison
have not been remeasured with this reduced package.

**Measured reduced check.** On one A100, the eight graph-path medians
were within approximately 1% of Table 3. The scaling endpoints gave 37.9 times
as many states and 2.94 times the iteration time, compared with 2.95 in the
paper. HUNL river eager and CPU timings were approximately 13% and 21% slower.
These checks used three within-process GPU repeats and one CPU process per
game without affinity pinning. They do not establish agreement with each
game's independent-process distribution. Table 10 reports dispersion, not a
universal acceptance tolerance. Timings on other hardware will differ.

## Python API

```python
from gpu_cfr.games.kuhn import build_kuhn_game
from gpu_cfr.config import CFRConfig, DeviceConfig
from gpu_cfr.solvers.stagewise import StagewiseCFRSolver
```

`help(gpu_cfr.solvers.stagewise)` lists the public classes and functions.
CUDA-graph replay agrees with eager execution on regret sums, strategy sums,
and root value at `atol=1e-9` in the tested configurations. Lifecycle tests
compare `reset()` followed by a solve with a fresh solver, and `set_payoffs()`
with recompilation. CPU comparisons use bitwise equality; CUDA comparisons
use `atol=1e-9`, not a bitwise-equality guarantee.

## How the modules are packaged

Each module `x.py` is compiled to `x.cpython-39-x86_64-linux-gnu.so`. A module
that is also a script is compiled as `x_impl.so`, and `x.py` remains as a short
launcher stub that re-exports the compiled module's names and runs the original
`__main__` body, so `python -m pkg.x`, `python path/x.py`, and
`from pkg.x import name` all behave as with the source. Package `__init__.py`
files are plain re-export lists. `inspect.getsource` on compiled functions
  - defaults
raises `TypeError`, and tracebacks carry line numbers without source lines.
