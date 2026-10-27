# Kernel IR → PTX, directly

A compiler from Emmy's Kernel IR straight to NVIDIA PTX, written for the Emmy hackathon. No CUDA C, no nvcc, no LLVM:
the program walks the kernel's statements and prints PTX text. NVIDIA's `ptxas` then turns it into GPU code.

```bash
pip install emmy-ml==0.3.11 numpy requests      # emmy-ml only loads the IR; no torch needed
python compile.py path/to/kernel.ir.json > out.ptx
python compile.py path/to/kernel.ir.json --arch sm_70 > out.ptx    # for another GPU generation
```

## Layout

`compile.py` loads the IR with `Graph.from_dict`, takes the parameter order from `meta.json`, and prints the PTX.
The compiler is the `ptx/` package: one emitter class, built from parts, one file per job.

| File | What it does |
|---|---|
| `ptx/kernel.py` | the kernel: parameters, the `Tile` thread grid, the statement walk, the PTX header |
| `ptx/target.py` | the GPU: PTX version, shared-memory limit, which instructions it has |
| `ptx/builder.py` | the output: fresh registers, labels, instruction lines, scopes of computed values |
| `ptx/linear.py` | the linear form `Σ coeff·register + const` every integer index is kept in |
| `ptx/integers.py` | index math on linear forms: folding, caching, comparisons, predicates |
| `ptx/control.py` | branches and loops, innermost-loop unrolling (`Loop`, `StridedLoop`, `Cond`) |
| `ptx/floats.py` | float math: elementwise ops, `expf`, `tanh`, 16-bit rounding (`Assign`, `Accum`, `Select`, …) |
| `ptx/memory.py` | addresses, loads, stores, shared buffers, `cp.async` (`Load`, `Write`, `Smem`, …) |
| `ptx/reductions.py` | warp shuffles, shared-memory trees, barriers (`WarpShuffle`, `TreeHalve`, `Sync`) |
| `ptx/common.py` | the `Unsupported` error and float immediates |

Each Kernel IR statement `Foo` is translated by one method, `s_Foo`, in the file for its topic.

## How it works

- **One walk, one rule per statement.** `Tile` becomes the thread-index math (`%tid`, `%ctaid`), `Loop` a counted
  loop with labels and branches, `Load`/`Write` `ld`/`st` with computed addresses, `Assign` the float instruction for
  its op, `WarpShuffle` a `shfl.sync.bfly` butterfly, `TreeHalve` a shared-memory tree with `bar.sync`, `CpAsyncCopy`
  `cp.async`. Registers are handed out fresh; `ptxas` allocates the real ones.
- **Index folding.** Every integer index is kept as "registers × constants + a constant" until an instruction needs
  it. Constants fold, the same index is computed once, and the constant part of an address becomes the instruction's
  offset (`[%rd1+512]`).
- **Unrolling.** Innermost loops run 4 copies per trip (or fully, when the IR asks), so several loads are in flight.
- **Launch bounds.** The header declares the block size (`.maxntid 32, 1, 1`), as `__launch_bounds__` does for nvcc.
  Without it ptxas picks a default register budget; told that attention a02's block is one warp of 32 threads, it
  gives each thread 80 registers instead of 56 (nvcc's kernel: 77), and the kernel runs 33% faster.
- **Memory instructions.** Read-only inputs use `ld.global.nc`; neighbouring elements load as one `v2`/`v4` vector.
- **Accurate math.** `exp` is the same 9-instruction sequence CUDA's `expf` uses; division and square root round
  exactly (`div.rn`, `sqrt.rn`).

An unsupported statement raises `Unsupported` with its name, instead of printing wrong PTX.

## GPU generations

`--arch` takes `sm_50` (Maxwell) through `sm_120` (Blackwell): sm_50/52/53, 60/61/62, 70/72, 75, 80/86/87, 89, 90,
100, 120. The default is `sm_89`, what the problems are scheduled for. Three things change per GPU:

- **The PTX version** in the header: the oldest one `ptxas` accepts for that GPU and every instruction we print there.
- **Instructions an older GPU lacks** are replaced by a sequence that computes the same result:

  | Instruction | First GPU | Before it |
  |---|---|---|
  | `cp.async` (global → shared, no registers) | sm_80 | an `ld.global` and an `st.shared`; the commit and wait drop out |
  | `cvt.rn.bf16.f32` | sm_80 | round to nearest even on the integer bits; NaN stays NaN |
  | `tanh.approx.f32` | sm_75 | `(1 − t) / (1 + t)` with `t = e^(−2·abs(x))`, then the sign of x copied back |
  | `griddepcontrol.wait` | sm_90 | left out (it only lets a launch overlap the one before it) |

- **Shared memory per CTA** (48 KB before Volta, 64 KB on Turing, 99–227 KB after): a kernel that needs more than
  the GPU has is refused with `Unsupported`, since it could not launch there.

PTX for an older GPU runs on every newer one, so each variant was tested on the RTX 5090 runner (see Results).

## Results

All 46 problems pass on the hackathon's RTX 5090 runner through `submit.py`: the 15 open problems and 31 more we
generated with the hackathon's own `tools/generate.py` (other shapes and ops, all five categories). Against Emmy's
own CUDA C answer (`ref.cu` through nvcc), 40 of 46 run within 5%. The attention shape a10 runs at 0.77× its time,
a11 at 0.63×. The slowest were a02 (1.70×), a12 (1.41×) and r02 (1.23×); launch bounds
brought a02 to 1.14× (49.8 → 33.4 µs).

Per GPU generation, every problem was assembled by `ptxas` for that exact GPU (sm_50 through sm_90 locally; 0
failures), and the variants for sm_60, 70, 75, 80, 90 and 120 were run on the RTX 5090 and checked: all 46 pass, except
m01 (72 KB of shared memory) on sm_60 and sm_75, which is refused as it could not launch there.

Tried and dropped, as they changed no kernel by more than 2% (these problems fit in the 5090's L2 cache): an L2
prefetch-size hint on loads (`.L2::256B`) and on `cp.async` (`.L2::128B`), and streaming stores (`st.global.cs`).
