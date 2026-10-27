# Kernel IR → PTX, directly

A compiler from Emmy's Kernel IR straight to NVIDIA PTX, written for the Emmy hackathon. No CUDA C, no nvcc, no LLVM:
the program walks the kernel's statements and prints PTX text. NVIDIA's `ptxas` then turns it into GPU code.

```bash
pip install emmy-ml==0.3.11 numpy requests      # emmy-ml only loads the IR; no torch needed
python compile.py path/to/kernel.ir.json > out.ptx
```

| File | What it does |
|---|---|
| `compile.py` | loads the IR with `Graph.from_dict`, takes the parameter order from `meta.json`, prints the PTX |
| `ptx.py` | the compiler: one translation per Kernel IR statement kind |

## How it works

- **One walk, one rule per statement.** `Tile` becomes the thread-index math (`%tid`, `%ctaid`), `Loop` a counted
  loop with labels and branches, `Load`/`Write` `ld`/`st` with computed addresses, `Assign` the float instruction for
  its op, `WarpShuffle` a `shfl.sync.bfly` butterfly, `TreeHalve` a shared-memory tree with `bar.sync`, `CpAsyncCopy`
  `cp.async`. Registers are handed out fresh; `ptxas` allocates the real ones.
- **Index folding.** Every integer index is kept as "registers × constants + a constant" until an instruction needs
  it. Constants fold, the same index is computed once, and the constant part of an address becomes the instruction's
  offset (`[%rd1+512]`).
- **Unrolling.** Innermost loops run 4 copies per trip (or fully, when the IR asks), so several loads are in flight.
- **Memory instructions.** Read-only inputs use `ld.global.nc`; neighbouring elements load as one `v2`/`v4` vector.
- **Accurate math.** `exp` is the same 9-instruction sequence CUDA's `expf` uses; division and square root round
  exactly (`div.rn`, `sqrt.rn`).

An unsupported statement raises `Unsupported` with its name, instead of printing wrong PTX.

## Results

All 46 problems pass on the hackathon's RTX 5090 runner through `submit.py`: the 15 open problems and 31 more we
generated with the hackathon's own `tools/generate.py` (other shapes and ops, all five categories). Against Emmy's
own CUDA C answer (`ref.cu` through nvcc), 40 of 46 run within 5%. The attention shape a10 runs at 0.77× its time,
a11 at 0.63×. The slowest are a02 (1.70×), a12 (1.41×) and r02 (1.23×).
