"""Kernel IR → PTX, printed directly: no CUDA C, no nvcc, no LLVM.

One walk over the ``KernelOp`` body, one translation per statement kind, PTX text out. ptxas does register allocation
and scheduling downstream, so registers are handed out fresh and declared by count at the top.

The emitter is one class built from parts, one file each:

==================  ===========================================================================================
``target.py``       the GPU being compiled for: PTX version, shared-memory limit, which instructions it has
``builder.py``      the output buffer: registers, labels, instructions, scopes of computed values
``linear.py``       the linear form ``Σ coeff·register + const`` every integer index is kept in
``integers.py``     index math on linear forms: folding, caching, comparisons, predicates
``control.py``      branches and loops, including innermost-loop unrolling (``Loop``, ``StridedLoop``, ``Cond``)
``floats.py``       float math: the elementwise ops, ``expf``, 16-bit rounding (``Assign``, ``Accum``, ``Select``…)
``memory.py``       addresses, loads, stores, shared buffers, ``cp.async`` (``Load``, ``Write``, ``Smem``…)
``reductions.py``   warp shuffles, shared-memory trees, barriers (``WarpShuffle``, ``TreeHalve``, ``Sync``)
``kernel.py``       the kernel itself: parameters, the ``Tile`` thread grid, the PTX header
==================  ===========================================================================================

A statement outside what these translate raises :class:`Unsupported`, naming it.
"""

from .common import Unsupported
from .kernel import emit
from .target import Target

__all__ = ["Target", "Unsupported", "emit"]
