"""The kernel: its parameters, the ``Tile`` that maps threads to cells, the statement walk, and the PTX header.

The header is written last, because it declares how many registers of each kind the body used.
"""

from math import prod

from emmy.compiler.ir.kernel.ir import pack_smem

from .builder import Builder
from .common import INT32_MAX, STORAGE, Unsupported
from .control import Control
from .floats import Floats
from .integers import Integers
from .linear import Linear
from .memory import STATIC_SMEM_CAP, Memory
from .reductions import Reductions
from .target import Target

BLOCK_SIZE = 256  # threads per CTA of a scalar tile (one thread per element)
_REG_TYPES = {"p": ".pred", "r": ".b32", "rd": ".b64", "f": ".f32", "h": ".b16"}


class Emitter(Builder, Integers, Control, Floats, Memory, Reductions):
    def __init__(self, kernel, tensors, arch: Target):
        super().__init__()
        self.kernel, self.arch = kernel, arch
        self.ints: dict[str, Linear] = {}  # axis variables and integer lets
        self.vals: dict[str, str] = {}  # float SSA names → their one f32 register
        self.shapes = {n: tuple(self.extent_of(d) for d in t.shape) for n, t in tensors.items()}
        self.dtypes = {n: str(t.dtype) for n, t in tensors.items()}
        for n, dt in self.dtypes.items():
            if dt not in STORAGE:
                raise Unsupported(f"buffer {n} of dtype {dt}")
            if prod(d for d in self.shapes[n] if isinstance(d, int)) > INT32_MAX:
                raise Unsupported(f"buffer {n} past 32-bit indexing")
        self.buffers = tuple(tensors)  # the launch order: the caller names every buffer in it
        self.smem: dict[str, Linear] = {}  # shared buffer → byte address (32-bit shared space)
        self.globals: dict[str, str] = {}  # global buffer → 64-bit base register
        self.decls: list[str] = []  # static .shared arrays, declared in the header
        offsets, total = pack_smem(list(kernel.smem_buffers.values()))
        if total > arch.smem_bytes:
            raise Unsupported(f"{total} bytes of shared memory: {arch.name} allows {arch.smem_bytes}")
        self.pool = offsets if total > STATIC_SMEM_CAP else None

    def body(self, stmts) -> None:
        """The walk: statement ``Foo`` is translated by method ``s_Foo``."""
        for s in stmts:
            k = type(s).__name__
            fn = getattr(self, f"s_{k}", None)
            if fn is None:
                raise Unsupported(f"statement {k}")
            fn(s)

    # ---- the thread grid ---------------------------------------------------------------------------------------

    def s_Tile(self, t) -> None:
        """Thread ``ctaid · cells + tid`` owns one cell of the tile's axes, decoded row-major (last axis fastest).
        Threads past the last cell skip the body, unless the grid divides exactly."""
        if t.aux_threads:
            raise Unsupported("warp-specialized tile")
        tid = self.reg("r")
        self.ins(f"mov.u32 {tid}, %tid.x")
        lane, warp = self.reg("r"), self.reg("r")
        self.ins(f"and.b32 {lane}, {tid}, 31")
        self.ins(f"shr.u32 {warp}, {tid}, 5")
        self.ints["lane"], self.ints["warp"] = Linear.of(lane), Linear.of(warp)
        cells = t.block_threads if t.block_threads is not None else BLOCK_SIZE
        self.block_threads = cells  # threads per CTA, declared in the header
        ctaid = self.reg("r")
        self.ins(f"mov.u32 {ctaid}, %ctaid.x")
        gid = self.reg("r")
        self.ins(f"mad.lo.s32 {gid}, {ctaid}, {cells}, {tid}")  # blockDim.x is the tile's cell count
        extents = [self.size(self.extent_of(a.extent)) for a in t.axes]
        total = Linear(const=1)
        for e in extents:
            total = self.lin_mul(total, e)
        if total.is_const and total.const > INT32_MAX:
            raise Unsupported("a grid past 32-bit cell ids")
        g = Linear.of(gid)

        def decode():
            stride = Linear(const=1)
            for i in range(len(t.axes) - 1, -1, -1):
                q = self.div(g, stride)
                self.ints[t.axes[i].name] = q if i == 0 else self.rem(q, extents[i])
                stride = self.lin_mul(stride, extents[i])
            self.raster(t, g)
            self.body(t.body)

        exact = total.is_const and total.const % cells == 0
        self.branch(True if exact else self.cmp("lt", g, total), decode)

    def raster(self, t, gid: Linear) -> None:
        """Grouped CTA order, as ``Tile.render`` spells it: ``G`` block-tiles of the grouped axis run fastest within
        each stripe of the other, so neighbouring CTAs share an operand slab in L2."""
        pair = _raster_pair(t)
        if pair is None:
            return
        gi, oi = pair
        ext = t.extents
        g = min(t.raster_group, ext[gi])
        if ext[gi] <= 1 or g <= 1:
            return
        mi, ni = min(gi, oi), max(gi, oi)
        ea, eb = ext[gi], ext[oi]
        sub = self.div(gid, Linear(const=prod(ext[ni + 1 :])))
        if mi != 0:
            sub = self.rem(sub, Linear(const=ea * eb))
        group = self.div(sub, Linear(const=g * eb))
        within = self.rem(sub, Linear(const=g * eb))
        rest = self.reg("r")
        self.ins(f"min.s32 {rest}, {self.in_ireg(Linear(const=ea) + group.scale(-g))}, {g}")
        size = Linear.of(rest)
        self.ints[t.axes[gi].name] = group.scale(g) + self.rem(within, size)
        self.ints[t.axes[oi].name] = self.div(within, size)

    # ---- the whole kernel --------------------------------------------------------------------------------------

    def emit(self) -> str:
        k = self.kernel
        if len(k.body) != 1 or type(k.body[0]).__name__ != "Tile":
            raise Unsupported("a kernel body other than one Tile")
        name = k.name
        params = [f"\t.param .u64 {name}_{b}" for b in self.buffers]
        for b in self.buffers:
            raw, base = self.reg("rd"), self.reg("rd")
            self.ins(f"ld.param.u64 {raw}, [{name}_{b}]")
            self.ins(f"cvta.to.global.u64 {base}, {raw}")
            self.globals[b] = base
        if self.arch.has_grid_dependency:
            # Wait for the grid ahead before the first read, as every CUDA kernel does: its presence is what lets the
            # runtime start this launch while that grid drains.
            self.ins("griddepcontrol.wait")
        self.body(k.body)
        head = [f".version {self.arch.ptx_version}", f".target {self.arch.name}", ".address_size 64", ""]
        if self.pool is not None:
            align = max(16, *(s.align or 0 for s in k.smem_buffers.values()))
            head.append(f".extern .shared .align {align} .b8 _smem_pool[];")
        # The block size, as ``__launch_bounds__`` tells nvcc: ptxas then sets its register budget for exactly this many threads.
        head += [f".visible .entry {name}(", ",\n".join(params), f") .maxntid {self.block_threads}, 1, 1", "{"]
        head += [f"\t.reg {t} %{p}<{self.counts[p] + 1}>;" for p, t in _REG_TYPES.items()]
        return "\n".join([*head, *self.decls, *self.lines, "\tret;", "}", ""])


def _raster_pair(t):
    """``(grouped, other)`` axis positions of an active grouped raster: the (m, n) block axes, adjacent, m first."""
    group, axes = t.raster_group, t.raster_axes
    if not (group and group > 1 and axes):
        return None
    names = [a.name for a in t.axes]
    if axes[0] not in names or axes[1] not in names:
        return None
    mi, ni = names.index(axes[0]), names.index(axes[1])
    if ni != mi + 1:
        return None
    return (mi, ni) if t.raster_orient == "m" else (ni, mi)


def emit(kernel, tensors, *, target: str | Target = "sm_89") -> str:
    """PTX for ``kernel`` (a ``KernelOp``) on GPU ``target`` (``sm_89`` by default: what the problems are scheduled
    for). ``tensors`` maps every buffer the kernel reads or writes to its ``Tensor``, in launch order (``meta.json``'s
    ``arg_order``): the kernel's parameters follow it."""
    arch = target if isinstance(target, Target) else Target.parse(target)
    return Emitter(kernel, tensors, arch).emit()
