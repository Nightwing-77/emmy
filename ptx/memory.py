"""Addresses, loads, stores, shared buffers, and asynchronous copies into shared memory.

Global buffers are kernel parameters, held as one 64-bit base register each. Shared buffers are 32-bit addresses in
the shared space: one static ``.shared`` array each, or, past ``STATIC_SMEM_CAP``, offsets into one dynamic pool.
"""

from math import prod

from .common import STORAGE, Unsupported
from .linear import Linear

STATIC_SMEM_CAP = 48 * 1024  # bytes of static shared memory a CTA may declare; past it, one dynamic pool
_SMEM_DTYPE = {"float": "f32", "__half": "f16", "__nv_bfloat16": "bf16"}  # the C type a shared buffer is declared in


class Memory:
    # ---- addresses ---------------------------------------------------------------------------------------------

    def flat(self, buf: str, index) -> Linear:
        """Row-major element offset of ``buf[index...]``."""
        shape = self.shapes[buf]
        if len(index) not in (1, len(shape)):
            raise Unsupported(f"{buf}: {len(index)} indices for rank {len(shape)}")
        if len(index) == 1:
            return self.lin(index[0])
        off, stride = Linear(), Linear(const=1)
        for e, d in zip(reversed(index), reversed(shape), strict=True):
            i = self.lin(e)
            if stride.is_const:
                off = off + i.scale(stride.const)
            else:
                off = off + self.lin_mul(i, stride)
            stride = stride.scale(d) if isinstance(d, int) else self.lin_mul(stride, self.ints[d])
        return off

    def address(self, buf: str, index, extra: int = 0) -> tuple[str, str]:
        """``(space, "[base+offset]")`` for element ``index`` (plus ``extra`` elements) of ``buf``: the constant part
        folds into the immediate, the rest is computed once per scope."""
        nbytes = STORAGE[self.dtypes[buf]][1]
        elems = self.flat(buf, index) + Linear(const=extra)
        if buf in self.smem:
            full = self.smem[buf] + elems.scale(nbytes)
            return "shared", f"[{self.value(full.without_const())}+{full.const}]" if full.terms else f"[{full.const}]"
        var = elems.without_const()
        if var.is_const:
            return "global", f"[{self.globals[buf]}+{elems.const * nbytes}]"
        key = ("addr", buf, var)
        if (a := self.cached(key)) is None:
            a = self.remember(key, self.reg("rd"))
            self.ins(f"mul.wide.s32 {a}, {self.value(var)}, {nbytes}")
            self.ins(f"add.s64 {a}, {a}, {self.globals[buf]}")
        return "global", f"[{a}+{elems.const * nbytes}]"

    def read_space(self, buf: str, space: str) -> str:
        """A global buffer the kernel never writes reads through the non-coherent cache, as ``const __restrict__``
        gives in CUDA C."""
        return "global.nc" if space == "global" and buf not in self.kernel.outputs else space

    # ---- statements --------------------------------------------------------------------------------------------

    def s_Load(self, s) -> None:
        """Up to four neighbouring elements; four or two f32 load as one vector instruction."""
        if s.input not in self.smem and s.input not in self.globals:
            raise Unsupported(f"load from {s.input}")
        ty, _ = STORAGE[self.dtypes[s.input]]
        regs = [self.dest(n) for n in s.names]
        space, addr = self.address(s.input, s.index)
        space = self.read_space(s.input, space)
        if ty == "f32" and len(regs) in (2, 4):
            self.ins(f"ld.{space}.v{len(regs)}.f32 {{{', '.join(regs)}}}, {addr}")
            return
        for k, d in enumerate(regs):
            _, addr = self.address(s.input, s.index, k)
            if ty == "f32":
                self.ins(f"ld.{space}.f32 {d}, {addr}")
            else:
                h = self.reg("h")
                self.ins(f"ld.{space}.b16 {h}, {addr}")
                self.widen(d, h, self.dtypes[s.input])

    def s_Write(self, s) -> None:
        """A plain store (vector for two or four f32), or an atomic add (``red``) when several CTAs share an output."""
        dt = self.dtypes[s.output]
        vals = [self.in_reg(self.fval(v)) for v in s.values]
        if s.atomic:
            if dt != "f32" or len(vals) != 1:
                raise Unsupported(f"atomic write of {len(vals)} x {dt}")
            space, addr = self.address(s.output, s.index)
            self.ins(f"red.{space}.add.f32 {addr}, {vals[0]}")
            return
        if dt == "f32" and len(vals) in (2, 4):
            space, addr = self.address(s.output, s.index)
            self.ins(f"st.{space}.v{len(vals)}.f32 {addr}, {{{', '.join(vals)}}}")
            return
        for k, v in enumerate(vals):
            space, addr = self.address(s.output, s.index, k)
            if dt == "f32":
                self.ins(f"st.{space}.f32 {addr}, {v}")
            else:
                h = self.reg("h")
                self.to_16(h, v, dt)
                self.ins(f"st.{space}.b16 {addr}, {h}")

    def s_Smem(self, s) -> None:
        dt = _SMEM_DTYPE.get(s.dtype)
        if dt is None:
            raise Unsupported(f"shared buffer of {s.dtype}")
        self.shapes[s.name], self.dtypes[s.name] = tuple(int(e) for e in s.extents), dt
        r = self.reg("r")
        if self.pool is not None:
            self.ins(f"mov.u32 {r}, _smem_pool")
            self.smem[s.name] = Linear.of(r) + Linear(const=self.pool[s.name])
            return
        nbytes = prod(int(e) for e in s.extents) * STORAGE[dt][1]
        self.decls.append(f"\t.shared .align {max(s.align or 0, STORAGE[dt][1])} .b8 {s.name}[{nbytes}];")
        self.ins(f"mov.u32 {r}, {s.name}")
        self.smem[s.name] = Linear.of(r)

    def s_CpAsyncCopy(self, s) -> None:
        """``nbytes`` from global straight into shared memory. Before sm_80 there is no ``cp.async``: the bytes go
        through registers (one load, one store), so they are in place before any wait."""
        if s.swizzle != "NONE":
            raise Unsupported("swizzled cp.async")
        index = s.smem_index
        if s.lane_index is not None:
            # The lane's own offset inside the slab tile: ``row * cols + col`` on top of the tile's start.
            cols = self.shapes[s.smem][-1]
            index = (self.flat(s.smem, s.smem_index) + self.lin(s.lane_index[0]).scale(cols) + self.lin(s.lane_index[1]),)
        _, dst = self.address(s.smem, index)
        space, src = self.address(s.src, s.src_index)
        if self.arch.has_cp_async:
            self.ins(f"cp.async.{'cg' if s.nbytes == 16 else 'ca'}.shared.global {dst}, {src}, {s.nbytes}")
            return
        words = {4: 1, 8: 2, 16: 4}.get(s.nbytes)
        if words is None:
            raise Unsupported(f"a copy of {s.nbytes} bytes")
        regs = [self.reg("r") for _ in range(words)]
        vec, ops = ("", regs[0]) if words == 1 else (f".v{words}", f"{{{', '.join(regs)}}}")
        self.ins(f"ld.{self.read_space(s.src, space)}{vec}.b32 {ops}, {src}")
        self.ins(f"st.shared{vec}.b32 {dst}, {ops}")

    def s_CpAsyncCommit(self, _) -> None:
        if self.arch.has_cp_async:
            self.ins("cp.async.commit_group")

    def s_CpAsyncWait(self, s) -> None:
        if self.arch.has_cp_async:
            self.ins(f"cp.async.wait_group {s.group}")
