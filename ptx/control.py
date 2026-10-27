"""Branches and loops.

An innermost loop the IR marks ``unroll`` is copied once per iteration with its axis a constant; any other innermost
loop, counted or strided, runs ``UNROLL`` copies per trip before a remainder loop, so that many loads are in flight per
thread. Only innermost loops unroll, as nvcc's front end does, so copies never multiply.
"""

from .common import Unsupported, fimm
from .linear import Linear

UNROLL = 4


def innermost(loop) -> bool:
    """Whether no loop runs inside ``loop``."""
    return not any(type(s).__name__ in ("Loop", "StridedLoop") for s in loop.body.iter())


class Control:
    # ---- building blocks ---------------------------------------------------------------------------------------

    def branch(self, p, then, otherwise=None) -> None:
        """``if (p) then() else otherwise()``; a constant ``p`` emits only the side taken."""
        if p is True or p is False:
            if p:
                self.block(then)
            elif otherwise is not None:
                self.block(otherwise)
            return
        skip, end = self.label(), self.label()
        self.ins(f"@!{p} bra {skip}")
        self.block(then)
        if otherwise is not None:
            self.ins(f"bra.uni {end}")
        self.mark(skip)
        if otherwise is not None:
            self.block(otherwise)
            self.mark(end)

    def counted(self, lo: Linear, hi: Linear, step: int, body) -> None:
        """``for (i = lo; i < hi; i += step)``, tested before the first trip and at the bottom."""
        if lo.is_const and hi.is_const and lo.const >= hi.const:
            return
        i = self.reg("r")
        self.ins(f"mov.b32 {i}, {self.value(lo)}")
        done, top = self.label(), self.label()
        if not (lo.is_const and hi.is_const):
            p = self.reg("p")
            self.ins(f"setp.ge.s32 {p}, {i}, {self.value(hi)}")
            self.ins(f"@{p} bra {done}")
        self.mark(top)
        self.block(lambda: body(Linear.of(i)))
        self.ins(f"add.s32 {i}, {i}, {step}")
        p = self.reg("p")
        self.ins(f"setp.lt.s32 {p}, {i}, {self.value(hi)}")
        self.ins(f"@{p} bra {top}")
        self.mark(done)

    def counted_from(self, i: str, hi: Linear, step: int, body) -> None:
        """``while (i < hi) { body(i); i += step; }`` on an existing counter register."""
        top, done = self.label(), self.label()
        p = self.reg("p")
        self.ins(f"setp.ge.s32 {p}, {i}, {self.value(hi)}")
        self.ins(f"@{p} bra {done}")
        self.mark(top)
        self.block(lambda: body(Linear.of(i)))
        self.ins(f"add.s32 {i}, {i}, {step}")
        p = self.reg("p")
        self.ins(f"setp.lt.s32 {p}, {i}, {self.value(hi)}")
        self.ins(f"@{p} bra {top}")
        self.mark(done)

    def unrolled(self, name: str, lo: Linear, hi: Linear, step: int, stmts) -> None:
        """``UNROLL`` copies of the body per trip, axis ``i + k·step`` in copy ``k``, while the last copy is still in
        range; then the rest one at a time. The copies share the trip's scope, so they share every address register."""

        def copies(n):
            def run(i):
                for k in range(n):
                    self.with_axis(name, i + Linear(const=k * step), lambda: self.body(stmts))

            return run

        i = self.reg("r")
        self.ins(f"mov.b32 {i}, {self.value(lo)}")
        self.counted_from(i, hi + Linear(const=-(UNROLL - 1) * step), UNROLL * step, copies(UNROLL))
        if not (lo.is_const and hi.is_const and (hi.const - lo.const) % (UNROLL * step) == 0):
            self.counted_from(i, hi, step, copies(1))

    def with_axis(self, name: str, value: Linear, body) -> None:
        """Run ``body`` with loop axis ``name`` bound to ``value``."""
        saved = self.ints.get(name)
        self.ints[name] = value
        body()
        if saved is None:
            del self.ints[name]
        else:
            self.ints[name] = saved

    def seed(self, loop) -> None:
        """``<acc> = identity`` for each accumulator of the loop's own body, as ``Loop.render`` seeds it."""
        if not loop.seed:
            return
        seen = set()
        for a in loop.body:
            if type(a).__name__ == "Accum" and a.name not in seen:
                seen.add(a.name)
                if a.op.identity is None:
                    raise Unsupported(f"accumulator {a.op.name}")
                self.ins(f"mov.f32 {self.dest(a.name)}, {fimm(a.op.identity)}")

    # ---- statements --------------------------------------------------------------------------------------------

    def s_Loop(self, s) -> None:
        if s.carries:
            raise Unsupported("loop-carried state")
        self.seed(s)
        ext = self.size(self.extent_of(s.axis.extent))
        if s.unroll and ext.is_const and innermost(s):
            # The copies run straight-line in one block, so they share one scope: whatever does not depend on the
            # axis is computed once for all of them.
            def copies():
                for v in range(ext.const):
                    self.with_axis(s.axis.name, Linear(const=v), lambda: self.body(s.body))

            self.block(copies)
            return
        if innermost(s):
            self.unrolled(s.axis.name, Linear(), ext, 1, s.body)
            return
        self.counted(Linear(), ext, 1, lambda i: self.with_axis(s.axis.name, i, lambda: self.body(s.body)))

    def s_StridedLoop(self, s) -> None:
        self.seed(s)
        lo = self.lin(s.start)
        hi = self.size(self.extent_of(s.axis.extent)) if s.end is None else self.lin(s.end)
        step = self.lin(s.step)
        if not step.is_const:
            raise Unsupported("a strided loop with a runtime step")
        if s.end is not None:
            self.ints[f"{s.axis.name}_end"] = hi
        if innermost(s):
            self.unrolled(s.axis.name, lo, hi, step.const, s.body)
            return
        self.counted(lo, hi, step.const, lambda i: self.with_axis(s.axis.name, i, lambda: self.body(s.body)))

    def s_Cond(self, s) -> None:
        otherwise = (lambda: self.body(s.else_body)) if s.else_body else None
        self.branch(self.pred(s.cond), lambda: self.body(s.body), otherwise)
