"""Combining values across threads: register shuffles inside a warp, shared-memory trees across a CTA, barriers."""

from emmy.compiler.ir.expr import Var

from .common import Unsupported
from .linear import Linear


class Reductions:
    def shuffle(self, state, state_b, program, length, dtype) -> None:
        """The XOR butterfly over ``length`` lanes: each step pulls every component from the lane ``k`` apart and
        folds it in with the stored combine, so every lane ends holding the whole combine."""
        k = length // 2
        while k:
            mask = self.reg("r")
            self.ins(f"activemask.b32 {mask}")
            for st, sb in zip(state, state_b, strict=True):
                self.ins(f"shfl.sync.bfly.b32 {self.dest(sb)}, {self.fval(st)}, {k}, 31, {mask}")
            self.merge(program, dtype)
            k //= 2

    def merge(self, program, dtype) -> None:
        """Run the IR's combine program: the ``Assign``-like steps that fold two partial states into one."""
        for a in program:
            self.elementwise(self.dest(a.name), a.op.name, [self.fval(x) for x in a.args])
            self.round_to(self.vals[a.name], dtype)

    # ---- statements --------------------------------------------------------------------------------------------

    def s_Sync(self, s) -> None:
        if s.barrier_id:
            raise Unsupported("named barrier")
        self.ins("bar.warp.sync -1" if s.warp else "bar.sync 0")

    def s_WarpShuffle(self, s) -> None:
        self.shuffle(s.state, s.state_b, s.combine_states, s.length, s.dtype)

    def s_TreeHalve(self, t) -> None:
        """Fold ``length`` partials in shared memory down to slot 0, then every thread reads the result back.

        One partial per warp (at most 32): warp 0 folds them with a register butterfly. Otherwise a halving tree:
        each round the lower half of the threads folds in the upper half, with a barrier between rounds."""
        if t.inner is not None or t.barrier_id:
            raise Unsupported("segmented or named-barrier tree")
        tid = self.lin(Var(t.tid_var))

        def load(buf, i, d):
            _, addr = self.address(buf, (i,))
            self.ins(f"ld.shared.f32 {d}, {addr}")

        def store(buf, i, v):
            _, addr = self.address(buf, (i,))
            self.ins(f"st.shared.f32 {addr}, {v}")

        for buf in t.bufs:
            if self.dtypes[buf] != "f32":
                raise Unsupported(f"a tree over {self.dtypes[buf]}")
        zero = Linear()
        if t.tid_var == "warp" and t.length <= 32:
            lane = self.ints["lane"]

            def write_back():
                for b, st in zip(t.bufs, t.state, strict=True):
                    store(b, zero, self.vals[st])

            def fold():
                slot = self.rem(lane, Linear(const=t.length))
                for buf, st in zip(t.bufs, t.state, strict=True):
                    load(buf, slot, self.dest(st))
                self.shuffle(t.state, t.state_b, t.combine_states, t.length, t.dtype)
                self.branch(self.cmp("eq", lane, zero), write_back)

            self.branch(self.cmp("eq", tid, zero), fold)
            self.ins("bar.sync 0")
        else:
            k = t.length // 2
            while k:

                def step(k=k):
                    for buf, st, sb in zip(t.bufs, t.state, t.state_b, strict=True):
                        load(buf, tid, self.dest(st))
                        load(buf, tid + Linear(const=k), self.dest(sb))
                    self.merge(t.combine_states, t.dtype)
                    for buf, st in zip(t.bufs, t.state, strict=True):
                        store(buf, tid, self.vals[st])

                self.branch(self.cmp("lt", tid, Linear(const=k)), step)
                self.ins("bar.sync 0")
                k //= 2
        for buf, st in zip(t.bufs, t.state, strict=True):
            load(buf, zero, self.dest(st))
