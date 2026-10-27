"""Kernel IR → PTX, printed directly.

One walk over the ``KernelOp`` body, one translation per statement kind, PTX text out. ptxas does register allocation
and scheduling downstream, so registers are handed out fresh and declared by count at the top.

Two things the emitter does itself, because ptxas does not:

- **Index folding.** Every integer index is held as a linear form — registers times constants, plus a constant — and
  only turned into instructions when an instruction needs it. Constants fold, terms cancel, the same form is computed
  once per scope, and the constant part of an address becomes the instruction's immediate offset (``[%rd1+512]``).
- **Unrolling.** An innermost loop the IR marks ``unroll`` is copied once per iteration with its axis a constant; any
  other innermost loop, counted or strided, runs ``UNROLL`` copies per trip before a remainder loop, so that many loads
  are in flight per thread. Only innermost loops unroll, as nvcc's front end does, so copies never multiply.

A statement outside the set below raises :class:`Unsupported`, naming it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from math import prod

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import pack_smem
STATIC_SMEM_CAP = 48 * 1024  # bytes of static shared memory a CTA may declare; past it, one dynamic pool
_BLOCK_SIZE = 256  # threads per CTA of a scalar tile (one thread per element)

# The PTX ISA version each capability first needs; ptxas accepts any later one.
_PTX_FOR_CAP = ((120, 87), (100, 86), (90, 78), (89, 78), (80, 70), (75, 63), (70, 60))
UNROLL = 4
_INT32_MAX = 2**31 - 1
# Storage dtype → (PTX load/store type, bytes).
_STORAGE = {"f32": ("f32", 4), "f16": ("b16", 2), "bf16": ("b16", 2)}
_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_FOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
LOG2E = 1.4426950408889634


class Unsupported(Exception):
    """A Kernel IR construct this emitter does not translate."""


def _fimm(v: float) -> str:
    """A PTX f32 immediate: the bit pattern, so every value (including -inf) prints exactly."""
    return f"0f{struct.unpack('<I', struct.pack('<f', float(v)))[0]:08X}"


@dataclass(frozen=True)
class _Linear:
    """``sum(coeff * register) + const`` over s32 registers. A constant is the form with no terms."""

    terms: tuple[tuple[str, int], ...] = ()
    const: int = 0

    @staticmethod
    def of(reg: str) -> _Linear:
        return _Linear(((reg, 1),))

    @property
    def is_const(self) -> bool:
        return not self.terms

    def __add__(self, other: _Linear) -> _Linear:
        acc = dict(self.terms)
        for r, c in other.terms:
            acc[r] = acc.get(r, 0) + c
        return _Linear(tuple(sorted((r, c) for r, c in acc.items() if c)), self.const + other.const)

    def scale(self, k: int) -> _Linear:
        return _Linear(tuple((r, c * k) for r, c in self.terms) if k else (), self.const * k)

    def without_const(self) -> _Linear:
        return _Linear(self.terms)


@dataclass
class _Scope:
    """Values computed in one block of control flow; a block's values do not outlive it."""

    forms: dict = field(default_factory=dict)  # _Linear / (op, key) → register


class _Emitter:
    def __init__(self, kernel, tensors, cap):
        self.kernel, self.cap = kernel, cap
        self.lines: list[str] = []
        self.counts = {"p": 0, "r": 0, "rd": 0, "f": 0, "h": 0}
        self.labels = 0
        self.scopes = [_Scope()]
        self.ints: dict[str, _Linear] = {}  # axis variables and integer lets
        self.vals: dict[str, str] = {}  # float SSA names → their one f32 register
        self.shapes = {n: tuple(self.extent_of(d) for d in t.shape) for n, t in tensors.items()}
        self.dtypes = {n: str(t.dtype) for n, t in tensors.items()}
        for n, dt in self.dtypes.items():
            if dt not in _STORAGE:
                raise Unsupported(f"buffer {n} of dtype {dt}")
            if prod(d for d in self.shapes[n] if isinstance(d, int)) > _INT32_MAX:
                raise Unsupported(f"buffer {n} past 32-bit indexing")
        self.buffers = tuple(tensors)  # the launch order: the caller names every buffer in it
        self.smem: dict[str, _Linear] = {}  # shared buffer → byte address (32-bit shared space)
        self.globals: dict[str, str] = {}  # global buffer → 64-bit base register
        self.decls: list[str] = []
        offsets, total = pack_smem(list(kernel.smem_buffers.values()))
        self.pool = offsets if total > STATIC_SMEM_CAP else None

    # ---- registers, labels, instructions ------------------------------------------------------------------------

    def reg(self, kind: str) -> str:
        self.counts[kind] += 1
        return f"%{kind}{self.counts[kind]}"

    def label(self) -> str:
        self.labels += 1
        return f"$L{self.labels}"

    def ins(self, text: str) -> None:
        self.lines.append(f"\t{text};")

    def mark(self, label: str) -> None:
        self.lines.append(f"{label}:")

    def push(self) -> None:
        self.scopes.append(_Scope())

    def pop(self) -> None:
        self.scopes.pop()

    def cached(self, key):
        for s in reversed(self.scopes):
            if key in s.forms:
                return s.forms[key]
        return None

    def remember(self, key, reg: str) -> str:
        self.scopes[-1].forms[key] = reg
        return reg

    # ---- integers: linear forms, folded and computed once -----------------------------------------------------

    def extent_of(self, d):
        if d.is_static:
            return d.as_static()
        try:
            return d.as_atom_name()
        except (TypeError, ValueError, AttributeError):
            raise Unsupported(f"composite extent {d}") from None

    def size(self, d) -> _Linear:
        return _Linear(const=d) if isinstance(d, int) else self.ints[d]

    def lin(self, e) -> _Linear:
        """An integer expression as a linear form; non-linear parts become one register each."""
        if isinstance(e, _Linear):
            return e
        k = type(e).__name__
        if k == "Literal":
            if isinstance(e.value, float) and not e.value.is_integer():
                raise Unsupported(f"float literal {e.value} as an index")
            return _Linear(const=int(e.value))
        if k == "Var":
            if e.name in self.ints:
                return self.ints[e.name]
            if e.name in self.vals:  # a gathered index: the loaded value truncated, as C's (int) cast does
                r = self.reg("r")
                self.ins(f"cvt.rzi.s32.f32 {r}, {self.vals[e.name]}")
                return _Linear.of(r)
            raise Unsupported(f"unbound variable {e.name}")
        if k == "BinaryExpr" and e.op in ("+", "-", "*", "/", "//", "%"):
            return self.arith(e.op, self.lin(e.left), self.lin(e.right))
        if k == "TernaryExpr":
            p = self.pred(e.cond)
            if p is True or p is False:
                return self.lin(e.if_true if p else e.if_false)
            a, b = self.lin(e.if_true), self.lin(e.if_false)
            key = ("selp", p, a, b)
            if (r := self.cached(key)) is None:
                r = self.remember(key, self.reg("r"))
                self.ins(f"selp.s32 {r}, {self.in_ireg(a)}, {self.value(b)}, {p}")
            return _Linear.of(r)
        raise Unsupported(f"integer expression {k} {getattr(e, 'op', '')}".strip())

    def arith(self, op: str, a: _Linear, b: _Linear) -> _Linear:
        """``a op b``: folded when both are constant, kept linear for ``+ - *`` by a constant, else one cached
        instruction."""
        if a.is_const and b.is_const:
            if op in _FOLD:
                return _Linear(const=_FOLD[op](a.const, b.const))
            if b.const == 0:
                raise Unsupported("division by zero in an index")
            q = abs(a.const) // abs(b.const) * (1 if (a.const >= 0) == (b.const >= 0) else -1)  # C truncation
            return _Linear(const=q if op != "%" else a.const - q * b.const)
        if op == "+":
            return a + b
        if op == "-":
            return a + b.scale(-1)
        if op == "*" and (a.is_const or b.is_const):
            return b.scale(a.const) if a.is_const else a.scale(b.const)
        if op in ("/", "//") and b.is_const and b.const == 1:
            return a
        if op == "%" and b.is_const and b.const == 1:
            return _Linear()
        ins = {"*": "mul.lo.s32", "/": "div.s32", "//": "div.s32", "%": "rem.s32"}[op]
        key = (ins, a, b)
        if (r := self.cached(key)) is None:
            r = self.remember(key, self.reg("r"))
            self.ins(f"{ins} {r}, {self.in_ireg(a)}, {self.value(b)}")
        return _Linear.of(r)

    def value(self, f: _Linear) -> str:
        """A form as an instruction operand: an immediate, a register, or one register computing it."""
        if f.is_const:
            return str(f.const)
        if f.terms == ((f.terms[0][0], 1),) and f.const == 0:
            return f.terms[0][0]
        if (r := self.cached(f)) is not None:
            return r
        r = self.reg("r")
        (first, c0), *rest = f.terms
        self.ins(f"mul.lo.s32 {r}, {first}, {c0}" if c0 != 1 else f"mov.b32 {r}, {first}")
        for reg_, c in rest:
            self.ins(f"mad.lo.s32 {r}, {reg_}, {c}, {r}" if c != 1 else f"add.s32 {r}, {r}, {reg_}")
        if f.const:
            self.ins(f"add.s32 {r}, {r}, {f.const}")
        return self.remember(f, r)

    def in_ireg(self, f: _Linear) -> str:
        """A form as a register even when it is a constant: the first operand of most PTX instructions."""
        v = self.value(f)
        if not f.is_const:
            return v
        r = self.reg("r")
        self.ins(f"mov.b32 {r}, {v}")
        return r

    def pred(self, e):
        """A predicate: ``True`` / ``False`` when it folds, else a ``.pred`` register."""
        k = type(e).__name__
        if k == "BinaryExpr" and e.op in _CMP:
            return self.cmp(_CMP[e.op], self.lin(e.left), self.lin(e.right))
        if k == "BinaryExpr" and e.op in ("&&", "||"):
            a, b = self.pred(e.left), self.pred(e.right)
            if isinstance(a, bool) or isinstance(b, bool):
                const, other = (a, b) if isinstance(a, bool) else (b, a)
                if e.op == "&&":
                    return other if const else False
                return True if const else other
            p = self.reg("p")
            self.ins(f"{'and' if e.op == '&&' else 'or'}.pred {p}, {a}, {b}")
            return p
        f = self.lin(e)
        if f.is_const:
            return f.const != 0
        p = self.reg("p")
        self.ins(f"setp.ne.s32 {p}, {self.value(f)}, 0")
        return p

    # ---- control flow ------------------------------------------------------------------------------------------

    def branch(self, p, then, otherwise=None) -> None:
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

    def block(self, fn) -> None:
        self.push()
        fn()
        self.pop()

    def counted(self, lo: _Linear, hi: _Linear, step: int, body) -> None:
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
        self.block(lambda: body(_Linear.of(i)))
        self.ins(f"add.s32 {i}, {i}, {step}")
        p = self.reg("p")
        self.ins(f"setp.lt.s32 {p}, {i}, {self.value(hi)}")
        self.ins(f"@{p} bra {top}")
        self.mark(done)

    def with_axis(self, name: str, value: _Linear, body) -> None:
        saved = self.ints.get(name)
        self.ints[name] = value
        body()
        if saved is None:
            del self.ints[name]
        else:
            self.ints[name] = saved

    # ---- floats -----------------------------------------------------------------------------------------------

    def fval(self, arg) -> str:
        """A float operand: an SSA name's register, or an immediate for a numeric literal."""
        if isinstance(arg, str):
            if arg in self.vals:
                return self.vals[arg]
            if arg in self.ints:
                r = self.reg("f")
                self.ins(f"cvt.rn.f32.s32 {r}, {self.value(self.ints[arg])}")
                return r
            try:
                return _fimm(float(arg))
            except ValueError:
                raise Unsupported(f"unbound value {arg}") from None
        return _fimm(arg)

    def target(self, name: str) -> str:
        """The one register an SSA name lives in; a redefinition writes into it again."""
        if name not in self.vals:
            self.vals[name] = self.reg("f")
        return self.vals[name]

    def exp(self, d: str, x: str, fast: bool) -> None:
        """``__expf`` is ``ex2.approx`` of x·log2(e). ``expf`` is 2^n · 2^r: n = round(x·log2 e), clamped to the normal
        range by a saturating step and rounded with a magic number, becomes the exponent bits of a float (``shl 23``);
        r = x·log2 e − n, computed with log2 e split in two parts, is the only ``ex2.approx`` argument. Eight plain
        instructions and one transcendental, the sequence CUDA's ``expf`` is."""
        if fast:
            self.ins(f"mul.f32 {d}, {x}, {_fimm(LOG2E)}")
            self.ins(f"ex2.approx.f32 {d}, {d}")
            return
        t, j, n, r = (self.reg("f") for _ in range(4))
        self.ins(f"fma.rn.f32 {t}, {x}, 0f3BBB989D, 0f3F000000")  # x·log2e/252 + 1/2
        self.ins(f"cvt.sat.f32.f32 {t}, {t}")  # clamped to [0, 1], so n stays in [-126, 126]
        self.ins(f"fma.rm.f32 {j}, {t}, 0f437C0000, 0f4B400001")  # 252·t + 1.5·2^23 + 1: n + 127 in the low bits
        self.ins(f"add.rn.f32 {n}, {j}, 0fCB40007F")  # n as a float
        self.ins(f"neg.f32 {n}, {n}")
        self.ins(f"fma.rn.f32 {r}, {x}, 0f3FB8AA3B, {n}")  # x·log2e (high part) − n
        self.ins(f"fma.rn.f32 {r}, {x}, 0f32A57060, {r}")  # + x·log2e (low part)
        self.ins(f"ex2.approx.ftz.f32 {r}, {r}")
        self.ins(f"shl.b32 {j}, {j}, 23")  # 2^n: n + 127 in the exponent field
        self.ins(f"mul.rn.f32 {d}, {r}, {j}")

    def elementwise(self, d: str, op: str, a: list[str]) -> None:
        one, zero = _fimm(1.0), _fimm(0.0)
        simple = {
            "add": "add.f32",
            "subtract": "sub.f32",
            "multiply": "mul.f32",
            "maximum": "max.f32",
            "minimum": "min.f32",
            "divide": "div.rn.f32",
        }
        unary = {
            "negative": "neg.f32",
            "abs": "abs.f32",
            "reciprocal": "rcp.rn.f32",
            "sqrt": "sqrt.rn.f32",
            "rsqrt": "rsqrt.approx.f32",
            "tanh": "tanh.approx.f32",
            "copy": "mov.f32",
            "pad": "mov.f32",
            "floor": "cvt.rmi.f32.f32",
            "ceil": "cvt.rpi.f32.f32",
        }
        if op in simple:
            self.ins(f"{simple[op]} {d}, {a[0]}, {a[1]}")
        elif op in unary:
            self.ins(f"{unary[op]} {d}, {a[0]}")
        elif op == "next":
            self.ins(f"mov.f32 {d}, {a[1]}")
        elif op == "square":
            self.ins(f"mul.f32 {d}, {a[0]}, {a[0]}")
        elif op == "relu":
            self.ins(f"max.f32 {d}, {a[0]}, {zero}")
        elif op in ("exp", "exp_fast"):
            self.exp(d, a[0], op == "exp_fast")
        elif op == "log":
            self.ins(f"lg2.approx.f32 {d}, {a[0]}")
            self.ins(f"mul.f32 {d}, {d}, {_fimm(1 / LOG2E)}")
        elif op in ("sigmoid", "silu"):
            t = self.reg("f")
            self.ins(f"neg.f32 {t}, {a[0]}")
            self.exp(t, t, False)
            self.ins(f"add.f32 {t}, {t}, {one}")
            self.ins(f"{simple['divide']} {d}, {a[0] if op == 'silu' else one}, {t}")
        elif op == "where":
            p = self.reg("p")
            self.ins(f"setp.neu.f32 {p}, {a[0]}, {zero}")
            self.ins(f"selp.f32 {d}, {a[1]}, {a[2]}, {p}")
        else:
            raise Unsupported(f"elementwise op {op}")

    def round_to(self, d: str, dtype) -> None:
        """An ``Assign`` at a 16-bit dtype rounds its f32 result to it."""
        name = str(dtype)
        if name == "f16":
            h = self.reg("h")
            self.ins(f"cvt.rn.f16.f32 {h}, {d}")
            self.ins(f"cvt.f32.f16 {d}, {h}")
        elif name == "bf16":
            h = self.reg("h")
            self.ins(f"cvt.rn.bf16.f32 {h}, {d}")
            r = self.reg("r")
            self.ins(f"cvt.u32.u16 {r}, {h}")
            self.ins(f"shl.b32 {r}, {r}, 16")
            self.ins(f"mov.b32 {d}, {r}")

    # ---- memory -----------------------------------------------------------------------------------------------

    def flat(self, buf: str, index) -> _Linear:
        """Row-major element offset of ``buf[index...]``."""
        shape = self.shapes[buf]
        if len(index) not in (1, len(shape)):
            raise Unsupported(f"{buf}: {len(index)} indices for rank {len(shape)}")
        if len(index) == 1:
            return self.lin(index[0])
        off, stride = _Linear(), _Linear(const=1)
        for e, d in zip(reversed(index), reversed(shape), strict=True):
            i = self.lin(e)
            if stride.is_const:
                off = off + i.scale(stride.const)
            else:
                off = off + self.lin_mul(i, stride)
            stride = stride.scale(d) if isinstance(d, int) else self.lin_mul(stride, self.ints[d])
        return off

    def lin_mul(self, a: _Linear, b: _Linear) -> _Linear:
        if a.is_const or b.is_const:
            return b.scale(a.const) if a.is_const else a.scale(b.const)
        key = ("mul.lo.s32", a, b)
        if (r := self.cached(key)) is None:
            r = self.remember(key, self.reg("r"))
            self.ins(f"mul.lo.s32 {r}, {self.value(a)}, {self.value(b)}")
        return _Linear.of(r)

    def address(self, buf: str, index, extra: int = 0) -> tuple[str, str]:
        """``(space, "[base+offset]")`` for element ``index`` of ``buf``: the constant part folds into the
        immediate, the rest is computed once per scope."""
        nbytes = _STORAGE[self.dtypes[buf]][1]
        elems = self.flat(buf, index) + _Linear(const=extra)
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

    def load(self, s) -> None:
        if s.input in self.smem or s.input in self.globals:
            ty, _ = _STORAGE[self.dtypes[s.input]]
        else:
            raise Unsupported(f"load from {s.input}")
        regs = [self.target(n) for n in s.names]
        space, addr = self.address(s.input, s.index)
        if space == "global" and s.input not in self.kernel.outputs:
            space = "global.nc"  # read-only for the whole kernel: the non-coherent cache, as ``const __restrict__`` gives
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

    def widen(self, d: str, h: str, dt: str) -> None:
        if dt == "f16":
            self.ins(f"cvt.f32.f16 {d}, {h}")
        else:
            r = self.reg("r")
            self.ins(f"cvt.u32.u16 {r}, {h}")
            self.ins(f"shl.b32 {r}, {r}, 16")
            self.ins(f"mov.b32 {d}, {r}")

    def store(self, s) -> None:
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
                self.ins(f"cvt.rn.{'f16' if dt == 'f16' else 'bf16'}.f32 {h}, {v}")
                self.ins(f"st.{space}.b16 {addr}, {h}")

    def in_reg(self, v: str) -> str:
        if v.startswith("%"):
            return v
        r = self.reg("f")
        self.ins(f"mov.f32 {r}, {v}")
        return r

    # ---- statements --------------------------------------------------------------------------------------------

    def body(self, stmts) -> None:
        for s in stmts:
            k = type(s).__name__
            fn = getattr(self, f"s_{k}", None)
            if fn is None:
                raise Unsupported(f"statement {k}")
            fn(s)

    def s_Tile(self, t) -> None:
        if t.aux_threads:
            raise Unsupported("warp-specialized tile")
        tid = self.reg("r")
        self.ins(f"mov.u32 {tid}, %tid.x")
        lane, warp = self.reg("r"), self.reg("r")
        self.ins(f"and.b32 {lane}, {tid}, 31")
        self.ins(f"shr.u32 {warp}, {tid}, 5")
        self.ints["lane"], self.ints["warp"] = _Linear.of(lane), _Linear.of(warp)
        cells = t.block_threads if t.block_threads is not None else _BLOCK_SIZE
        ctaid = self.reg("r")
        self.ins(f"mov.u32 {ctaid}, %ctaid.x")
        gid = self.reg("r")
        self.ins(f"mad.lo.s32 {gid}, {ctaid}, {cells}, {tid}")  # blockDim.x is the tile's cell count
        extents = [self.size(self.extent_of(a.extent)) for a in t.axes]
        total = _Linear(const=1)
        for e in extents:
            total = self.lin_mul(total, e)
        if total.is_const and total.const > _INT32_MAX:
            raise Unsupported("a grid past 32-bit cell ids")
        g = _Linear.of(gid)

        def decode():
            stride = _Linear(const=1)
            for i in range(len(t.axes) - 1, -1, -1):
                q = self.div(g, stride)
                self.ints[t.axes[i].name] = q if i == 0 else self.rem(q, extents[i])
                stride = self.lin_mul(stride, extents[i])
            self.raster(t, g)
            self.body(t.body)

        exact = total.is_const and total.const % cells == 0
        self.branch(True if exact else self.cmp("lt", g, total), decode)

    def cmp(self, op: str, a: _Linear, b: _Linear):
        """``setp`` of two forms; a constant on the left swaps sides so it lands where PTX takes an immediate."""
        if a.is_const and b.is_const:
            return {
                "lt": a.const < b.const,
                "le": a.const <= b.const,
                "gt": a.const > b.const,
                "ge": a.const >= b.const,
                "eq": a.const == b.const,
                "ne": a.const != b.const,
            }[op]
        if a.is_const:
            a, b, op = b, a, {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}.get(op, op)
        p = self.reg("p")
        self.ins(f"setp.{op}.s32 {p}, {self.value(a)}, {self.value(b)}")
        return p

    def div(self, a: _Linear, b: _Linear) -> _Linear:
        return self.arith("/", a, b)

    def rem(self, a: _Linear, b: _Linear) -> _Linear:
        return self.arith("%", a, b)

    def raster(self, t, gid: _Linear) -> None:
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
        sub = self.div(gid, _Linear(const=prod(ext[ni + 1 :])))
        if mi != 0:
            sub = self.rem(sub, _Linear(const=ea * eb))
        group = self.div(sub, _Linear(const=g * eb))
        within = self.rem(sub, _Linear(const=g * eb))
        rest = self.reg("r")
        self.ins(f"min.s32 {rest}, {self.in_ireg(_Linear(const=ea) + group.scale(-g))}, {g}")
        size = _Linear.of(rest)
        self.ints[t.axes[gi].name] = group.scale(g) + self.rem(within, size)
        self.ints[t.axes[oi].name] = self.div(within, size)

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
                self.ins(f"mov.f32 {self.target(a.name)}, {_fimm(a.op.identity)}")

    def s_Loop(self, s) -> None:
        if s.carries:
            raise Unsupported("loop-carried state")
        self.seed(s)
        ext = self.size(self.extent_of(s.axis.extent))
        if s.unroll and ext.is_const and _innermost(s):
            # The copies run straight-line in one block, so they share one scope: whatever does not depend on the
            # axis is computed once for all of them.
            def copies():
                for v in range(ext.const):
                    self.with_axis(s.axis.name, _Linear(const=v), lambda: self.body(s.body))

            self.block(copies)
            return
        if _innermost(s):
            self.unrolled(s.axis.name, _Linear(), ext, 1, s.body)
            return
        self.counted(_Linear(), ext, 1, lambda i: self.with_axis(s.axis.name, i, lambda: self.body(s.body)))

    def s_StridedLoop(self, s) -> None:
        self.seed(s)
        lo = self.lin(s.start)
        hi = self.size(self.extent_of(s.axis.extent)) if s.end is None else self.lin(s.end)
        step = self.lin(s.step)
        if not step.is_const:
            raise Unsupported("a strided loop with a runtime step")
        if s.end is not None:
            self.ints[f"{s.axis.name}_end"] = hi
        if _innermost(s):
            self.unrolled(s.axis.name, lo, hi, step.const, s.body)
            return
        self.counted(lo, hi, step.const, lambda i: self.with_axis(s.axis.name, i, lambda: self.body(s.body)))

    def unrolled(self, name: str, lo: _Linear, hi: _Linear, step: int, stmts) -> None:
        """``UNROLL`` copies of the body per trip, axis ``i + k·step`` in copy ``k``, while the last copy is still in
        range; then the rest one at a time. The copies share the trip's scope, so they share every address register."""

        def copies(n):
            def run(i):
                for k in range(n):
                    self.with_axis(name, i + _Linear(const=k * step), lambda: self.body(stmts))

            return run

        i = self.reg("r")
        self.ins(f"mov.b32 {i}, {self.value(lo)}")
        self.counted_from(i, hi + _Linear(const=-(UNROLL - 1) * step), UNROLL * step, copies(UNROLL))
        if not (lo.is_const and hi.is_const and (hi.const - lo.const) % (UNROLL * step) == 0):
            self.counted_from(i, hi, step, copies(1))

    def counted_from(self, i: str, hi: _Linear, step: int, body) -> None:
        """``while (i < hi) { body(i); i += step; }`` on an existing counter register."""
        top, done = self.label(), self.label()
        p = self.reg("p")
        self.ins(f"setp.ge.s32 {p}, {i}, {self.value(hi)}")
        self.ins(f"@{p} bra {done}")
        self.mark(top)
        self.block(lambda: body(_Linear.of(i)))
        self.ins(f"add.s32 {i}, {i}, {step}")
        p = self.reg("p")
        self.ins(f"setp.lt.s32 {p}, {i}, {self.value(hi)}")
        self.ins(f"@{p} bra {top}")
        self.mark(done)

    def s_Cond(self, s) -> None:
        self.branch(self.pred(s.cond), lambda: self.body(s.body), (lambda: self.body(s.else_body)) if s.else_body else None)

    def s_Load(self, s) -> None:
        self.load(s)

    def s_Write(self, s) -> None:
        self.store(s)

    def s_Assign(self, s) -> None:
        args = [self.fval(a) for a in s.args]
        d = self.target(s.name)
        self.elementwise(d, s.op.name, args)
        self.round_to(d, s.dtype)

    def s_Affine(self, s) -> None:
        self.body(s.as_assigns())

    def s_Accum(self, s) -> None:
        op = {"add": "add.f32", "multiply": "mul.f32", "maximum": "max.f32", "minimum": "min.f32"}.get(s.op.name)
        if op is None:
            raise Unsupported(f"accumulator {s.op.name}")
        d = self.target(s.name)
        base = self.fval(s.base) if s.base is not None else d
        self.ins(f"{op} {d}, {base}, {self.fval(s.value)}")

    def s_Select(self, s) -> None:
        d = self.target(s.name)
        self.ins(f"mov.f32 {d}, {self.fval(s.branches[-1].value)}")
        for b in reversed(s.branches[:-1]):
            p = self.pred(b.select)
            if p is True:
                self.ins(f"mov.f32 {d}, {self.fval(b.value)}")
            elif p is not False:
                self.ins(f"selp.f32 {d}, {self.fval(b.value)}, {d}, {p}")

    def s_Let(self, s) -> None:
        v = s.value
        if type(v).__name__ == "Literal" and isinstance(v.value, float) or str(s.dtype) in ("f32", "f16", "bf16"):
            if type(v).__name__ != "Literal":
                raise Unsupported("a float let of an expression")
            self.ins(f"mov.f32 {self.target(s.name)}, {_fimm(v.value)}")
        else:
            self.ints[s.name] = self.lin(v)

    def s_Init(self, s) -> None:
        self.ins(f"mov.f32 {self.target(s.name)}, {_fimm(s.identity)}")

    def s_Smem(self, s) -> None:
        dt = {"float": "f32", "__half": "f16", "__nv_bfloat16": "bf16"}.get(s.dtype)
        if dt is None:
            raise Unsupported(f"shared buffer of {s.dtype}")
        self.shapes[s.name], self.dtypes[s.name] = tuple(int(e) for e in s.extents), dt
        r = self.reg("r")
        if self.pool is not None:
            self.ins(f"mov.u32 {r}, _smem_pool")
            self.smem[s.name] = _Linear.of(r) + _Linear(const=self.pool[s.name])
            return
        nbytes = prod(int(e) for e in s.extents) * _STORAGE[dt][1]
        self.decls.append(f"\t.shared .align {max(s.align or 0, _STORAGE[dt][1])} .b8 {s.name}[{nbytes}];")
        self.ins(f"mov.u32 {r}, {s.name}")
        self.smem[s.name] = _Linear.of(r)

    def s_Sync(self, s) -> None:
        if s.barrier_id:
            raise Unsupported("named barrier")
        self.ins("bar.warp.sync -1" if s.warp else "bar.sync 0")

    def s_WarpShuffle(self, s) -> None:
        self.shuffle(s.state, s.state_b, s.combine_states, s.length, s.dtype)

    def shuffle(self, state, state_b, program, length, dtype) -> None:
        """The XOR butterfly over ``length`` lanes: each step pulls every component from the lane ``k`` apart and
        folds it in with the stored combine, so every lane ends holding the whole combine."""
        k = length // 2
        while k:
            mask = self.reg("r")
            self.ins(f"activemask.b32 {mask}")
            for st, sb in zip(state, state_b, strict=True):
                self.ins(f"shfl.sync.bfly.b32 {self.target(sb)}, {self.fval(st)}, {k}, 31, {mask}")
            self.merge(program, dtype)
            k //= 2

    def merge(self, program, dtype) -> None:
        for a in program:
            self.elementwise(self.target(a.name), a.op.name, [self.fval(x) for x in a.args])
            self.round_to(self.vals[a.name], dtype)

    def s_TreeHalve(self, t) -> None:
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
        zero = _Linear()
        if t.tid_var == "warp" and t.length <= 32:
            # One partial per warp: warp 0 folds the slab with a register butterfly, then one barrier.
            lane = self.ints["lane"]

            def fold():
                slot = self.rem(lane, _Linear(const=t.length))
                for buf, st in zip(t.bufs, t.state, strict=True):
                    load(buf, slot, self.target(st))
                self.shuffle(t.state, t.state_b, t.combine_states, t.length, t.dtype)
                self.branch(
                    self.cmp("eq", lane, zero), lambda: [store(b, zero, self.vals[st]) for b, st in zip(t.bufs, t.state, strict=True)]
                )

            self.branch(self.cmp("eq", tid, zero), fold)
            self.ins("bar.sync 0")
        else:
            k = t.length // 2
            while k:

                def step(k=k):
                    for buf, st, sb in zip(t.bufs, t.state, t.state_b, strict=True):
                        load(buf, tid, self.target(st))
                        load(buf, tid + _Linear(const=k), self.target(sb))
                    self.merge(t.combine_states, t.dtype)
                    for buf, st in zip(t.bufs, t.state, strict=True):
                        store(buf, tid, self.vals[st])

                self.branch(self.cmp("lt", tid, _Linear(const=k)), step)
                self.ins("bar.sync 0")
                k //= 2
        for buf, st in zip(t.bufs, t.state, strict=True):
            load(buf, zero, self.target(st))

    def s_CpAsyncCopy(self, s) -> None:
        if s.swizzle != "NONE":
            raise Unsupported("swizzled cp.async")
        index = s.smem_index
        if s.lane_index is not None:
            # The lane's own offset inside the slab tile: ``row * cols + col`` on top of the tile's start.
            cols = self.shapes[s.smem][-1]
            index = (self.flat(s.smem, s.smem_index) + self.lin(s.lane_index[0]).scale(cols) + self.lin(s.lane_index[1]),)
        _, dst = self.address(s.smem, index)
        _, src = self.address(s.src, s.src_index)
        self.ins(f"cp.async.{'cg' if s.nbytes == 16 else 'ca'}.shared.global {dst}, {src}, {s.nbytes}")

    def s_CpAsyncCommit(self, _) -> None:
        self.ins("cp.async.commit_group")

    def s_CpAsyncWait(self, s) -> None:
        self.ins(f"cp.async.wait_group {s.group}")

    # ---- the kernel --------------------------------------------------------------------------------------------

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
        if self.cap >= (9, 0):
            # Wait for the grid ahead before the first read, as every CUDA kernel does: its presence is what lets the
            # runtime start this launch while that grid drains.
            self.ins("griddepcontrol.wait")
        self.body(k.body)
        sm = self.cap[0] * 10 + self.cap[1]
        version = next((v for c, v in _PTX_FOR_CAP if sm >= c), None)
        if version is None:
            raise Unsupported(f"compute capability {self.cap}")
        regs = {"p": ".pred", "r": ".b32", "rd": ".b64", "f": ".f32", "h": ".b16"}
        head = [f".version {version // 10}.{version % 10}", f".target sm_{sm}", ".address_size 64", ""]
        if self.pool is not None:
            align = max(16, *(s.align or 0 for s in k.smem_buffers.values()))
            head.append(f".extern .shared .align {align} .b8 _smem_pool[];")
        head += [f".visible .entry {name}(", ",\n".join(params), ")", "{"]
        head += [f"\t.reg {t} %{p}<{self.counts[p] + 1}>;" for p, t in regs.items()]
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


def _innermost(loop) -> bool:
    """Whether no loop runs inside ``loop``: only an innermost loop is unrolled, so copies never multiply."""
    return not any(type(s).__name__ in ("Loop", "StridedLoop") for s in loop.body.iter())


def emit(kernel, tensors, *, cap: tuple[int, int]) -> str:
    """PTX for ``kernel`` (a ``KernelOp``) at compute capability ``cap``. ``tensors`` maps every buffer it reads or
    writes to its ``Tensor``, in launch order (``meta.json``'s ``arg_order``): the kernel's parameters follow it."""
    return _Emitter(kernel, tensors, cap).emit()
