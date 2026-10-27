"""Float math. Every float value lives in one f32 register, named after its SSA name in the IR."""

from .common import LOG2E, Unsupported, fimm

ONE, ZERO = fimm(1.0), fimm(0.0)
# One instruction each: binary ops, then unary ops.
_BINARY = {
    "add": "add.f32",
    "subtract": "sub.f32",
    "multiply": "mul.f32",
    "maximum": "max.f32",
    "minimum": "min.f32",
    "divide": "div.rn.f32",
}
_UNARY = {
    "negative": "neg.f32",
    "abs": "abs.f32",
    "reciprocal": "rcp.rn.f32",
    "sqrt": "sqrt.rn.f32",
    "rsqrt": "rsqrt.approx.f32",
    "copy": "mov.f32",
    "pad": "mov.f32",
    "floor": "cvt.rmi.f32.f32",
    "ceil": "cvt.rpi.f32.f32",
}
_ACCUM = {"add": "add.f32", "multiply": "mul.f32", "maximum": "max.f32", "minimum": "min.f32"}


class Floats:
    # ---- operands ----------------------------------------------------------------------------------------------

    def fval(self, arg) -> str:
        """A float operand: an SSA name's register, an integer converted, or an immediate for a numeric literal."""
        if isinstance(arg, str):
            if arg in self.vals:
                return self.vals[arg]
            if arg in self.ints:
                r = self.reg("f")
                self.ins(f"cvt.rn.f32.s32 {r}, {self.value(self.ints[arg])}")
                return r
            try:
                return fimm(float(arg))
            except ValueError:
                raise Unsupported(f"unbound value {arg}") from None
        return fimm(arg)

    def dest(self, name: str) -> str:
        """The one register an SSA name lives in; a redefinition writes into it again."""
        if name not in self.vals:
            self.vals[name] = self.reg("f")
        return self.vals[name]

    def in_reg(self, v: str) -> str:
        """An operand as a register: an immediate is moved into one first."""
        if v.startswith("%"):
            return v
        r = self.reg("f")
        self.ins(f"mov.f32 {r}, {v}")
        return r

    # ---- the ops -----------------------------------------------------------------------------------------------

    def elementwise(self, d: str, op: str, a: list[str]) -> None:
        if op in _BINARY:
            self.ins(f"{_BINARY[op]} {d}, {a[0]}, {a[1]}")
        elif op in _UNARY:
            self.ins(f"{_UNARY[op]} {d}, {a[0]}")
        elif op == "tanh":
            self.tanh(d, a[0])
        elif op == "next":
            self.ins(f"mov.f32 {d}, {a[1]}")
        elif op == "square":
            self.ins(f"mul.f32 {d}, {a[0]}, {a[0]}")
        elif op == "relu":
            self.ins(f"max.f32 {d}, {a[0]}, {ZERO}")
        elif op in ("exp", "exp_fast"):
            self.exp(d, a[0], op == "exp_fast")
        elif op == "log":
            self.ins(f"lg2.approx.f32 {d}, {a[0]}")
            self.ins(f"mul.f32 {d}, {d}, {fimm(1 / LOG2E)}")
        elif op in ("sigmoid", "silu"):  # x / (1 + e^-x), or 1 / (1 + e^-x)
            t = self.reg("f")
            self.ins(f"neg.f32 {t}, {a[0]}")
            self.exp(t, t, False)
            self.ins(f"add.f32 {t}, {t}, {ONE}")
            self.ins(f"div.rn.f32 {d}, {a[0] if op == 'silu' else ONE}, {t}")
        elif op == "where":
            p = self.reg("p")
            self.ins(f"setp.neu.f32 {p}, {a[0]}, {ZERO}")
            self.ins(f"selp.f32 {d}, {a[1]}, {a[2]}, {p}")
        else:
            raise Unsupported(f"elementwise op {op}")

    def exp(self, d: str, x: str, fast: bool) -> None:
        """``__expf`` is ``ex2.approx`` of x·log2(e). ``expf`` is 2^n · 2^r: n = round(x·log2 e), clamped to the normal
        range by a saturating step and rounded with a magic number, becomes the exponent bits of a float (``shl 23``);
        r = x·log2 e − n, computed with log2 e split in two parts, is the only ``ex2.approx`` argument. Eight plain
        instructions and one transcendental, the sequence CUDA's ``expf`` is."""
        if fast:
            self.ins(f"mul.f32 {d}, {x}, {fimm(LOG2E)}")
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

    def tanh(self, d: str, x: str) -> None:
        """``tanh.approx`` where the GPU has it (sm_75+). Before that: with t = e^(-2|x|), which never overflows,
        tanh|x| = (1 − t) / (1 + t), and the sign of x copied back on."""
        if self.arch.has_tanh:
            self.ins(f"tanh.approx.f32 {d}, {x}")
            return
        t, n = self.reg("f"), self.reg("f")
        self.ins(f"abs.f32 {t}, {x}")
        self.ins(f"mul.f32 {t}, {t}, {fimm(-2 * LOG2E)}")
        self.ins(f"ex2.approx.ftz.f32 {t}, {t}")  # t = e^(-2|x|), in (0, 1]
        self.ins(f"sub.f32 {n}, {ONE}, {t}")
        self.ins(f"add.f32 {t}, {t}, {ONE}")
        self.ins(f"div.rn.f32 {n}, {n}, {t}")
        self.ins(f"copysign.f32 {d}, {x}, {n}")  # |result| of n, sign of x

    # ---- 16-bit floats -----------------------------------------------------------------------------------------

    def bf16_bits(self, f: str) -> str:
        """``f`` rounded to bf16 (round to nearest even), as the low 16 bits of a b32 register: what
        ``cvt.rn.bf16.f32`` does, for GPUs before sm_80. bf16 is the top half of an f32, so rounding adds 0x7FFF plus
        the lowest kept bit, then drops the low half. A NaN stays a NaN."""
        r, t, p = self.reg("r"), self.reg("r"), self.reg("p")
        self.ins(f"mov.b32 {r}, {f}")
        self.ins(f"shr.u32 {t}, {r}, 16")
        self.ins(f"and.b32 {t}, {t}, 1")
        self.ins(f"add.s32 {t}, {t}, 32767")
        self.ins(f"add.s32 {t}, {r}, {t}")
        self.ins(f"shr.u32 {t}, {t}, 16")
        self.ins(f"setp.nan.f32 {p}, {f}, {f}")
        self.ins(f"selp.b32 {t}, 32704, {t}, {p}")  # 0x7FC0, the quiet NaN
        return t

    def to_16(self, h: str, f: str, dt: str) -> None:
        """f32 ``f`` rounded into 16-bit register ``h`` (f16 or bf16)."""
        if dt == "bf16" and not self.arch.has_bf16_cvt:
            self.ins(f"cvt.u16.u32 {h}, {self.bf16_bits(f)}")
        else:
            self.ins(f"cvt.rn.{dt}.f32 {h}, {f}")

    def widen(self, d: str, h: str, dt: str) -> None:
        """16-bit register ``h`` to f32 ``d``. A bf16 is an f32's top half, so it widens by a shift."""
        if dt == "f16":
            self.ins(f"cvt.f32.f16 {d}, {h}")
        else:
            r = self.reg("r")
            self.ins(f"cvt.u32.u16 {r}, {h}")
            self.ins(f"shl.b32 {r}, {r}, 16")
            self.ins(f"mov.b32 {d}, {r}")

    def round_to(self, d: str, dtype) -> None:
        """An ``Assign`` at a 16-bit dtype rounds its f32 result to it, in place."""
        name = str(dtype)
        if name == "f16":
            h = self.reg("h")
            self.ins(f"cvt.rn.f16.f32 {h}, {d}")
            self.ins(f"cvt.f32.f16 {d}, {h}")
        elif name == "bf16" and not self.arch.has_bf16_cvt:
            t = self.bf16_bits(d)
            self.ins(f"shl.b32 {t}, {t}, 16")
            self.ins(f"mov.b32 {d}, {t}")
        elif name == "bf16":
            h = self.reg("h")
            self.ins(f"cvt.rn.bf16.f32 {h}, {d}")
            self.widen(d, h, "bf16")

    # ---- statements --------------------------------------------------------------------------------------------

    def s_Assign(self, s) -> None:
        args = [self.fval(a) for a in s.args]
        d = self.dest(s.name)
        self.elementwise(d, s.op.name, args)
        self.round_to(d, s.dtype)

    def s_Affine(self, s) -> None:
        self.body(s.as_assigns())

    def s_Accum(self, s) -> None:
        op = _ACCUM.get(s.op.name)
        if op is None:
            raise Unsupported(f"accumulator {s.op.name}")
        d = self.dest(s.name)
        base = self.fval(s.base) if s.base is not None else d
        self.ins(f"{op} {d}, {base}, {self.fval(s.value)}")

    def s_Select(self, s) -> None:
        """The last branch is the default; earlier branches overwrite it, the first one last, so the first match wins."""
        d = self.dest(s.name)
        self.ins(f"mov.f32 {d}, {self.fval(s.branches[-1].value)}")
        for b in reversed(s.branches[:-1]):
            p = self.pred(b.select)
            if p is True:
                self.ins(f"mov.f32 {d}, {self.fval(b.value)}")
            elif p is not False:
                self.ins(f"selp.f32 {d}, {self.fval(b.value)}, {d}, {p}")

    def s_Let(self, s) -> None:
        """A float let is a constant; an integer let is a linear form, with no instruction until it is used."""
        v = s.value
        if type(v).__name__ == "Literal" and isinstance(v.value, float) or str(s.dtype) in ("f32", "f16", "bf16"):
            if type(v).__name__ != "Literal":
                raise Unsupported("a float let of an expression")
            self.ins(f"mov.f32 {self.dest(s.name)}, {fimm(v.value)}")
        else:
            self.ints[s.name] = self.lin(v)

    def s_Init(self, s) -> None:
        self.ins(f"mov.f32 {self.dest(s.name)}, {fimm(s.identity)}")
