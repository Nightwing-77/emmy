"""Integer index math: IR expressions to linear forms, folded where constant, computed once per scope where not."""

from .common import Unsupported
from .linear import Linear

CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_FOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
_SWAP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}  # the comparison with its operands swapped
_CONST_CMP = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


class Integers:
    def extent_of(self, d):
        """A dimension's size: an ``int``, or the name of the runtime size it is."""
        if d.is_static:
            return d.as_static()
        try:
            return d.as_atom_name()
        except (TypeError, ValueError, AttributeError):
            raise Unsupported(f"composite extent {d}") from None

    def size(self, d) -> Linear:
        return Linear(const=d) if isinstance(d, int) else self.ints[d]

    def lin(self, e) -> Linear:
        """An integer expression as a linear form; non-linear parts become one register each."""
        if isinstance(e, Linear):
            return e
        k = type(e).__name__
        if k == "Literal":
            if isinstance(e.value, float) and not e.value.is_integer():
                raise Unsupported(f"float literal {e.value} as an index")
            return Linear(const=int(e.value))
        if k == "Var":
            if e.name in self.ints:
                return self.ints[e.name]
            if e.name in self.vals:  # a gathered index: the loaded value truncated, as C's (int) cast does
                r = self.reg("r")
                self.ins(f"cvt.rzi.s32.f32 {r}, {self.vals[e.name]}")
                return Linear.of(r)
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
            return Linear.of(r)
        raise Unsupported(f"integer expression {k} {getattr(e, 'op', '')}".strip())

    def arith(self, op: str, a: Linear, b: Linear) -> Linear:
        """``a op b``: folded when both are constant, kept linear for ``+ - *`` by a constant, else one cached
        instruction."""
        if a.is_const and b.is_const:
            if op in _FOLD:
                return Linear(const=_FOLD[op](a.const, b.const))
            if b.const == 0:
                raise Unsupported("division by zero in an index")
            q = abs(a.const) // abs(b.const) * (1 if (a.const >= 0) == (b.const >= 0) else -1)  # C truncation
            return Linear(const=q if op != "%" else a.const - q * b.const)
        if op == "+":
            return a + b
        if op == "-":
            return a + b.scale(-1)
        if op == "*" and (a.is_const or b.is_const):
            return b.scale(a.const) if a.is_const else a.scale(b.const)
        if op in ("/", "//") and b.is_const and b.const == 1:
            return a
        if op == "%" and b.is_const and b.const == 1:
            return Linear()
        ins = {"*": "mul.lo.s32", "/": "div.s32", "//": "div.s32", "%": "rem.s32"}[op]
        key = (ins, a, b)
        if (r := self.cached(key)) is None:
            r = self.remember(key, self.reg("r"))
            self.ins(f"{ins} {r}, {self.in_ireg(a)}, {self.value(b)}")
        return Linear.of(r)

    def div(self, a: Linear, b: Linear) -> Linear:
        return self.arith("/", a, b)

    def rem(self, a: Linear, b: Linear) -> Linear:
        return self.arith("%", a, b)

    def lin_mul(self, a: Linear, b: Linear) -> Linear:
        """``a * b`` of two forms: a scale when either is constant, else one cached ``mul``."""
        if a.is_const or b.is_const:
            return b.scale(a.const) if a.is_const else a.scale(b.const)
        key = ("mul.lo.s32", a, b)
        if (r := self.cached(key)) is None:
            r = self.remember(key, self.reg("r"))
            self.ins(f"mul.lo.s32 {r}, {self.value(a)}, {self.value(b)}")
        return Linear.of(r)

    def value(self, f: Linear) -> str:
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

    def in_ireg(self, f: Linear) -> str:
        """A form as a register even when it is a constant: the first operand of most PTX instructions."""
        v = self.value(f)
        if not f.is_const:
            return v
        r = self.reg("r")
        self.ins(f"mov.b32 {r}, {v}")
        return r

    # ---- predicates --------------------------------------------------------------------------------------------

    def cmp(self, op: str, a: Linear, b: Linear):
        """``setp`` of two forms; a constant on the left swaps sides so it lands where PTX takes an immediate."""
        if a.is_const and b.is_const:
            return _CONST_CMP[op](a.const, b.const)
        if a.is_const:
            a, b, op = b, a, _SWAP.get(op, op)
        p = self.reg("p")
        self.ins(f"setp.{op}.s32 {p}, {self.value(a)}, {self.value(b)}")
        return p

    def pred(self, e):
        """A predicate: ``True`` / ``False`` when it folds, else a ``.pred`` register."""
        k = type(e).__name__
        if k == "BinaryExpr" and e.op in CMP:
            return self.cmp(CMP[e.op], self.lin(e.left), self.lin(e.right))
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
