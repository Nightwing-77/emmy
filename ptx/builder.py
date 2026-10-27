"""The output buffer: fresh registers, labels, instruction lines, and scopes of already-computed values."""

from dataclasses import dataclass, field


@dataclass
class Scope:
    """Values computed in one block of control flow; a block's values do not outlive it."""

    forms: dict = field(default_factory=dict)  # Linear / (op, key) → register


class Builder:
    def __init__(self):
        self.lines: list[str] = []
        self.counts = {"p": 0, "r": 0, "rd": 0, "f": 0, "h": 0}  # registers handed out, per kind
        self.labels = 0
        self.scopes = [Scope()]

    def reg(self, kind: str) -> str:
        """A fresh register: ``p`` predicate, ``r`` 32-bit int, ``rd`` 64-bit int, ``f`` f32, ``h`` 16-bit."""
        self.counts[kind] += 1
        return f"%{kind}{self.counts[kind]}"

    def label(self) -> str:
        self.labels += 1
        return f"$L{self.labels}"

    def ins(self, text: str) -> None:
        self.lines.append(f"\t{text};")

    def mark(self, label: str) -> None:
        self.lines.append(f"{label}:")

    # ---- scopes: a value computed once is reused until its block ends ----------------------------------------------

    def block(self, fn) -> None:
        self.scopes.append(Scope())
        fn()
        self.scopes.pop()

    def cached(self, key):
        for s in reversed(self.scopes):
            if key in s.forms:
                return s.forms[key]
        return None

    def remember(self, key, reg: str) -> str:
        self.scopes[-1].forms[key] = reg
        return reg
