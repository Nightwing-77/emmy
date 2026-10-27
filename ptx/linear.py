"""The linear form every integer index is held in until an instruction needs it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Linear:
    """``sum(coeff * register) + const`` over s32 registers. A constant is the form with no terms.

    Keeping indices in this form is what lets constants fold, terms cancel, and the constant part of an address become
    the instruction's immediate offset (``[%rd1+512]``) instead of an ``add``."""

    terms: tuple[tuple[str, int], ...] = ()
    const: int = 0

    @staticmethod
    def of(reg: str) -> Linear:
        return Linear(((reg, 1),))

    @property
    def is_const(self) -> bool:
        return not self.terms

    def __add__(self, other: Linear) -> Linear:
        acc = dict(self.terms)
        for r, c in other.terms:
            acc[r] = acc.get(r, 0) + c
        return Linear(tuple(sorted((r, c) for r, c in acc.items() if c)), self.const + other.const)

    def scale(self, k: int) -> Linear:
        return Linear(tuple((r, c * k) for r, c in self.terms) if k else (), self.const * k)

    def without_const(self) -> Linear:
        return Linear(self.terms)
