"""The GPU a kernel is compiled for.

A GPU generation is named by its compute capability, ``sm_XY``. Three things change with it:

- **The PTX version.** Every ``.target sm_XY`` needs a minimum ``.version``; ptxas refuses an older one.
- **The instructions.** A few instructions we print appear only on newer GPUs. On an older one the emitter prints a
  replacement sequence that computes the same result (see :class:`Target`'s properties).
- **Shared memory.** How much one CTA may use. A kernel that needs more cannot launch there, so it is refused.
"""

from dataclasses import dataclass

from .common import Unsupported

# sm → (PTX ISA version as 10·major + minor, most shared memory one CTA may use in KB). The version is the oldest one
# ptxas accepts for the target and that has every instruction we print there: never below 6.2, the first with
# ``activemask``, which the warp shuffles use.
_GPUS = {
    50: (62, 48),  # Maxwell
    52: (62, 48),
    53: (62, 48),
    60: (62, 48),  # Pascal
    61: (62, 48),
    62: (62, 48),
    70: (62, 96),  # Volta
    72: (62, 96),
    75: (70, 64),  # Turing: 7.0 is the first with tanh.approx
    80: (70, 163),  # Ampere
    86: (71, 99),
    87: (74, 163),
    89: (78, 99),  # Ada Lovelace
    90: (78, 227),  # Hopper
    100: (86, 227),  # Blackwell, data center
    120: (87, 99),  # Blackwell, consumer
}


@dataclass(frozen=True)
class Target:
    sm: int

    @staticmethod
    def parse(name: str) -> "Target":
        """``sm_89``, ``89`` or ``8.9``."""
        s = name.lower().removeprefix("sm_").replace(".", "")
        if not s.isdigit() or int(s) not in _GPUS:
            raise Unsupported(f"GPU {name}: known are {', '.join(f'sm_{k}' for k in _GPUS)}")
        return Target(int(s))

    @property
    def name(self) -> str:
        return f"sm_{self.sm}"

    @property
    def ptx_version(self) -> str:
        v = _GPUS[self.sm][0]
        return f"{v // 10}.{v % 10}"

    @property
    def smem_bytes(self) -> int:
        return _GPUS[self.sm][1] * 1024

    # ---- instructions that only some GPUs have ---------------------------------------------------------------------

    @property
    def has_cp_async(self) -> bool:
        """``cp.async`` (global → shared without registers), sm_80+. Before it: a load and a store."""
        return self.sm >= 80

    @property
    def has_bf16_cvt(self) -> bool:
        """``cvt.rn.bf16.f32``, sm_80+. Before it: round-to-nearest-even on the integer bits."""
        return self.sm >= 80

    @property
    def has_tanh(self) -> bool:
        """``tanh.approx.f32``, sm_75+. Before it: tanh from one ``ex2.approx``."""
        return self.sm >= 75

    @property
    def has_grid_dependency(self) -> bool:
        """``griddepcontrol``, sm_90+: lets this launch start while the grid ahead of it drains."""
        return self.sm >= 90


TARGETS = tuple(Target(sm) for sm in _GPUS)
