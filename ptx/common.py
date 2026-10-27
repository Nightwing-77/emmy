"""Small pieces every part of the emitter uses."""

import struct

INT32_MAX = 2**31 - 1
LOG2E = 1.4426950408889634
# Storage dtype → (PTX load/store type, bytes).
STORAGE = {"f32": ("f32", 4), "f16": ("b16", 2), "bf16": ("b16", 2)}


class Unsupported(Exception):
    """A Kernel IR construct (or a construct on a GPU) this emitter does not translate."""


def fimm(v: float) -> str:
    """A PTX f32 immediate: the bit pattern, so every value (including -inf) prints exactly."""
    return f"0f{struct.unpack('<I', struct.pack('<f', float(v)))[0]:08X}"
