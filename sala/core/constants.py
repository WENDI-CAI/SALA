"""Shared numerical and chemistry constants."""

EPS = 1.0e-8

METAL_ATOMIC_NUMBERS: frozenset[int] = frozenset(
    {3, 11, 19, 37, 55, 87}
    | {4, 12, 20, 38, 56, 88}
    | set(range(21, 31))
    | set(range(39, 49))
    | set(range(72, 81))
    | set(range(57, 72))
    | set(range(89, 104))
    | {13, 31, 49, 50, 81, 82, 83}
)

__all__ = ["EPS", "METAL_ATOMIC_NUMBERS"]
