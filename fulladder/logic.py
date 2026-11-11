"""Full-adder truth table.

The three input bits (A, B, Cin) switch the amplitude of three spin-wave
antennas between a low and a high level. The eight cases are named by their
bits, e.g. "101" means A = 1, B = 0, Cin = 1.
"""

from __future__ import annotations

import numpy as np

CASES = ("000", "001", "010", "011", "100", "101", "110", "111")


def bits(case: str) -> tuple[int, int, int]:
    a, b, c = (int(ch) for ch in case)
    return a, b, c


def sum_bit(a: int, b: int, c: int) -> int:
    """SUM = A xor B xor Cin."""
    return a ^ b ^ c


def carry_bit(a: int, b: int, c: int) -> int:
    """COUT = majority(A, B, Cin)."""
    return (a & b) | (c & (a ^ b))


def targets(cases=CASES) -> dict[str, np.ndarray]:
    """Truth-table columns for the given cases: {"SUM": ..., "COUT": ...}."""
    s = np.array([sum_bit(*bits(c)) for c in cases])
    k = np.array([carry_bit(*bits(c)) for c in cases])
    return {"SUM": s, "COUT": k}
