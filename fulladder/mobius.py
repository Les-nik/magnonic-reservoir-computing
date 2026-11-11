"""Where does the nonlinearity come from? Mobius decomposition of the 8 cases.

Any quantity Z measured for the eight input combinations can be written as

    Z(a, b, c) = z0 + za a + zb b + zc c + zab ab + zbc bc + zac ac + zabc abc

with a, b, c in {0, 1}. This is exact (8 numbers, 8 coefficients) and the
coefficients follow from inclusion-exclusion, e.g. zab = Z110 - Z100 - Z010 + Z000.

A linear medium adds the waves of the three antennas, so it only produces
z0, za, zb, zc. Even a nonlinear response to one antenna alone stays inside
these four terms, because any function of a single bit is affine in that bit.
The mixed terms (ab, bc, ac, abc) need waves from different inputs to
interact. SUM = a xor b xor c = a + b + c - 2(ab + bc + ac) + 4abc needs them,
so a readout that is linear in features without mixed terms cannot produce SUM.
"""

from __future__ import annotations

import numpy as np

from .logic import CASES

TERMS = ("0", "a", "b", "c", "ab", "bc", "ac", "abc")
MIXED = ("ab", "bc", "ac", "abc")


def decompose(Z: dict) -> dict:
    """Mobius coefficients from {case: value or array}. Works element-wise on arrays."""
    z = {
        "0": Z["000"],
        "a": Z["100"] - Z["000"],
        "b": Z["010"] - Z["000"],
        "c": Z["001"] - Z["000"],
        "ab": Z["110"] - Z["100"] - Z["010"] + Z["000"],
        "bc": Z["011"] - Z["010"] - Z["001"] + Z["000"],
        "ac": Z["101"] - Z["100"] - Z["001"] + Z["000"],
    }
    z["abc"] = (Z["111"] - Z["110"] - Z["101"] - Z["011"]
                + Z["100"] + Z["010"] + Z["001"] - Z["000"])
    return z


def reconstruct(z: dict, case: str):
    """Value for one case from the coefficients (inverse of ``decompose``)."""
    a, b, c = (int(ch) for ch in case)
    weight = {"0": 1, "a": a, "b": b, "c": c, "ab": a * b, "bc": b * c, "ac": a * c, "abc": a * b * c}
    return sum(weight[t] * z[t] for t in TERMS)


def mixed_fraction(Z: dict) -> np.ndarray:
    """Share of the input-dependent power carried by mixed terms, P_mixed / (P_single + P_mixed)."""
    z = decompose(Z)
    p_single = sum(np.abs(z[t]) ** 2 for t in ("a", "b", "c"))
    p_mixed = sum(np.abs(z[t]) ** 2 for t in MIXED)
    return p_mixed / (p_single + p_mixed + 1e-300)


def affine_part(Z: dict) -> dict:
    """What a linear medium would give: the same z0, za, zb, zc and no mixed terms."""
    z = decompose(Z)
    out = {}
    for case in CASES:
        a, b, c = (int(ch) for ch in case)
        out[case] = z["0"] + a * z["a"] + b * z["b"] + c * z["c"]
    return out
