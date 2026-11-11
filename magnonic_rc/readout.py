"""Linear readout of the full adder and checks of how much it can be trusted.

Readout model
-------------
For k readout points with complex values a_1..a_k (one row per input case),
the prediction is a linear function of their real and imaginary parts:

    y = w0 + sum_j (u_j Re a_j + v_j Im a_j)

fitted by least squares to the 0/1 truth table. That is 2k + 1 parameters
for 8 cases: 7 parameters for k = 3. A prediction counts as 1 above 0.8, as 0
below 0.2, and as wrong in between.

Why the checks matter
---------------------
With 2k + 1 close to 8, a perfect in-sample fit is easy, and searching over
all C(P, k) choices of points makes it easier still. Three checks in this
module tell a real effect from fitting freedom:

* ``null_distribution``: the same search on random features. If random
  numbers also reach MAE ~ 0, a perfect score on the simulation means little.
* ``leave_one_out``: choose points and weights on 7 cases, predict the 8th.
* ``noise_accuracy``: accuracy when the trained weights see noisy features.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np


# ---------------------------------------------------------------- basic fit

def design_matrix(A: np.ndarray) -> np.ndarray:
    """[Re A, Im A, 1] for a complex (n_cases, k) array."""
    return np.hstack([A.real, A.imag, np.ones((A.shape[0], 1))])


def fit_readout(A: np.ndarray, t: np.ndarray):
    """Least-squares weights and in-sample predictions."""
    X = design_matrix(A)
    w, *_ = np.linalg.lstsq(X, t, rcond=None)
    return w, X @ w


def classify(pred: np.ndarray, low: float = 0.2, high: float = 0.8) -> np.ndarray:
    """1 above ``high``, 0 below ``low``, -1 (undecided) in between."""
    out = np.full(pred.shape, -1, dtype=int)
    out[pred >= high] = 1
    out[pred <= low] = 0
    return out


def accuracy(pred: np.ndarray, t: np.ndarray, low: float = 0.2, high: float = 0.8) -> float:
    return float(np.mean(classify(pred, low, high) == t))


# ---------------------------------------------------------- point selection

def exhaustive_search(A: np.ndarray, t: np.ndarray, k: int, allowed=None, chunk: int = 50_000):
    """Best set of k points (lowest in-sample MAE) among all combinations.

    Vectorised: for a chunk of combinations at once, the least-squares
    predictions are X pinv(X) t with a batched pseudo-inverse.

    Returns (indices, mae, predictions).
    """
    allowed = np.arange(A.shape[1]) if allowed is None else np.asarray(allowed)
    best = (None, np.inf, None)
    combos = combinations(allowed, k)
    n_total = comb(len(allowed), k)
    done = 0
    while done < n_total:
        batch = np.array([c for _, c in zip(range(chunk), combos)])
        done += len(batch)
        sel = A[:, batch]                              # (n_cases, n_batch, k)
        X = np.concatenate([sel.real, sel.imag, np.ones(sel.shape[:2] + (1,))], axis=2)
        X = np.moveaxis(X, 0, 1)                       # (n_batch, n_cases, 2k+1)
        w = np.linalg.pinv(X) @ t                      # (n_batch, 2k+1)
        pred = np.einsum("bnk,bk->bn", X, w)           # (n_batch, n_cases)
        mae = np.mean(np.abs(pred - t), axis=1)
        i = int(np.argmin(mae))
        if mae[i] < best[1]:
            best = (batch[i], float(mae[i]), pred[i])
    return best


def greedy_search(A: np.ndarray, t: np.ndarray, k: int, allowed=None):
    """Forward selection: add, one at a time, the point that lowers the MAE most.

    Much cheaper than ``exhaustive_search`` for large k; not guaranteed optimal.
    """
    remaining = list(np.arange(A.shape[1]) if allowed is None else allowed)
    chosen: list[int] = []
    mae = np.inf
    pred = None
    for _ in range(k):
        scores = []
        for j in remaining:
            _, p = fit_readout(A[:, chosen + [j]], t)
            scores.append(np.mean(np.abs(p - t)))
        best = int(np.argmin(scores))
        chosen.append(remaining.pop(best))
        mae = float(scores[best])
        _, pred = fit_readout(A[:, chosen], t)
    return np.array(chosen), mae, pred


# ------------------------------------------------------------------ checks

def null_distribution(n_points: int, k: int, t: np.ndarray, trials: int = 20, seed: int = 0):
    """Best in-sample MAE of the exhaustive search on random complex features.

    Features are independent standard complex Gaussians, shape (len(t), n_points).
    Returns an array of ``trials`` best MAE values.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        R = (rng.standard_normal((len(t), n_points)) + 1j * rng.standard_normal((len(t), n_points))) / np.sqrt(2)
        out.append(exhaustive_search(R, t, k)[1])
    return np.array(out)


def leave_one_out(A: np.ndarray, t: np.ndarray, k: int, allowed=None, search=exhaustive_search):
    """Predict each case from a readout chosen and fitted on the other cases.

    Point selection is repeated inside every fold, otherwise the held-out case
    would still influence the choice of points.
    Returns the out-of-sample predictions (one per case).
    """
    n = len(t)
    pred = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        idx, _, _ = search(A[keep], t[keep], k, allowed)
        w, _ = fit_readout(A[keep][:, idx], t[keep])
        pred[i] = (design_matrix(A[i:i + 1, idx]) @ w)[0]
    return pred


def noise_accuracy(A_sel: np.ndarray, t: np.ndarray, levels=(0.01, 0.02, 0.05, 0.1, 0.2),
                   draws: int = 2000, seed: int = 0) -> dict:
    """Mean accuracy when complex Gaussian noise (relative to the mean |A|) is added.

    The weights are fitted once on clean data, as a fixed readout would be.
    """
    rng = np.random.default_rng(seed)
    w, _ = fit_readout(A_sel, t)
    scale = np.abs(A_sel).mean()
    out = {}
    for s in levels:
        acc = []
        for _ in range(draws):
            noise = s * scale * (rng.standard_normal(A_sel.shape) + 1j * rng.standard_normal(A_sel.shape)) / np.sqrt(2)
            acc.append(accuracy(design_matrix(A_sel + noise) @ w, t))
        out[s] = float(np.mean(acc))
    return out


# --------------------------------------------------- intensity readout (iTLS)

def itls_intensity(A: np.ndarray, b: np.ndarray, k: int, steps: int = 2000, mu: float = 0.05,
                   seed: int = 0):
    """Sparse complex weights W with |A W| close to b (an intensity-detector readout).

    Minimises sum_m (|z_m|^2 - b_m^2)^2 with z = A W by gradient descent,
    keeping the k largest weights after every step (iterative hard
    thresholding). For complex W the descent direction is the Wirtinger
    derivative d/dW* = 2 A^H (e * z), with e = |z|^2 - b^2. Using conj(z)
    instead of z here is a common slip; it equals the right gradient only
    when A and W are real.
    """
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1
    An = A / norms
    bs = max(np.linalg.norm(b), 1e-30)
    bn = b / bs
    W = (rng.standard_normal(A.shape[1]) + 1j * rng.standard_normal(A.shape[1])) * 1e-2
    for _ in range(steps):
        z = An @ W
        e = np.abs(z) ** 2 - bn ** 2
        W = W - mu * 2 * (An.conj().T @ (e * z))
        if k < W.size:
            W[np.argsort(np.abs(W))[:-k]] = 0
    z = An @ W
    loss = float(np.sum((np.abs(z) ** 2 - bn ** 2) ** 2))
    return W / norms * bs, np.flatnonzero(W), loss
