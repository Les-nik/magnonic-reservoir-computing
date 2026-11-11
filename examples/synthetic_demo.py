"""What a perfect in-sample score does and does not tell you: a synthetic test.

Three toy "devices", each giving complex readout features for the 8 input cases:

* linear:     waves from the three antennas only add up (affine in the bits),
              plus a tiny noise floor (1e-3 of the signal).
* nonlinear:  the same, plus mixed terms (ab, bc, ac, abc) of 20 % strength.
* random:     independent random numbers for every case and point.

For each device and k = 1, 2, 3 readout points the script reports the
in-sample MAE of the best exhaustive search, and the leave-one-out MAE, where
points and weights are chosen on 7 cases and the 8th is predicted.

    python examples/synthetic_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnonic_rc import logic, mobius, readout

ROOT = Path(__file__).resolve().parents[1]
N_POINTS = 40
T = logic.targets()


def device(kind: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    if kind == "random":
        return cplx(8, N_POINTS)
    z = {name: cplx(N_POINTS) for name in mobius.TERMS}
    strength = 0.2 if kind == "nonlinear" else 0.0
    for name in mobius.MIXED:
        z[name] *= strength
    A = np.array([mobius.reconstruct(z, c) for c in logic.CASES])
    return A + 1e-3 * cplx(8, N_POINTS)


def main():
    kinds = ["linear", "nonlinear", "random"]
    ks = [1, 2, 3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, target in zip(axes, ["SUM", "COUT"]):
        t = T[target]
        for kind, color in zip(kinds, ["tab:blue", "tab:green", "tab:red"]):
            A = device(kind)
            ins, loo = [], []
            for k in ks:
                ins.append(readout.exhaustive_search(A, t, k)[1])
                pred = readout.leave_one_out(A, t, k)
                loo.append(np.mean(np.abs(pred - t)))
                print(f"{target:4s} {kind:9s} k={k}: in-sample MAE {ins[-1]:.3f}, leave-one-out MAE {loo[-1]:.3f}")
            ax.plot(ks, ins, "o-", color=color, label=f"{kind}, in-sample")
            ax.plot(ks, loo, "s--", color=color, mfc="none", label=f"{kind}, leave-one-out")
        ax.set_title(target)
        ax.set_xlabel("k (readout points)")
        ax.set_xticks(ks)
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    axes[0].set_ylabel("MAE")
    axes[1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    out = ROOT / "figures" / "synthetic_check.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
