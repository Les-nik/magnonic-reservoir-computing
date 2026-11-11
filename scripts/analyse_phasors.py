"""Readout analysis of the eight phasor maps from ``phasor_maps.py``.

    python scripts/analyse_phasors.py phasors.npz --points "1210,480;1210,520;1250,640"
    python scripts/analyse_phasors.py phasors.npz --search 3 --roi 1000,1400,300,900 --block 20

Points and the region of interest are in simulation cells (5 nm). The script
prints, for SUM and COUT:

* the in-sample fit at the given (or searched) readout points,
* leave-one-out predictions, where points and weights come from 7 cases,
* the best in-sample MAE reached by random features with the same search,
* the share of mixed (nonlinear) Mobius terms at the readout points.

and saves a map of the mixed-term share next to the input file.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fulladder import fields, logic, mobius, readout


def load(path: Path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    step = meta[0].get("step", meta[0].get("krok", 1))
    maps = {}
    for m in meta:
        name = m.get("run", m.get("przebieg"))
        case = [c for c in logic.CASES if c in name]
        if case:
            maps[case[-1]] = data[f"{name}__Zz"].astype(np.complex128)
    missing = [c for c in logic.CASES if c not in maps]
    if missing:
        raise SystemExit(f"missing cases {missing} in {path}")
    return {c: maps[c] for c in logic.CASES}, step


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phasors", type=Path)
    p.add_argument("--points", help='readout points "x,y;x,y;..." in cells')
    p.add_argument("--search", type=int, help="search k points on a block grid instead")
    p.add_argument("--roi", default=None, help="x0,x1,y0,y1 in cells for --search")
    p.add_argument("--block", type=int, default=20, help="block size in cells for --search")
    p.add_argument("--fwhm", type=float, default=200, help="detector size (Gaussian FWHM) in nm")
    p.add_argument("--null-trials", type=int, default=10)
    args = p.parse_args()

    maps, step = load(args.phasors)
    cell_nm = 5 * step
    blurred = {c: fields.blur_complex(m, args.fwhm, cell_nm) for c, m in maps.items()}

    if args.points:
        pts = np.array([[int(v) // step for v in s.split(",")] for s in args.points.split(";")])
        A = np.array([[blurred[c][y, x] for x, y in pts] for c in logic.CASES])
        k, allowed = len(pts), None
    elif args.search:
        x0, x1, y0, y1 = (int(v) // step for v in args.roi.split(","))
        A, centres = fields.pool_blocks(blurred, (x0, x1, y0, y1), max(1, args.block // step))
        k, allowed = args.search, None
    else:
        raise SystemExit("give --points or --search")

    targets = logic.targets()
    for name, t in targets.items():
        if args.points:
            idx = np.arange(k)
            _, pred = readout.fit_readout(A, t)
        else:
            idx, _, pred = readout.exhaustive_search(A, t, k, allowed)
        loo = readout.leave_one_out(A[:, idx] if args.points else A, t, k,
                                    search=readout.exhaustive_search)
        null = readout.null_distribution(A.shape[1], k, t, trials=args.null_trials)
        print(f"\n{name}: k = {k}, {2 * k + 1} parameters for 8 cases")
        print(f"  in-sample MAE {np.mean(np.abs(pred - t)):.3f}, accuracy {readout.accuracy(pred, t):.2f}")
        print(f"  leave-one-out MAE {np.mean(np.abs(loo - t)):.3f}, accuracy {readout.accuracy(loo, t):.2f}")
        print(f"  random features, same search: best MAE {null.min():.3f} to {null.max():.3f}")
        if not args.points:
            print(f"  chosen points (cells): {[tuple(c * step) for c in centres[idx]]}")

    share = mobius.mixed_fraction(blurred)
    if args.points:
        sel = {c: A[i] for i, c in enumerate(logic.CASES)}
        print(f"\nmixed-term share at the readout points: {np.round(mobius.mixed_fraction(sel), 3)}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(share, origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[0, share.shape[1] * cell_nm, 0, share.shape[0] * cell_nm])
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("share of mixed Mobius terms")
    fig.colorbar(im, ax=ax)
    out = args.phasors.with_name(args.phasors.stem + "_mixed_share.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
