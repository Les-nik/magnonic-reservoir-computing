"""Phasor maps of the eight full-adder runs, for use on the cluster.

Reads ``<case>.zarr/<dataset>`` for every run in a folder, fits the drive
frequency in every cell and writes one compressed .npz (tens of MB instead
of tens of GB). Only reads the simulation data.

    python scripts/phasor_maps.py RUN_FOLDER --frequency 2.1e9 --step 2 --last 48 -o phasors.npz

Output keys: ``<case>__Zz`` (complex64, y by x) for each run and ``meta``
(JSON: run name, number of samples, frequency, step). The same layout as the
earlier cluster script, so existing exports load with ``analyse_phasors.py``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fulladder.zarr_io import phasor_map


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    p.add_argument("--dataset", default="m_z11", help="dataset inside each .zarr (top YIG layer)")
    p.add_argument("--frequency", type=float, default=2.1e9)
    p.add_argument("--step", type=int, default=2, help="keep every step-th cell (2 = 10 nm)")
    p.add_argument("--last", type=int, default=48, help="number of final time steps to fit")
    p.add_argument("-o", "--output", type=Path, default=Path("phasors.npz"))
    args = p.parse_args()

    maps, meta = {}, []
    for run in sorted(args.folder.glob("*.zarr")):
        Z = phasor_map(run / args.dataset, args.frequency, n_last=args.last, step=args.step)
        maps[f"{run.stem}__Zz"] = Z.astype(np.complex64)
        meta.append({"run": run.stem, "f": args.frequency, "step": args.step, "n_last": args.last})
        print(f"{run.stem}: max |Z| = {np.abs(Z).max():.4g}")
    np.savez_compressed(args.output, meta=json.dumps(meta), **maps)
    print(f"saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
