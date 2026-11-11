"""Reading amumax output (zarr v2) without the zarr package.

amumax saves every autosaved quantity as a zarr v2 array of shape
(t, z, y, x, component), one chunk per time step, with the snapshot times in
the ``.zattrs`` file. On the cluster the full arrays are tens of GB, so this
reader decodes one time step at a time.

Compressed chunks need ``numcodecs`` (installed together with zarr or pyzfn).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fields import phasor_fit


class ZarrArray:
    """One zarr v2 array whose chunks are whole time steps."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        meta = json.loads((self.path / ".zarray").read_text())
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.order = meta.get("order", "C")
        self.separator = meta.get("dimension_separator", ".")
        if self.chunks[1:] != self.shape[1:]:
            raise ValueError("expected one chunk per time step")
        self.codec = None
        if meta.get("compressor"):
            import numcodecs  # only needed for compressed data

            self.codec = numcodecs.get_codec(meta["compressor"])
        attrs = self.path / ".zattrs"
        self.attrs = json.loads(attrs.read_text()) if attrs.exists() else {}

    def __len__(self) -> int:
        return self.shape[0]

    def frame(self, i: int) -> np.ndarray | None:
        """Time step ``i`` as an array (z, y, x, component); None if the chunk is missing."""
        key = self.separator.join([str(i)] + ["0"] * (len(self.shape) - 1))
        path = self.path / key
        if not path.exists():
            return None
        raw = path.read_bytes()
        if self.codec is not None:
            raw = self.codec.decode(raw)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks, order=self.order)[0]

    def times(self) -> np.ndarray:
        for key in ("t", "time", "times"):
            if isinstance(self.attrs.get(key), list):
                return np.asarray(self.attrs[key], dtype=float)
        raise KeyError(f"no snapshot times in {self.path / '.zattrs'}")


def phasor_map(path: str | Path, frequency: float, n_last: int = 48, z_index: int = 0,
               component: int = 2, step: int = 1) -> np.ndarray:
    """Complex (y, x) phasor of one component at ``frequency``.

    ``path`` is the dataset inside a run, e.g. ``run.zarr/m_z11``. Uses the
    last ``n_last`` time steps; ``step`` keeps every step-th cell in x and y.
    """
    arr = ZarrArray(path)
    t = arr.times()
    idx = np.arange(len(arr))[-n_last:]
    frames, times = [], []
    for i in idx:
        frame = arr.frame(int(i))
        if frame is None:
            continue
        frames.append(frame[z_index, ::step, ::step, component].astype(np.float64))
        times.append(t[i])
    if len(frames) < 4:
        raise ValueError(f"{path}: only {len(frames)} time steps found")
    return phasor_fit(np.stack(frames), np.asarray(times), frequency)
