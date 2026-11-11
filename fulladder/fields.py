"""From simulated m(t, y, x) to complex readout features.

Pipeline for each of the eight input cases:

1. ``frequency_map``: Fourier component of m_z at the drive frequency in every
   cell (amplitude and phase of the steady-state oscillation).
2. ``blur_complex``: Gaussian smoothing of that complex map. A detector has a
   finite size, so a single cell is not a realistic readout point.
3. ``pool_blocks``: sample the smoothed map at the centres of a grid of
   blocks inside a region of interest. Each block centre is one candidate
   readout point; the result is a complex matrix of shape (8 cases, P points).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def time_step(times: np.ndarray) -> float:
    """Sampling interval from a linear fit to the snapshot times."""
    slope, _ = np.polyfit(np.arange(len(times)), times, 1)
    return float(slope)


def frequency_map(data: np.ndarray, dt: float, frequency: float) -> np.ndarray:
    """Complex amplitude at the FFT bin closest to ``frequency``.

    ``data`` has time on axis 0. The amplitude is normalised so that a signal
    A cos(2 pi f t) gives |result| = A when f lies on an FFT bin.
    """
    n = data.shape[0]
    spectrum = np.fft.fft(data, axis=0) / (0.5 * n)
    freqs = np.fft.fftfreq(n, dt)
    i = int(np.argmin(np.abs(freqs - frequency)))
    return spectrum[i]


def phasor_fit(data: np.ndarray, times: np.ndarray, frequency: float) -> np.ndarray:
    """Least-squares fit of m(t) = c + a sin(2 pi f t) + b cos(2 pi f t) in every cell.

    Returns Z = a + i b. Unlike ``frequency_map`` it uses the actual snapshot
    times, so it does not need the drive frequency to fall on an FFT bin or
    the samples to be evenly spaced. For a signal A cos(2 pi f t + phi) the
    result is Z = i A exp(i phi): |Z| = A and the phase is phi + pi/2.
    """
    w = 2 * np.pi * frequency
    X = np.column_stack([np.ones_like(times), np.sin(w * times), np.cos(w * times)])
    coef, *_ = np.linalg.lstsq(X, data.reshape(len(times), -1), rcond=None)
    Z = coef[1] + 1j * coef[2]
    return Z.reshape(data.shape[1:])


def blur_complex(field: np.ndarray, fwhm_nm: float, cell_nm: float) -> np.ndarray:
    """Gaussian blur of a complex map (real and imaginary parts separately)."""
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2))) / cell_nm
    if np.iscomplexobj(field):
        return gaussian_filter(field.real, sigma) + 1j * gaussian_filter(field.imag, sigma)
    return gaussian_filter(field, sigma)


def pool_blocks(maps: dict, roi: tuple[int, int, int, int], block: int):
    """Sample every map at the block centres of a grid inside ``roi``.

    Parameters
    ----------
    maps : {case: complex (ny, nx) array}
    roi : (x_start, x_end, y_start, y_end) in cells
    block : block size in cells

    Returns
    -------
    features : complex array (n_cases, n_points)
    centres : int array (n_points, 2) of (x, y) cell coordinates
    """
    x0, x1, y0, y1 = roi
    nbx = (x1 - x0) // block
    nby = (y1 - y0) // block
    xs = x0 + block // 2 + block * np.arange(nbx)
    ys = y0 + block // 2 + block * np.arange(nby)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    centres = np.column_stack([xx.ravel(), yy.ravel()])
    cases = list(maps)
    features = np.array([maps[c][centres[:, 1], centres[:, 0]] for c in cases])
    return features, centres


def mask_absorbing_layer(field: np.ndarray, width_x: int, width_y: int = 0) -> np.ndarray:
    """Copy of ``field`` with the absorbing boundary layers set to zero."""
    out = field.copy()
    if width_x:
        out[..., :width_x] = 0
        out[..., -width_x:] = 0
    if width_y:
        out[..., :width_y, :] = 0
        out[..., -width_y:, :] = 0
    return out
