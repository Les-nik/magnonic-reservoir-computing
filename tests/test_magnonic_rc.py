import numpy as np
import pytest

from magnonic_rc import fields, logic, mobius, readout

T = logic.targets()


def test_truth_table():
    assert T["SUM"].tolist() == [0, 1, 1, 0, 1, 0, 0, 1]
    assert T["COUT"].tolist() == [0, 0, 0, 1, 0, 1, 1, 1]


def test_mobius_roundtrip():
    rng = np.random.default_rng(0)
    Z = {c: rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4)) for c in logic.CASES}
    z = mobius.decompose(Z)
    for c in logic.CASES:
        np.testing.assert_allclose(mobius.reconstruct(z, c), Z[c])


def test_sum_needs_mixed_terms():
    """XOR3 written in the Mobius basis: a + b + c - 2(ab + bc + ac) + 4abc."""
    Z = {c: float(T["SUM"][i]) for i, c in enumerate(logic.CASES)}
    z = mobius.decompose(Z)
    assert (z["a"], z["b"], z["c"]) == (1, 1, 1)
    assert (z["ab"], z["bc"], z["ac"], z["abc"]) == (-2, -2, -2, 4)


def test_affine_features_cannot_give_sum():
    """Features of a linear medium: the best linear readout of SUM is the constant 0.5."""
    rng = np.random.default_rng(1)
    coef = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    A = np.array([coef[0] + a * coef[1] + b * coef[2] + c * coef[3]
                  for a, b, c in map(logic.bits, logic.CASES)])
    _, pred = readout.fit_readout(A, T["SUM"])
    np.testing.assert_allclose(pred, 0.5, atol=1e-9)
    _, pred_c = readout.fit_readout(A, T["COUT"])
    assert readout.accuracy(pred_c, T["COUT"]) < 1  # majority is not affine either


def test_exhaustive_finds_planted_points():
    rng = np.random.default_rng(2)
    A = (rng.normal(size=(8, 12)) + 1j * rng.normal(size=(8, 12))) * 0.01
    A[:, 3] = T["SUM"] + 0.3j
    idx, mae, _ = readout.exhaustive_search(A, T["SUM"], 1)
    assert idx.tolist() == [3] and mae < 1e-12


def test_exhaustive_matches_loop():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 9)) + 1j * rng.normal(size=(8, 9))
    idx, mae, _ = readout.exhaustive_search(A, T["COUT"], 2, chunk=7)
    brute = min(np.mean(np.abs(readout.fit_readout(A[:, [i, j]], T["COUT"])[1] - T["COUT"]))
                for i in range(9) for j in range(i + 1, 9))
    assert mae == pytest.approx(brute)


def test_random_features_fit_perfectly_at_k3():
    """With 7 parameters for 8 cases and 85 320 combinations, noise alone reaches MAE ~ 0."""
    best = readout.null_distribution(n_points=81, k=3, t=T["COUT"], trials=2)
    assert np.all(best < 0.02)


def test_leave_one_out_shapes():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(8, 6)) + 1j * rng.normal(size=(8, 6))
    pred = readout.leave_one_out(A, T["COUT"], 1)
    assert pred.shape == (8,)


def test_itls_gradient_is_wirtinger():
    """The update direction must equal the numerical gradient of the loss for complex data."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    b = rng.uniform(0.5, 1.5, 8)
    W = rng.normal(size=4) + 1j * rng.normal(size=4)
    loss = lambda w: np.sum((np.abs(A @ w) ** 2 - b ** 2) ** 2)
    z = A @ W
    grad = 2 * (A.conj().T @ ((np.abs(z) ** 2 - b ** 2) * z))   # d loss / d W*
    h = 1e-6
    for j in range(4):
        e = np.zeros(4); e[j] = h
        d_re = (loss(W + e) - loss(W - e)) / (2 * h)
        d_im = (loss(W + 1j * e) - loss(W - 1j * e)) / (2 * h)
        # d/dW* = (d/dRe + i d/dIm) / 2
        assert grad[j] * 2 == pytest.approx(d_re + 1j * d_im, rel=1e-5)


def test_frequency_map_amplitude():
    dt, n, f0 = 1e-11, 400, 2.5e9
    t = np.arange(n) * dt
    sig = 0.7 * np.cos(2 * np.pi * f0 * t + 0.4)[:, None, None] * np.ones((1, 2, 3))
    m = fields.frequency_map(sig, dt, f0)
    np.testing.assert_allclose(np.abs(m), 0.7, rtol=1e-6)


def test_pool_blocks():
    maps = {c: np.arange(100 * 80).reshape(100, 80).astype(complex) for c in logic.CASES}
    feats, centres = fields.pool_blocks(maps, (10, 50, 0, 60), 20)
    assert feats.shape == (8, 6)
    assert centres[0].tolist() == [20, 10]


def test_phasor_fit_uneven_times():
    rng = np.random.default_rng(3)
    f, A, phi = 2.1e9, 0.3, 0.7
    t = np.sort(rng.uniform(0, 10 / f, 60))
    data = (0.1 + A * np.cos(2 * np.pi * f * t + phi))[:, None, None] * np.ones((1, 2, 3))
    Z = fields.phasor_fit(data, t, f)
    np.testing.assert_allclose(Z, 1j * A * np.exp(1j * phi), atol=1e-12)


def test_zarr_reader_and_phasor_map(tmp_path):
    import json
    from magnonic_rc.zarr_io import ZarrArray, phasor_map

    nt, ny, nx, f = 40, 4, 6, 2.1e9
    t = np.arange(nt) / (8 * f)
    ds = tmp_path / "000.zarr" / "m_z11"
    ds.mkdir(parents=True)
    meta = {"zarr_format": 2, "shape": [nt, 1, ny, nx, 3], "chunks": [1, 1, ny, nx, 3],
            "dtype": "<f4", "order": "C", "compressor": None, "fill_value": 0, "filters": None}
    (ds / ".zarray").write_text(json.dumps(meta))
    (ds / ".zattrs").write_text(json.dumps({"t": t.tolist()}))
    x = np.arange(nx)
    for i in range(nt):
        frame = np.zeros((1, 1, ny, nx, 3), dtype="<f4")
        frame[..., 2] = 0.01 * np.cos(2 * np.pi * f * t[i] - 0.5 * x)
        (ds / f"{i}.0.0.0.0").write_bytes(frame.tobytes())

    arr = ZarrArray(ds)
    assert len(arr) == nt and arr.frame(3).shape == (1, ny, nx, 3)
    Z = phasor_map(ds, f, n_last=32)
    np.testing.assert_allclose(np.abs(Z), 0.01, rtol=1e-4)
    np.testing.assert_allclose(np.angle(Z[0, 1] / Z[0, 0]), -0.5, atol=1e-4)


def test_no_resonator_script_changes_one_line():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("mnr", root / "scripts" / "make_no_resonators.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original = (root / "simulations" / "bowtie" / "000.mx3").read_text()
    changed = mod.convert(original)
    diff = [(a, b) for a, b in zip(original.splitlines(), changed.splitlines()) if a != b]
    assert len(diff) == 1
    assert diff[0][1].startswith("geometry := yig ")
