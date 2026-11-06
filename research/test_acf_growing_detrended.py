from types import SimpleNamespace

import numpy as np

from pages.offline_keygen_v2 import decorrelation as D
from research.acf_growing_detrended import MEAN_WINDOW, calibrate, residual


def test_residual_is_exact_and_strictly_causal():
    rng = np.random.default_rng(81)
    a = rng.normal(size=(100, 2))
    a[::7, 0] = np.nan
    full = residual({"alice": a})["alice"]
    prefix = residual({"alice": a[:60]})["alice"]
    np.testing.assert_array_equal(full[:60], prefix)
    for j in range(a.shape[1]):
        ix = np.flatnonzero(np.isfinite(a[:, j]))
        v = a[ix, j]
        expected = v[MEAN_WINDOW:] - np.lib.stride_tricks.sliding_window_view(
            v[:-1], MEAN_WINDOW
        ).mean(axis=1)
        np.testing.assert_allclose(full[ix[MEAN_WINDOW:], j], expected)


def test_quantizer_preserves_acf_status_and_uses_growing_history():
    rng = np.random.default_rng(82)
    n = 1400
    x = {"alice": rng.normal(size=(n, 2))}
    x["bob"] = x["alice"] + rng.normal(scale=0.02, size=(n, 2))
    status = np.full((n, 2), D.ADMITTED, np.uint8)
    raw = D.DecorRun(
        session_id="test",
        config=D.DecorConfig(),
        dt_s=1.0,
        speed_mps=0.0,
        x=x,
        dtau={},
        steps=[SimpleNamespace(elapsed_s=float(i), time_idx=i) for i in range(n)],
        status=status,
        votes={},
    )
    full = calibrate(raw)
    np.testing.assert_array_equal(raw.status, status)
    assert full.config.update == "growing"
    assert full.config.n_target == 1000
    assert full.config.bits == 1
    assert full.config.check_bits == 1
    assert not full.config.rebalance

    small = D.DecorRun(
        session_id="test-prefix",
        config=raw.config,
        dt_s=1.0,
        speed_mps=0.0,
        x={p: a[:1200] for p, a in x.items()},
        dtau={},
        steps=raw.steps[:1200],
        status=status[:1200],
        votes={},
    )
    prefix = calibrate(small)
    before = full.step < 1200
    np.testing.assert_array_equal(full.step[before], prefix.step)
    np.testing.assert_array_equal(full.profile[before], prefix.profile)
    np.testing.assert_array_equal(full.erased[before], prefix.erased)
    for p in ("alice", "bob"):
        np.testing.assert_array_equal(full.bins[p][before], prefix.bins[p])
