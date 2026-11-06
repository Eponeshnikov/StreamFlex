from types import SimpleNamespace

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import decorrelation as D
from research.acf_growing_full_budget import calibrate as budget_calibrate
from research.acf_growing_linear9 import (
    KEY_BITS,
    TREND_WINDOW,
    binary_words,
    calibrate,
    inverse_gray,
    plane_separated,
    residual,
    trend_weights,
)


def make_raw(n=1400):
    rng = np.random.default_rng(82)
    x = {"alice": rng.normal(size=(n, 2))}
    x["bob"] = x["alice"] + rng.normal(scale=0.02, size=(n, 2))
    status = np.full((n, 2), D.ADMITTED, np.uint8)
    return D.DecorRun(
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


def test_linear_residual_is_exact_causal_and_has_constant_line_response():
    rng = np.random.default_rng(81)
    a = rng.normal(size=(100, 2))
    a[::7, 0] = np.nan
    full = residual({"alice": a})["alice"]
    prefix = residual({"alice": a[:60]})["alice"]
    np.testing.assert_array_equal(full[:60], prefix)
    for j in range(a.shape[1]):
        ix = np.flatnonzero(np.isfinite(a[:, j]))
        v = a[ix, j]
        expected = v[TREND_WINDOW:] - np.lib.stride_tricks.sliding_window_view(
            v[:-1], TREND_WINDOW
        ) @ trend_weights()
        np.testing.assert_allclose(full[ix[TREND_WINDOW:], j], expected)

    line = np.arange(100, dtype=float).reshape(-1, 1)
    line = 3.0 + 2.0 * line
    z = residual({"alice": line})["alice"]
    np.testing.assert_allclose(z[TREND_WINDOW:], 2.0, atol=1e-12)


def test_quantizer_preserves_acf_status_and_binary_prefixes():
    raw = make_raw()
    original_status = raw.status.copy()
    full = calibrate(raw)
    np.testing.assert_array_equal(raw.status, original_status)
    assert full.config.update == "growing"
    assert full.config.n_target == 1000
    assert full.config.bits == KEY_BITS
    assert full.config.check_bits == 2
    assert full.config.rebalance

    small = D.DecorRun(
        session_id="test-prefix",
        config=raw.config,
        dt_s=1.0,
        speed_mps=0.0,
        x={p: a[:1200] for p, a in raw.x.items()},
        dtau={},
        steps=raw.steps[:1200],
        status=raw.status[:1200],
        votes={},
    )
    prefix = calibrate(small)
    before = full.step < 1200
    np.testing.assert_array_equal(full.step[before], prefix.step)
    np.testing.assert_array_equal(full.profile[before], prefix.profile)
    np.testing.assert_array_equal(full.erased[before], prefix.erased)
    for p in ("alice", "bob"):
        np.testing.assert_array_equal(full.bins[p][before], prefix.bins[p])
        assert full.bits[p] == [
            format(int(word), f"0{KEY_BITS}b") for word in full.bins[p][full.kept]
        ]


def test_plane_order_and_ldpc_inverse_gray_are_bijections():
    raw = make_raw()
    c = calibrate(raw)
    data = plane_separated(c)
    assert len(data) == sum(c.widths[c.kept])
    assert np.all(np.diff(data[:, 3]) >= 0)
    for p, col in (("alice", 4), ("bob", 5)):
        direct = binary_words(c)
        expected = np.frombuffer("".join(direct.bits[p]).encode(), np.uint8) - 48
        original = from_original_order(data, col)
        np.testing.assert_array_equal(original, expected)
        np.testing.assert_array_equal(C.gray(inverse_gray(c.bins[p])), c.bins[p])


def from_original_order(data, col):
    order = np.lexsort((data[:, 3], data[:, 1], data[:, 0]))
    return data[order, col]


def test_zero_check_bits_keeps_supported_words_and_preserves_acf_status():
    raw = make_raw()
    original_status = raw.status.copy()
    c = budget_calibrate(raw, bits=3, check_bits=0)
    np.testing.assert_array_equal(raw.status, original_status)
    assert c.config.bits == 3
    assert c.config.check_bits == 0
    assert c.kept.all()
    for p in ("alice", "bob"):
        assert c.bits[p] == [format(int(word), "03b") for word in c.bins[p]]
