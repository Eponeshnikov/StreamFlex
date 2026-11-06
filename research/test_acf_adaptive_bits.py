import numpy as np

from pages.offline_keygen_v2 import decorrelation as D
from pages.offline_keygen_v2 import joint_leakage as J
from research.acf_adaptive_bits import adaptive_calibrate
from research.test_acf_growing_linear9 import make_raw


def test_public_check_adaptation_is_causal_and_replays_exactly():
    raw = make_raw()
    original_status = raw.status.copy()
    full = adaptive_calibrate(raw)
    np.testing.assert_array_equal(raw.status, original_status)
    assert set(np.unique(full.widths)) <= {2, 3, 4}
    assert full.config.update == "growing"
    assert full.config.check_bits == 2
    J.check_transcript(full)

    small = D.DecorRun(
        session_id="test-adaptive-prefix",
        config=raw.config,
        dt_s=1.0,
        speed_mps=0.0,
        x={p: a[:1200] for p, a in raw.x.items()},
        dtau={},
        steps=raw.steps[:1200],
        status=raw.status[:1200],
        votes={},
    )
    prefix = adaptive_calibrate(small)
    before = full.step < 1200
    np.testing.assert_array_equal(full.step[before], prefix.step)
    np.testing.assert_array_equal(full.profile[before], prefix.profile)
    np.testing.assert_array_equal(full.widths[before], prefix.widths)
    np.testing.assert_array_equal(full.erased[before], prefix.erased)
    for p in ("alice", "bob"):
        np.testing.assert_array_equal(full.bins[p][before], prefix.bins[p])
        assert full.bits[p] == [
            format(int(word), f"0{int(width)}b")
            for word, width in zip(
                full.bins[p][full.kept], full.widths[full.kept], strict=True
            )
        ]
