import numpy as np

from pages.offline_keygen_v2 import joint_leakage as J
from research.acf_profile_rebalance import calibrate
from research.test_acf_growing_linear9 import make_raw


def test_profile_rebalance_replays_and_preserves_standard_acf_status():
    raw = make_raw()
    original_status = raw.status.copy()
    c = calibrate(raw, 4, profile_bits={0: 2, 1: 4})
    np.testing.assert_array_equal(raw.status, original_status)
    assert c.config.update == "growing"
    assert c.config.cdf == "per_profile"
    assert set(np.unique(c.widths)) == {2, 4}
    J.check_transcript(c)
    for p in ("alice", "bob"):
        assert c.bits[p] == [
            format(int(word), f"0{int(width)}b")
            for word, width in zip(
                c.bins[p][c.kept], c.widths[c.kept], strict=True
            )
        ]
