import numpy as np

from pages.offline_keygen_v2 import decorrelation as D
from research.acf_maq import calibrate, maq_word
from research.test_acf_growing_linear9 import make_raw


def test_maq_one_bit_codebooks_match_patwari_table():
    # One-based Table 1: k=1,4,5,8 use d0; k=2,3,6,7 use d1.
    assert [maq_word(k, 1, int((k + 1) % 4 >= 2)) for k in range(8)] == [
        0, 0, 0, 1, 1, 1, 1, 0
    ]


def test_maq_is_causal_and_drops_no_candidates():
    raw = make_raw()
    original_status = raw.status.copy()
    profile_bits = {0: 2, 1: 4}
    full, transcript = calibrate(raw, profile_bits)
    np.testing.assert_array_equal(raw.status, original_status)
    assert full.kept.all()
    assert transcript.kept.all()
    assert np.array_equal(transcript.alice, transcript.bob)

    small = D.DecorRun(
        session_id="test-maq-prefix",
        config=raw.config,
        dt_s=1.0,
        speed_mps=0.0,
        x={p: a[:1200] for p, a in raw.x.items()},
        dtau={},
        steps=raw.steps[:1200],
        status=raw.status[:1200],
        votes={},
    )
    prefix, _ = calibrate(small, profile_bits)
    before = full.step < 1200
    np.testing.assert_array_equal(full.step[before], prefix.step)
    np.testing.assert_array_equal(full.profile[before], prefix.profile)
    np.testing.assert_array_equal(full.erased[before], prefix.erased)
    for p in ("alice", "bob"):
        np.testing.assert_array_equal(full.bins[p][before], prefix.bins[p])
