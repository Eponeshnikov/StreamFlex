from pages.offline_keygen_v2 import decorrelation as D
from research.acf_causal_selection import select
from research.test_acf_growing_linear9 import make_raw


def test_pilot_selection_is_prefix_invariant():
    raw = make_raw()
    full = select(raw)
    small = D.DecorRun(
        session_id="test-pilot-prefix",
        config=raw.config,
        dt_s=raw.dt_s,
        speed_mps=raw.speed_mps,
        x={p: a[:1200] for p, a in raw.x.items()},
        dtau={},
        steps=raw.steps[:1200],
        status=raw.status[:1200],
        votes={},
    )
    prefix = select(small)
    for key in (
        "profile_bits", "block_bits", "pilot_bdr_4bit",
        "pilot_kept_values", "pilot_profile_counts",
    ):
        assert full[key] == prefix[key]
    assert full["profile_bits"] == {0: 3, 1: 3}
    assert full["block_bits"] == 768


def test_long_low_bdr_record_selects_four_bits_and_large_blocks():
    selection = select(make_raw(n=6000))
    assert selection["profile_bits"] == {0: 4, 1: 4}
    assert selection["block_bits"] == 4096
