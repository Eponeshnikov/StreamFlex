"""Raw-key NIST check of the production universal policy on the research records."""
import json
import numpy as np

from pages.key_generation.compute.nist.runner import run_all
from pages.offline_keygen_v2 import growing_policy as G
from pages.offline_keygen_v2 import ldpc as L
from research import linear_key_scheme as core

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_02", "july_kaiser9_03")
manifest = json.loads(core.PARENT.read_text())


def raw_stream(res, party: str) -> np.ndarray:
    """The stream LDPC reads: every kept word in time order, MSB first."""
    words = [L.word_bits(res.bins[party][i: i + 1], int(res.widths[i]))[0]
             for i in np.flatnonzero(res.kept)]
    return np.concatenate(words) if words else np.zeros(0, np.uint8)


for case in CASES:
    raw = core.read(manifest["caches"][case]["decor"]["path"])
    raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
    cfg = G.baseline_config()
    _, c = G.run(raw, cfg)
    report = {}
    for party in ("alice", "bob"):
        seq = raw_stream(c, party)
        fails = [(t.name, round(t.p_value, 6)) for t in run_all(seq) if t.passed is False] \
            if seq.size >= 100 else ["too short"]
        report[party] = {"bits": int(seq.size), "ones": round(float(seq.mean()), 4),
                         "failed": fails}
    print(case, json.dumps(report), flush=True)
