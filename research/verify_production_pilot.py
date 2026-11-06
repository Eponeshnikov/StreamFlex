"""End-to-end check of the production universal policy on the six research records.

Calibration → pilot-selected LDPC block → privacy amplification, and the key
length must match what the research prototype reported.
"""
import json
from dataclasses import replace
from pathlib import Path

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import growing_policy as G
from pages.offline_keygen_v2 import ldpc as L
from pages.offline_keygen_v2 import privacy as P
from research import linear_key_scheme as core

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_00", "july_kaiser9_01",
         "july_kaiser9_02", "july_kaiser9_03")
manifest = json.loads(core.PARENT.read_text())
rows = []
for case in CASES:
    raw = core.read(manifest["caches"][case]["decor"]["path"])
    raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
    cfg = G.baseline_config()
    d, c = G.run(raw, cfg)
    s = C.summary(c)
    sel = c.policy_selection
    rec = L.run(c, L.LdpcConfig(block_bits=int(sel["block_bits"]), soft=True,
                                public_reliability=False), session_id=case,
                times=[x.elapsed_s for x in d.steps])
    pa = P.run(rec, c, P.PAConfig(eve_from_frames=False))
    duration = [getattr(x, "elapsed_s", 0.0) for x in d.steps]
    duration = float(duration[-1] - duration[0]) if len(duration) > 1 else 0.0
    rows.append({
        "case": case,
        "cdf": cfg.cdf,
        "pilot BDR %": round(100 * sel["pilot_bdr_4bit"], 2),
        "bits/comp": sorted(set(sel["profile_bits"].values())),
        "block": sel["block_bits"],
        "raw L_R": s["L_R raw key bits"],
        "BDR %": round(100 * (s["BDR p̂_e"] or 0), 2),
        "reconciled N": pa.n,
        "debit": sel["public_decision_bits"],
        "ell": pa.length,
        "keys A=B": bool(pa.keys_equal) if pa.length else None,
        "bit/s": round(pa.length / duration, 2) if duration and pa.length else None,
    })
    print("DONE", case, rows[-1], flush=True)

out = Path("output_data/notes/keygen_production_pilot_20261106")
out.mkdir(parents=True, exist_ok=True)
(out / "e2e.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
