"""End-to-end numbers of the shipped universal policy on the six research records.

Uses the production functions only: `growing_policy.run` -> `ldpc.run` with the
block the policy selected -> `privacy.run`, plus raw-key NIST on the stream
LDPC reads and on the final key.
"""
import json
from pathlib import Path

import numpy as np

from pages.key_generation.compute.nist.runner import run_all
from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import growing_policy as G
from pages.offline_keygen_v2 import ldpc as L
from pages.offline_keygen_v2 import privacy as P
from research import linear_key_scheme as core

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_00", "july_kaiser9_01",
         "july_kaiser9_02", "july_kaiser9_03")
OUT = Path("output_data/notes/keygen_universal_candidate_20261107")


def stream(res, party: str) -> np.ndarray:
    words = [L.word_bits(res.bins[party][i: i + 1], int(res.widths[i]))[0]
             for i in np.flatnonzero(res.kept)]
    return np.concatenate(words) if words else np.zeros(0, np.uint8)


def main() -> None:
    manifest = json.loads(core.PARENT.read_text())
    rows = []
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        _, res = G.run(raw, G.baseline_config())
        sel = res.policy_selection
        raw_bits = int(res.widths[res.kept].sum())
        times = [s.elapsed_s for s in raw.steps]
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        rec = L.run(res, L.LdpcConfig(block_bits=int(sel["block_bits"]), soft=True,
                                      public_reliability=False),
                    session_id=case, times=times)
        pa = P.run(rec, res, P.PAConfig(eve_from_frames=False))
        raw_fails = {p: [t.name for t in run_all(stream(res, p))
                         if t.passed is False] for p in ("alice", "bob")}
        key_fails = ([t.name for t in run_all(np.asarray(pa.key_alice, np.uint8))
                      if t.passed is False] if pa.length >= 100 else None)
        row = {
            "case": case,
            "pilot BDR %": round(100 * sel["pilot_bdr_4bit"], 2),
            "selected bits": sorted(set(sel["profile_bits"].values())),
            "block": int(sel["block_bits"]),
            "L_R": raw_bits,
            "BDR %": round(100 * (C.summary(res)["BDR p̂_e"] or 0), 2),
            "NIST raw A/B": f"{len(raw_fails['alice'])}/{len(raw_fails['bob'])}",
            "NIST raw names": raw_fails,
            "N": pa.n,
            "debit": int(sel["public_decision_bits"]),
            "ell": pa.length,
            "bit/s": round(pa.length / duration, 2) if duration and pa.length else None,
            "keys equal": bool(pa.keys_equal) if pa.length else None,
            "NIST final": key_fails,
        }
        rows.append(row)
        print("DONE", json.dumps(row, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e2e.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
