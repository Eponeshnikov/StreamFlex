"""Preset candidates through the production path (growing_policy.run).

Each candidate is a `CalibConfig` the page itself would build: the CDF sample,
the number of public check bits and (for the low-error point) the width cap.
Everything else — ACF admission, linear9, local RMS, per-component CDF, graded
widths, rebalance, LDPC, PA — is the shipped code.

Run: `python -m research.preset_production [variant ...]`
"""
import argparse
import json
from dataclasses import replace

import numpy as np

from pages.key_generation.compute.nist.runner import run_all
from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import growing_policy as G
from pages.offline_keygen_v2 import ldpc as L
from pages.offline_keygen_v2 import privacy as P
from research import linear_key_scheme as core
from research.universal_matrix import stream

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_00", "july_kaiser9_01",
         "july_kaiser9_02", "july_kaiser9_03")
VARIANTS = {
    "recommended": {"n_target": 200, "check_bits": 1, "bits": 4, "cdf": "per_profile"},
    "fast": {"n_target": 100, "check_bits": 1, "bits": 4, "cdf": "per_profile"},
    "high-rate": {"n_target": 200, "check_bits": 1, "bits": 4, "cdf": "pooled"},
    "clean": {"n_target": 200, "check_bits": 2, "bits": 4, "cdf": "per_profile"},
    "fidelity": {"n_target": 200, "check_bits": 3, "bits": 4, "cdf": "per_profile"},
    "low-error": {"n_target": 200, "check_bits": 2, "bits": 3, "cdf": "per_profile"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--out", default="presets.json")
    parser.add_argument("--cases", default="")
    args = parser.parse_args()
    manifest = json.loads(core.PARENT.read_text())
    rows = []
    cases = tuple(args.cases.split(",")) if args.cases else CASES
    for case in cases:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        times = [s.elapsed_s for s in raw.steps]
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        print(f"\n== {case}", flush=True)
        for name in args.variants:
            cfg = replace(G.baseline_config(), **VARIANTS[name])
            _, res = G.run(raw, cfg)
            sel = res.policy_selection
            n = int(sel.get("block_bits", 768))
            rec = L.run(res, L.LdpcConfig(block_bits=n, soft=True,
                                          public_reliability=False),
                        session_id=f"{case}-{name}", times=times)
            pa = P.run(rec, res, P.PAConfig(eve_from_frames=False))
            tested = {p: run_all(stream(res, p)) for p in ("alice", "bob")}
            fails = {p: [t.name for t in tested[p] if t.passed is False]
                     for p in ("alice", "bob")}
            applicable = {p: sum(t.passed is not None for t in tested[p])
                          for p in ("alice", "bob")}
            key_fails = ([t.name for t in run_all(np.asarray(pa.key_alice, np.uint8))
                          if t.passed is False] if pa.length >= 100 else None)
            lr = int(res.widths[res.kept].sum())
            row = {"case": case, "variant": name, **VARIANTS[name], "L_R": lr,
                   "BDR %": round(100 * (C.summary(res)["BDR p̂_e"] or 0), 2),
                   "NIST raw": f"{len(fails['alice'])}/{len(fails['bob'])}",
                   "NIST applicable": applicable,
                   "NIST raw names": fails, "N": pa.n,
                   "debit": int(sel.get("public_decision_bits", 0)),
                   "ell": pa.length,
                   "bit/s": round(pa.length / duration, 2) if duration and pa.length else None,
                   "keys equal": bool(pa.keys_equal) if pa.length else None,
                   "NIST final": key_fails}
            rows.append(row)
            print(f"   {name:11s} n={row['n_target']:3d} c={row['check_bits']} b≤{row['bits']} "
                  f"L_R={row['L_R']:6d} BDR={row['BDR %']:5.2f}% NIST={row['NIST raw']:4s} "
                  f"N={row['N']:6d} debit={row['debit']:2d} ell={row['ell']:6d} "
                  f"bit/s={row['bit/s']} fin={key_fails}", flush=True)
    out = core.PARENT.parent / "output_data/notes/keygen_presets_20261107"
    out.mkdir(parents=True, exist_ok=True)
    (out / args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
