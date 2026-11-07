"""Candidate keygen presets: one causal chain, measured on the six records.

Every variant keeps the standard ACF admission, the causal linear9 detrend,
the local RMS scale, one growing CDF per component and the graded widths.
They differ only in how much is spent on public checking and how early the
CDF is considered full:

- recommended : CDF sample 200, one check bit, no guard
- fast        : CDF sample 100, one check bit, no guard (more key values)
- clean       : CDF sample 200, two check bits, no guard (lower BDR)
- low-error   : CDF sample 200, two check bits, guard band 5 %

Reported: raw and reconciled length, BDR, raw-key NIST, final key and rate.
"""
import json
from dataclasses import replace

import numpy as np

from pages.key_generation.compute.nist.runner import run_all
from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import growing_policy as G
from pages.offline_keygen_v2 import ldpc as L
from pages.offline_keygen_v2 import privacy as P
from research import linear_key_scheme as core
from research.universal_matrix import calibrate, stream

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_00", "july_kaiser9_01",
         "july_kaiser9_02", "july_kaiser9_03")
VARIANTS = {
    "sample300": {"n_target": 300, "checks": 1, "guard": 0.0},
    "sample500": {"n_target": 500, "checks": 1, "guard": 0.0},
    "sample1000": {"n_target": 1000, "checks": 1, "guard": 0.0},
}


def main():
    manifest = json.loads(core.PARENT.read_text())
    rows = []
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        selection = G.select(raw)
        times = [s.elapsed_s for s in raw.steps]
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        print(f"\n== {case} b={sorted(set(selection['profile_bits'].values()))} "
              f"pilot={100 * selection['pilot_bdr_4bit']:.2f}%", flush=True)
        for name, kw in VARIANTS.items():
            res = calibrate(raw, selection, cdf_cdf="per_profile", preprocess="rms",
                            rms_window=50, rebalance="per_profile", widths="pilot", **kw)
            raw_bits = int(res.widths[res.kept].sum())
            n = G.block_bits(raw_bits)
            used = int(np.unique(res.profile).size)
            sel = dict(selection) | {"block_bits": n, "public_decision_bits": used + 2}
            res = replace(res, policy_selection=sel)
            rec = L.run(res, L.LdpcConfig(block_bits=n, soft=True,
                                          public_reliability=False),
                        session_id=f"{case}-{name}", times=times)
            pa = P.run(rec, res, P.PAConfig(eve_from_frames=False))
            fails = {p: [t.name for t in run_all(stream(res, p)) if t.passed is False]
                     for p in ("alice", "bob")}
            key_fails = ([t.name for t in run_all(np.asarray(pa.key_alice, np.uint8))
                          if t.passed is False] if pa.length >= 100 else None)
            row = {"case": case, "variant": name, **kw, "L_R": raw_bits,
                   "BDR %": round(100 * (C.summary(res)["BDR p̂_e"] or 0), 2),
                   "NIST raw": f"{len(fails['alice'])}/{len(fails['bob'])}",
                   "NIST raw names": fails, "N": pa.n, "debit": used + 2,
                   "ell": pa.length,
                   "bit/s": round(pa.length / duration, 2) if duration and pa.length else None,
                   "keys equal": bool(pa.keys_equal) if pa.length else None,
                   "NIST final": key_fails}
            rows.append(row)
            print(f"   {name:11s} n={kw['n_target']:3d} c={kw['checks']} g={kw['guard']:.2f} "
                  f"L_R={row['L_R']:6d} BDR={row['BDR %']:5.2f}% "
                  f"NIST={row['NIST raw']:4s} N={row['N']:6d} debit={row['debit']:2d} "
                  f"ell={row['ell']:6d} bit/s={row['bit/s']} fin={key_fails}",
                  flush=True)
    out = core.PARENT.parent / "output_data/notes/keygen_presets_20261107"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
