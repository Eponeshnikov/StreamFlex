"""Second preset grid: 3-bit words, a third check bit, longer detrend windows.

The first grid showed a larger CDF sample only shortens the key (v4 8,605 ->
6,159 -> 3,927 -> 980 at 200/300/500/1000) and a guard band is useless. This
grid tests the remaining one-pass knobs that could define a "low error" or a
"high randomness" preset without adding a second pass or a rank transform.
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
from research.universal_matrix import (
    calibrate,
    component_quality,
    local_rms,
    residual_window,
    stream,
)

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_02", "july_kaiser9_03")


def width_map_for(raw, trend_window, t4=G.GRADED_T4):
    """Graded widths measured on the same detrend window the key uses."""
    z = residual_window(raw.x, trend_window)
    work = local_rms(z, 50)
    quality = component_quality(raw, z, work)
    steps, profiles = G._pilot(raw, z)
    pooled = G.pooled_pilot_bdr(work, steps, profiles)
    fallback = 4 if pooled <= G.PILOT_BDR_THRESHOLD and len(raw.steps) >= 5000 else 3
    cap = G.SHORT_RECORD_MAX_BITS if len(raw.steps) < 5000 else 4
    return {j: min(G.graded_width(q["n"], q["bdr"], fallback), cap)
            for j, q in quality.items()}


VARIANTS = {
    "rec": {"checks": 1, "n_target": 200, "trend_window": 9},
    "c2-3bit": {"checks": 2, "n_target": 200, "trend_window": 9, "flat3": True},
    "c3-graded": {"checks": 3, "n_target": 200, "trend_window": 9},
    "tw15": {"checks": 1, "n_target": 200, "trend_window": 15},
    "tw25": {"checks": 1, "n_target": 200, "trend_window": 25},
}


def main():
    manifest = json.loads(core.PARENT.read_text())
    rows = []
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        times = [s.elapsed_s for s in raw.steps]
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        print(f"\n== {case}", flush=True)
        for name, kw in VARIANTS.items():
            mpc = width_map_for(raw, kw["trend_window"])
            if kw.get("flat3"):
                mpc = {j: 3 for j in mpc}
            res = calibrate(raw, G.select(raw), cdf_cdf="per_profile", preprocess="rms",
                            rms_window=50, rebalance="per_profile", width_map=mpc,
                            checks=kw["checks"], n_target=kw["n_target"],
                            trend_window=kw["trend_window"])
            raw_bits = int(res.widths[res.kept].sum())
            n = G.block_bits(raw_bits)
            used = int(np.unique(res.profile).size)
            sel = {"block_bits": n, "public_decision_bits": used + 2}
            res = replace(res, policy_selection=sel)
            rec = L.run(res, L.LdpcConfig(block_bits=n, soft=True,
                                          public_reliability=False),
                        session_id=f"{case}-{name}", times=times)
            pa = P.run(rec, res, P.PAConfig(eve_from_frames=False))
            fails = {p: [t.name for t in run_all(stream(res, p)) if t.passed is False]
                     for p in ("alice", "bob")}
            key_fails = ([t.name for t in run_all(np.asarray(pa.key_alice, np.uint8))
                          if t.passed is False] if pa.length >= 100 else None)
            seq = stream(res, "alice")
            row = {"case": case, "variant": name, **{k: v for k, v in kw.items()
                                                     if k != "flat3"},
                   "L_R": raw_bits, "BDR %": round(100 * (C.summary(res)["BDR p̂_e"] or 0), 2),
                   "ones": round(float(seq.mean()), 4) if seq.size else None,
                   "NIST raw": f"{len(fails['alice'])}/{len(fails['bob'])}",
                   "NIST raw names": fails, "N": pa.n, "ell": pa.length,
                   "bit/s": round(pa.length / duration, 2) if duration and pa.length else None,
                   "NIST final": key_fails}
            rows.append(row)
            print(f"   {name:10s} c={kw['checks']} n={kw['n_target']:3d} tw={kw['trend_window']:2d} "
                  f"L_R={row['L_R']:6d} BDR={row['BDR %']:5.2f}% ones={row['ones']} "
                  f"NIST={row['NIST raw']:4s} N={row['N']:6d} ell={row['ell']:6d} "
                  f"bit/s={row['bit/s']} fin={key_fails}", flush=True)
    out = core.PARENT.parent / "output_data/notes/keygen_presets_20261107"
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid2.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
