"""Per-component check bits: two where the pilot is noisy, one where it is clean.

The notes say two thirds of the flipped bits come from ~9 % of values that land
on different rays, and that a second check bit removes them but also erases
good values. This grid asks whether spending the second check bit only on the
components that need it keeps more pre-hash length at a lower BDR.

Reference: graded widths, one check everywhere. BDR is reported over all kept
raw bits; L_R is the length before hashing.
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

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_02", "july_kaiser9_03")


def plan(raw):
    selection = G.select(raw)
    bdr = selection["profile_pilot_bdr"]
    widths = {int(k): v for k, v in selection["profile_bits"].items()}
    flat3 = {j: 3 for j in widths}
    flat4 = {j: 4 for j in widths}

    def checks(limit):
        return {j: (2 if bdr.get(str(j)) is not None and bdr[str(j)] > limit else 1)
                for j in widths}

    return widths, {
        "graded-c1": (widths, 1),
        "graded-c2": (widths, 2),
        "graded-c2@1%": (widths, checks(0.01)),
        "graded-c2@2%": (widths, checks(0.02)),
        "flat3-c2@1%": (flat3, checks(0.01)),
        "flat4-c2@1%": (flat4, checks(0.01)),
    }


def main():
    manifest = json.loads(core.PARENT.read_text())
    rows = []
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        times = [s.elapsed_s for s in raw.steps]
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        widths, variants = plan(raw)
        print(f"\n== {case} widths={sorted(set(widths.values()))}", flush=True)
        for name, (wmap, checks) in variants.items():
            res = calibrate(raw, G.select(raw), cdf_cdf="per_profile", preprocess="rms",
                            rms_window=50, rebalance="per_profile", width_map=wmap,
                            checks=checks, n_target=G.CDF_SAMPLE)
            lr = int(res.widths[res.kept].sum())
            n = G.block_bits(lr)
            used = int(np.unique(res.profile).size)
            res = replace(res, policy_selection={"block_bits": n,
                                                 "public_decision_bits": used + 2})
            rec = L.run(res, L.LdpcConfig(block_bits=n, soft=True,
                                          public_reliability=False),
                        session_id=f"{case}-{name}", times=times)
            pa = P.run(rec, res, P.PAConfig(eve_from_frames=False))
            tested = {p: run_all(stream(res, p)) for p in ("alice", "bob")}
            fails = {p: [t.name for t in tested[p] if t.passed is False]
                     for p in ("alice", "bob")}
            row = {"case": case, "variant": name, "L_R": lr,
                   "kept": int(res.kept.sum()),
                   "BDR %": round(100 * (C.summary(res)["BDR p̂_e"] or 0), 2),
                   "NIST raw": f"{len(fails['alice'])}/{len(fails['bob'])}",
                   "N": pa.n, "ell": pa.length,
                   "ell/L_R": round(pa.length / lr, 3) if lr else None,
                   "bit/s": round(pa.length / duration, 2) if duration and pa.length else None}
            rows.append(row)
            print(f"   {name:14s} L_R={row['L_R']:6d} kept={row['kept']:5d} "
                  f"BDR={row['BDR %']:5.2f}% NIST={row['NIST raw']:4s} N={row['N']:6d} "
                  f"ell={row['ell']:6d} ell/L_R={row['ell/L_R']} bit/s={row['bit/s']}",
                  flush=True)
    out = core.PARENT.parent / "output_data/notes/keygen_presets_20261107"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_component_checks.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
