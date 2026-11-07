"""Graded per-component widths: end-to-end score on the research records.

Rule: a component with at least `n_min` pilot observations and a low pilot BDR
gets more key bits; a high BDR gets fewer; a component whose pilot is too short
to judge takes the record's fallback width. Measured for a per-component CDF,
one pooled CDF, and the hybrid (own CDF when the component has full statistics,
one shared pooled fallback for rare components).

Every variant goes through the production LDPC and privacy amplification, so the
number that matters is the final key length at an acceptable raw-key NIST.
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
from research.universal_matrix import (
    calibrate,
    component_quality,
    graded_widths,
    stream,
)

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_00", "july_kaiser9_01",
         "july_kaiser9_02", "july_kaiser9_03")
OUT = core.PARENT.parent / "output_data/notes/keygen_graded_widths_20261107"


def rules(quality, fallback):
    """Width maps to compare on one record."""
    return {
        "now-3/4": None,  # production: 4 bits when the pilot is good, else 3
        "graded": graded_widths(quality, fallback=fallback),
        "graded-tight": graded_widths(quality, fallback=fallback, t4=0.01, t3=0.03),
        "graded-5bit": graded_widths(quality, fallback=fallback, t4=0.01, t3=0.03,
                                     top=0.005, n_top=100),
    }


def evaluate(case, raw, selection, quality, rule, width_map):
    cdf = rule["cdf"]
    kw = {"cdf_cdf": cdf, "n_target": 1000 if cdf == "pooled" else 200,
          "preprocess": "rms", "rms_window": 50, "checks": 1,
          "rebalance": "per_profile"}
    if width_map is None:
        res = calibrate(raw, selection, widths="pilot", **kw)
    else:
        res = calibrate(raw, selection, width_map=width_map, **kw)
    raw_bits = int(res.widths[res.kept].sum())
    n = G.block_bits(raw_bits)
    sel = dict(selection) | {"block_bits": n,
                             "public_decision_bits": 2 * raw.status.shape[1] + 2}
    res = replace(res, policy_selection=sel)
    times = [s.elapsed_s for s in raw.steps]
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    rec = L.run(res, L.LdpcConfig(block_bits=n, soft=True,
                                  public_reliability=False),
                session_id=f"{case}-{rule['name']}-{cdf}", times=times)
    pa = P.run(rec, res, P.PAConfig(eve_from_frames=False))
    fails = {p: [t.name for t in run_all(stream(res, p)) if t.passed is False]
             for p in ("alice", "bob")}
    key_fails = ([t.name for t in run_all(np.asarray(pa.key_alice, np.uint8))
                  if t.passed is False] if pa.length >= 100 else None)
    widths = np.bincount(res.widths[res.kept], minlength=6)
    return {
        "case": case, "cdf": cdf, "rule": rule["name"],
        "widths": {int(b): int(c) for b, c in enumerate(widths) if c and b},
        "L_R": raw_bits, "BDR %": round(100 * (C.summary(res)["BDR p̂_e"] or 0), 2),
        "NIST raw": f"{len(fails['alice'])}/{len(fails['bob'])}",
        "NIST raw names": fails,
        "N": pa.n, "debit": sel["public_decision_bits"], "ell": pa.length,
        "bit/s": round(pa.length / duration, 2) if duration and pa.length else None,
        "keys equal": bool(pa.keys_equal) if pa.length else None,
        "NIST final": key_fails,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(CASES))
    parser.add_argument("--cdfs", default="per_profile,pooled,hybrid")
    args = parser.parse_args()
    manifest = json.loads(core.PARENT.read_text())
    cdfs = tuple(c for c in args.cdfs.split(",") if c)
    rows = []
    for case in args.cases:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        selection = G.select(raw)
        quality = component_quality(raw)
        fallback = 4 if (selection["pilot_bdr_4bit"] <= G.PILOT_BDR_THRESHOLD
                         and not selection["short_record"]) else 3
        print(f"\n== {case} pilot bdr={100 * selection['pilot_bdr_4bit']:.2f}% "
              f"fallback={fallback} quality="
              + ",".join(f"{j}:{n}/{'-' if b is None else f'{100 * b:.1f}'}"
                         for j, (n, b) in quality.items() if n), flush=True)
        for name, wmap in rules(quality, fallback).items():
            for cdf in cdfs:
                row = evaluate(case, raw, selection, quality,
                               {"name": name, "cdf": cdf}, wmap)
                rows.append(row)
                print(f"   {name:14s} {cdf:12s} b={row['widths']} L_R={row['L_R']:6d} "
                      f"BDR={row['BDR %']:5.2f}% NIST={row['NIST raw']:4s} "
                      f"N={row['N']:6d} ell={row['ell']:6d} bit/s={row['bit/s']}",
                      flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graded.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
