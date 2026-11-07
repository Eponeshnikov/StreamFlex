"""Per-component pilot quality vs the actual per-component error rate.

For every record: each Delta-tau component's pilot observation count and its
pilot Gray BDR (4 key + 2 check bits, both checks agreeing), then the real
per-component BDR of the post-pilot key at 2, 3 and 4 key bits with one check
bit, for a pooled and for a per-component CDF. Measurement only.
"""
import json

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import growing_policy as G
from research import linear_key_scheme as core
from research.universal_matrix import calibrate

CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_02", "july_kaiser9_03")
PARTIES = ("alice", "bob")


def component_pilot_bdr(work, steps, profiles, n_components, checks=2, bits=4):
    """Pilot BDR per component at `bits` key + `checks` check bits."""
    out = {}
    levels = 1 << (bits + checks)
    for j in range(n_components):
        take = profiles == j
        s, p = steps[take], profiles[take]
        if len(s) < 8:
            out[j] = (len(s), None)
            continue
        fine = {}
        for party in PARTIES:
            values = work[party][s, p]
            ordered = np.sort(values)
            u = (np.searchsorted(ordered, values, "left")
                 + np.searchsorted(ordered, values, "right")) / (2 * len(ordered))
            fine[party] = np.floor(u * levels).clip(0, levels - 1).astype(np.int64)
        keep = (fine["alice"] & ((1 << checks) - 1)) == (fine["bob"] & ((1 << checks) - 1))
        if keep.sum() < 4:
            out[j] = (len(s), None)
            continue
        a = C.gray(fine["alice"][keep] >> checks)
        b = C.gray(fine["bob"][keep] >> checks)
        out[j] = (len(s), float(np.bitwise_count(a ^ b).sum() / (bits * keep.sum())))
    return out


def component_actual(res, n_components):
    """Kept values, raw bits and BDR per component of a calibrated result."""
    out = {}
    for j in range(n_components):
        take = res.kept & (res.profile == j)
        bits = int(res.widths[take].sum())
        flips = int(np.bitwise_count(
            C.gray(res.bins["alice"][take]) ^ C.gray(res.bins["bob"][take])).sum())
        out[j] = (int(take.sum()), bits, flips / bits if bits else None)
    return out


def main():
    manifest = json.loads(core.PARENT.read_text())
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        n_comp = raw.status.shape[1]
        z = G.residual(raw.x)
        work = G.local_rms(z)
        sel = G.select(raw)
        steps, profiles = G._pilot(raw, z)
        usable = np.isfinite(work["alice"][steps, profiles]) & np.isfinite(
            work["bob"][steps, profiles])
        pilot = component_pilot_bdr(work, steps[usable], profiles[usable], n_comp)
        print(f"\n== {case}  pilot={sel['pilot_values']} used={int(usable.sum())} "
              f"bdr={sel['pilot_bdr_4bit']:.4f}")
        tables = {}
        for cdf, nt in (("per_profile", 200), ("pooled", 1000)):
            for w in (2, 3, 4):
                res = calibrate(raw, sel, cdf_cdf=cdf, n_target=nt, preprocess="rms",
                                rms_window=50, checks=1, rebalance="per_profile",
                                widths=w, update="growing")
                tables[(cdf, w)] = component_actual(res, n_comp)
        print("  j  pilot_n  bdr4+2 | pp w2 n/bdr   pp w3 n/bdr   pp w4 n/bdr"
              " | po w2 n/bdr   po w3 n/bdr   po w4 n/bdr")

        def cell(t):
            n, _bits, bdr = t
            return f"{n:4d}/{'  -  ' if bdr is None else f'{100 * bdr:4.1f}'}"

        for j in range(n_comp):
            n, bdr = pilot[j]
            if n == 0 and tables[("pooled", 4)][j][0] == 0:
                continue
            print(f" {j:2d} {n:7d} {('  -  ' if bdr is None else f'{100 * bdr:5.1f}')} | "
                  + " ".join(cell(tables[(cdf, w)][j])
                             for cdf in ("per_profile", "pooled") for w in (2, 3, 4)))

        for cdf in ("per_profile", "pooled"):
            for w in (2, 3, 4):
                t = tables[(cdf, w)]
                bits = sum(v[1] for v in t.values())
                flips = sum(v[2] * v[1] for v in t.values() if v[2] is not None)
                print(f"  total {cdf:12s} w={w}: kept={sum(v[0] for v in t.values()):5d} "
                      f"bits={bits:6d} BDR={100 * flips / bits if bits else 0:5.2f}%")


if __name__ == "__main__":
    main()
