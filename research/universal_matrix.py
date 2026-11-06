"""Matrix: ACF admission -> linear9 detrend -> CDF quant, one pass.

Varied: the CDF that maps a residual to u (one pooled history, or one history
per Delta-tau component with its own sample), the public rejection mechanism
(2 check bits + rebalance, 1 check bit, or a guard band), and the word width
(pilot 3/4 per component or one width for all). Raw-key NIST is measured on
the stream LDPC reads: whole Gray words in time order. Measurement only.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pages.key_generation.compute.nist.runner import run_all
from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import growing_policy as G
from pages.offline_keygen_v2 import ldpc as L
from research import linear_key_scheme as core

OUT = Path("output_data/notes/keygen_universal_matrix_20261107")
CASES = ("kazan_v4_00", "kazan_long_00", "july_kaiser9_00", "july_kaiser9_01",
         "july_kaiser9_02", "july_kaiser9_03")
PARTIES = ("alice", "bob")


def stream(res, party: str) -> np.ndarray:
    words = [L.word_bits(res.bins[party][i: i + 1], int(res.widths[i]))[0]
             for i in np.flatnonzero(res.kept)]
    return np.concatenate(words) if words else np.zeros(0, np.uint8)


def nist(seq: np.ndarray) -> list[str]:
    return [t.name for t in run_all(seq) if t.passed is False] if seq.size >= 100 else []


def local_rms(z: dict[str, np.ndarray], window: int = 20) -> dict[str, np.ndarray]:
    """Divide each residual by the RMS of the previous `window` residuals of its column."""
    out = {}
    for party, a in z.items():
        y = a.copy()
        for j in range(a.shape[1]):
            ix = np.flatnonzero(np.isfinite(a[:, j]))
            v = a[ix, j]
            for k in range(window, len(v)):
                scale = float(np.sqrt(np.mean(v[k - window:k] ** 2)))
                if scale > 0:
                    y[ix[k], j] = v[k] / scale
        out[party] = y
    return out


def _rebalanced_u_windowed(u, steps, key_of, fine, c, cfg, window):
    """`C._rebalanced_u` with the kept-u reference capped to its last `window` values."""
    out = {p: v.copy() for p, v in u.items()}
    low = (1 << c) - 1
    for k in np.unique(key_of):
        rows = np.flatnonzero(key_of == k)
        hist = {"alice": [], "bob": []}
        pending = []
        for i, n in enumerate(rows):
            if i and steps[n] != steps[rows[i - 1]]:
                for p, h in hist.items():
                    h.extend(u[p][m] for m in pending)
                    del h[:-window]
                pending = []
            if len(hist["alice"]) >= cfg.rebalance_min:
                for p, row in out.items():
                    ref = np.asarray(hist[p])
                    row[n] = (np.count_nonzero(ref < u[p][n])
                              + np.count_nonzero(ref <= u[p][n])) / 2 / ref.size
            b = {p: C.quantize(out[p][n:n + 1], fine[n:n + 1], 0.0)[0][0] for p in out}
            if (b["alice"] & low) == (b["bob"] & low):
                pending.append(int(n))
    return out


def calibrate(raw, selection, *, cdf_cdf="per_profile", n_target=300, rebalance=None,
              checks=0, guard=0.0, widths="pilot", preprocess="linear9",
              update="growing", rms_window=20, rebalance_window=0) -> C.CalibResult:
    """One pass: admitted residual -> own/pooled CDF -> words -> public mask."""
    z = G.residual(raw.x)
    work = replace(raw, x=local_rms(z, rms_window) if preprocess == "rms" else z)
    pilot_end = int(selection["pilot_end_step"])
    cfg = C.CalibConfig(
        algorithm="standard", n_target=n_target, cdf=cdf_cdf, bits=4, bits_rule="fixed",
        update=update, guard=guard, check_bits=0, check_extra=True, rebalance=False,
        detrend="off",
    )
    base = C.calibrate(work, cfg)
    keep = base.step > pilot_end  # the excluded pilot never becomes key bits
    sub = lambda a: a[keep]  # noqa: E731
    base = replace(base, step=sub(base.step), profile=sub(base.profile),
                   u={p: sub(v) for p, v in base.u.items()},
                   bins={p: sub(v) for p, v in base.bins.items()},
                   erased=sub(base.erased), value_bits=sub(base.widths),
                   cdf_size=sub(base.cdf_size) if base.cdf_size.size == keep.size
                   else np.zeros(int(keep.sum()), np.int64))
    w = (np.array([selection["profile_bits"][str(int(p))] for p in base.profile], np.int64)
         if widths == "pilot" else np.full(base.step.size, int(widths), np.int64))
    u = base.u
    if rebalance:
        result_cfg = replace(cfg, cdf=rebalance, check_bits=checks, rebalance=True)
        groups = base.profile if rebalance == "per_profile" else np.zeros_like(base.profile)
        u = (_rebalanced_u_windowed(base.u, base.step, groups, w + checks, checks,
                                    result_cfg, rebalance_window)
             if rebalance_window else
             C._rebalanced_u(base.u, base.step, groups, w + checks, checks, result_cfg))
    fine, erased = {}, base.erased.copy()
    for p in PARTIES:
        fine[p], band = C.quantize(u[p], w + checks, guard)
        erased |= band
    low = (1 << checks) - 1 if checks else 0
    failed = ((fine["alice"] & low) != (fine["bob"] & low)) if checks else np.zeros_like(erased)
    erased |= failed
    bins = {p: v >> checks for p, v in fine.items()}
    bits = {p: [format(int(v), f"0{int(b)}b")
                for v, b in zip(C.gray(bins[p][~erased]), w[~erased], strict=True)]
            for p in PARTIES}
    return replace(base, config=replace(cfg, cdf=rebalance or cdf_cdf, check_bits=checks),
                   u=u, bins=bins, erased=erased, bits=bits, value_bits=w,
                   check_fail=failed, detrended=True, decision=None)


def report(res) -> dict:
    s = C.summary(res)
    seq = {p: stream(res, p) for p in PARTIES}
    out = {"L_R": s["L_R raw key bits"], "BDR %": round(100 * (s["BDR p̂_e"] or 0), 2),
           "ones A": round(float(seq["alice"].mean()), 4) if seq["alice"].size else None}
    for p in PARTIES:
        fails = nist(seq[p])
        out[f"NIST {p}"] = f"{len(fails)}/9" + (f" {fails}" if fails else "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(CASES))
    parser.add_argument("--grid", default="reb")
    args = parser.parse_args()
    base = dict(cdf_cdf="per_profile", n_target=300, rebalance=None, checks=0,
                guard=0.0, widths="pilot", preprocess="linear9", update="growing")
    grids = {
        "reb": [
            ("slide300-c2-rw500", base | dict(update="sliding", checks=2,
                                              rebalance="per_profile", rebalance_window=500)),
            ("slide300-c2-rw200", base | dict(update="sliding", checks=2,
                                              rebalance="per_profile", rebalance_window=200)),
            ("slide300-c1-rw500", base | dict(update="sliding", checks=1,
                                              rebalance="per_profile", rebalance_window=500)),
            ("grow200-c2-rw500", base | dict(n_target=200, checks=2,
                                             rebalance="per_profile", rebalance_window=500)),
            ("rms50-grow300-c2-rw500", base | dict(preprocess="rms", rms_window=50, checks=2,
                                                   rebalance="per_profile", rebalance_window=500)),
        ],
        "g": [
            ("c1-g05", base | dict(preprocess="rms", rms_window=50, n_target=200, checks=1,
                                   guard=0.05, rebalance="per_profile")),
            ("c1-g10", base | dict(preprocess="rms", rms_window=50, n_target=200, checks=1,
                                   guard=0.10, rebalance="per_profile")),
            ("c1-g15", base | dict(preprocess="rms", rms_window=50, n_target=200, checks=1,
                                   guard=0.15, rebalance="per_profile")),
            ("c2-g05", base | dict(preprocess="rms", rms_window=50, n_target=200, checks=2,
                                   guard=0.05, rebalance="per_profile")),
            ("c2-g10", base | dict(preprocess="rms", rms_window=50, n_target=200, checks=2,
                                   guard=0.10, rebalance="per_profile")),
            ("c1-g05-n300", base | dict(preprocess="rms", rms_window=50, n_target=300, checks=1,
                                        guard=0.05, rebalance="per_profile")),
        ],
        "tune": [
            ("pp-rms50-c1-n200", base | dict(preprocess="rms", rms_window=50, n_target=200,
                                             checks=1, rebalance="per_profile")),
            ("pp-rms50-c1-n300", base | dict(preprocess="rms", rms_window=50, checks=1,
                                             rebalance="per_profile")),
            ("pp-rms50-slide300-c1", base | dict(preprocess="rms", rms_window=50,
                                                 update="sliding", checks=1,
                                                 rebalance="per_profile")),
            ("pp-rms20-slide300-c1", base | dict(preprocess="rms", update="sliding", checks=1,
                                                 rebalance="per_profile")),
            ("pp-rms50-c2-n100", base | dict(preprocess="rms", rms_window=50, n_target=100,
                                             checks=2, rebalance="per_profile")),
            ("pp-rms50-c2-n200", base | dict(preprocess="rms", rms_window=50, n_target=200,
                                             checks=2, rebalance="per_profile")),
            ("pp-slide300-c1-n150", base | dict(update="sliding", n_target=150, checks=1,
                                                rebalance="per_profile")),
            ("rms50-slide300-c2", base | dict(preprocess="rms", rms_window=50, update="sliding",
                                              checks=2, rebalance="per_profile")),
        ],
        "guard": [
            ("pp-guard10", base | dict(guard=0.10)),
            ("pp-check1", base | dict(checks=1, rebalance="per_profile")),
            ("pp-check2", base | dict(checks=2, rebalance="per_profile")),
            ("pool-check2-b4", base | dict(cdf_cdf="pooled", n_target=1000, checks=2,
                                           rebalance="per_profile", widths=4)),
        ],
    }
    rows = []
    for case in args.cases:
        raw = core.read(json.loads(core.PARENT.read_text())["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        admitted = (raw.status == G.D.ADMITTED).sum(axis=0)
        selection = G.select(raw, "per_profile")
        print(f"\n== {case} steps={len(raw.steps)} admitted/profile={admitted.tolist()}\n"
              f"   pilot bdr={selection['pilot_bdr_4bit']:.4f} block={selection['block_bits']}",
              flush=True)
        for name, kw in grids[args.grid]:
            res = calibrate(raw, selection, **kw)
            row = {"case": case, "variant": name} | {
                k: v for k, v in kw.items() if k != "cdf_cdf"} | {"cdf": kw["cdf_cdf"]} | report(res)
            rows.append(row)
            print("  ", json.dumps(row, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"matrix_{args.grid}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
