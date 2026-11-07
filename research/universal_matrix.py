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


def component_quality(raw, z: dict | None = None, work: dict | None = None,
                      checks: int = 2, bits: int = 4) -> dict[int, tuple[int, float | None]]:
    """(pilot observations, pilot Gray BDR) per Delta-tau component.

    The BDR is measured at `bits` key + `checks` check bits on the pilot values
    where both checks agree. Components with fewer than 8 pilot observations or
    4 agreeing checks get `None`: the estimate is not trustworthy.
    """
    z = G.residual(raw.x) if z is None else z
    work = G.local_rms(z) if work is None else work
    steps, profiles = G._pilot(raw, z)
    usable = (np.isfinite(work["alice"][steps, profiles])
              & np.isfinite(work["bob"][steps, profiles]))
    steps, profiles = steps[usable], profiles[usable]
    levels = 1 << (bits + checks)
    low = (1 << checks) - 1
    out = {}
    for j in range(raw.status.shape[1]):
        take = profiles == j
        n = int(take.sum())
        if n < 8:
            out[j] = (n, None)
            continue
        fine = {}
        for party in PARTIES:
            values = work[party][steps[take], profiles[take]]
            ordered = np.sort(values)
            u = (np.searchsorted(ordered, values, "left")
                 + np.searchsorted(ordered, values, "right")) / (2 * len(ordered))
            fine[party] = np.floor(u * levels).clip(0, levels - 1).astype(np.int64)
        keep = (fine["alice"] & low) == (fine["bob"] & low)
        if keep.sum() < 4:
            out[j] = (n, None)
            continue
        a = C.gray(fine["alice"][keep] >> checks)
        b = C.gray(fine["bob"][keep] >> checks)
        out[j] = (n, float(np.bitwise_count(a ^ b).sum() / (bits * keep.sum())))
    return out


def graded_widths(quality: dict[int, tuple[int, float | None]], *,
                  fallback: int = 3, t4: float = 0.015, t3: float = 0.05,
                  n_min: int = 30, top: float | None = None,
                  n_top: int = 100) -> dict[int, int]:
    """More bits to components with enough statistics and a low pilot BDR.

    A component with fewer than `n_min` pilot observations has no trustworthy
    estimate and takes the record's `fallback` width (never an extreme one).
    `top` promotes a component with at least `n_top` observations and a BDR at
    or below it to five bits.
    """
    out = {}
    for j, (n, bdr) in quality.items():
        if bdr is None or n < n_min:
            out[j] = fallback
        elif top is not None and bdr <= top and n >= n_top:
            out[j] = 5
        elif bdr <= t4:
            out[j] = 4
        elif bdr <= t3:
            out[j] = 3
        else:
            out[j] = 2
    return out


def _subset(res: C.CalibResult, mask: np.ndarray) -> C.CalibResult:
    return replace(res, step=res.step[mask], profile=res.profile[mask],
                   u={q: v[mask] for q, v in res.u.items()},
                   bins={q: v[mask] for q, v in res.bins.items()},
                   erased=res.erased[mask], value_bits=res.widths[mask],
                   cdf_size=res.cdf_size[mask], calib_end={})


def _concat_results(parts: list[C.CalibResult]) -> C.CalibResult:
    """Concatenate key rows of several CDF passes, restoring time order."""
    order = np.argsort(np.concatenate([p.step for p in parts]), kind="stable")
    first = parts[0]
    return replace(
        first,
        step=np.concatenate([p.step for p in parts])[order],
        profile=np.concatenate([p.profile for p in parts])[order],
        u={q: np.concatenate([p.u[q] for p in parts])[order] for q in PARTIES},
        bins={q: np.concatenate([p.bins[q] for p in parts])[order] for q in PARTIES},
        erased=np.concatenate([p.erased for p in parts])[order],
        value_bits=np.concatenate([p.widths for p in parts])[order],
        cdf_size=np.concatenate([p.cdf_size for p in parts])[order],
        calib_end={k: v for p in parts for k, v in p.calib_end.items()},
    )


def residual_window(x: dict[str, np.ndarray], window: int) -> dict[str, np.ndarray]:
    """`G.residual` with a configurable trend window."""
    out = {}
    weights = G.trend_weights(window)
    for party, a in x.items():
        z = np.full(a.shape, np.nan)
        for j in range(a.shape[1]):
            ix = np.flatnonzero(np.isfinite(a[:, j]))
            v = a[ix, j]
            if len(v) > window:
                history = np.lib.stride_tricks.sliding_window_view(v[:-1], window)
                z[ix[window:], j] = v[window:] - history @ weights
        out[party] = z
    return out


def calibrate(raw, selection, *, cdf_cdf="per_profile", n_target=300, rebalance=None,
              checks=0, guard=0.0, widths="pilot", preprocess="linear9",
              update="growing", rms_window=20, rebalance_window=0,
              rebalance_min=None, width_map=None, rare_sample=32,
              trend_window=G.TREND_WINDOW) -> C.CalibResult:
    """One pass: admitted residual -> own/pooled CDF -> words -> public mask.

    `cdf_cdf="hybrid"` reads a component on its own CDF only when that component
    has collected `n_target` admitted values; rarer components share one pooled
    fallback CDF instead of being dropped. `width_map` is a per-component width
    array that overrides `widths`.
    """
    z = (G.residual(raw.x) if trend_window == G.TREND_WINDOW
         else residual_window(raw.x, trend_window))
    work = replace(raw, x=local_rms(z, rms_window) if preprocess == "rms" else z)
    pilot_end = int(selection["pilot_end_step"])
    cfg = C.CalibConfig(
        algorithm="standard", n_target=n_target,
        cdf="per_profile" if cdf_cdf in ("hybrid", "rare_own") else cdf_cdf,
        bits=4, bits_rule="fixed", update=update, guard=guard, check_bits=0,
        check_extra=True, rebalance=False, detrend="off",
    )
    if cdf_cdf == "hybrid":
        own = C.calibrate(work, cfg)
        pooled = C.calibrate(work, replace(cfg, cdf="pooled"))
        full = set(map(int, own.calib_end))
        rare_rows = ~np.isin(pooled.profile, list(full))
        base = _concat_results([own, _subset(pooled, rare_rows)])
        base = replace(base, calib_end=own.calib_end)
    elif cdf_cdf == "rare_own":
        # Full components: own CDF, normal sample. Rare components: their own
        # CDF again, built from the `rare_sample` values they actually have.
        own = C.calibrate(work, cfg)
        full = set(map(int, own.calib_end))
        rare = [j for j in range(work.status.shape[1]) if j not in full]
        if rare:
            small = C.calibrate(work, replace(cfg, n_target=rare_sample))
            base = _concat_results([own, _subset(small, np.isin(small.profile, rare))])
            base = replace(base, calib_end=own.calib_end)
        else:
            base = own
    else:
        base = C.calibrate(work, cfg)
    keep = base.step > pilot_end  # the excluded pilot never becomes key bits
    sub = lambda a: a[keep]  # noqa: E731
    base = replace(base, step=sub(base.step), profile=sub(base.profile),
                   u={p: sub(v) for p, v in base.u.items()},
                   bins={p: sub(v) for p, v in base.bins.items()},
                   erased=sub(base.erased), value_bits=sub(base.widths),
                   cdf_size=sub(base.cdf_size) if base.cdf_size.size == keep.size
                   else np.zeros(int(keep.sum()), np.int64))
    if isinstance(width_map, dict):
        w = np.asarray([width_map[int(p)] for p in base.profile], np.int64)
    elif width_map is not None:
        w = np.full(base.step.size, int(width_map), np.int64)
    elif widths == "pilot":
        w = np.array([selection["profile_bits"][str(int(p))] for p in base.profile], np.int64)
    else:
        w = np.full(base.step.size, int(widths), np.int64)
    u = base.u
    if rebalance:
        result_cfg = replace(cfg, cdf=rebalance, check_bits=checks, rebalance=True,
                             rebalance_min=rebalance_min or cfg.rebalance_min)
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
        "fine2": [
            ("rms20-reb50", base | dict(preprocess="rms", rms_window=20, n_target=200,
                                        checks=1, rebalance="per_profile", rebalance_min=50)),
            ("rms50-reb50", base | dict(preprocess="rms", rms_window=50, n_target=200,
                                        checks=1, rebalance="per_profile", rebalance_min=50)),
            ("rms50-reb100", base | dict(preprocess="rms", rms_window=50, n_target=200,
                                         checks=1, rebalance="per_profile", rebalance_min=100)),
            ("rms100", base | dict(preprocess="rms", rms_window=100, n_target=200,
                                   checks=1, rebalance="per_profile")),
            ("rms30", base | dict(preprocess="rms", rms_window=30, n_target=200,
                                  checks=1, rebalance="per_profile")),
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
        selection = G.select(raw)
        print(f"\n== {case} steps={len(raw.steps)} admitted/profile={admitted.tolist()}\n"
              f"   pilot bdr={selection['pilot_bdr_4bit']:.4f}",
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
