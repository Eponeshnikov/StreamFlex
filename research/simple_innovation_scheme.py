"""Pilot-selected scalar innovations and ONE causal threshold quantizer.

Research only. Same rule on Kazan and July. No conditional ranks, RMS,
check-bit rejection, rebalance, NIST-driven selection or claim of secrecy.
"""

from __future__ import annotations

import argparse
import gc
import json
from bisect import bisect_left, bisect_right, insort
from collections import deque
from pathlib import Path

import numpy as np

from research import linear_key_scheme as core

OUT = Path("output_data/notes/keygen_simple_innovation_20261105")
CASES = ["kazan_v4_00", "kazan_long_00"] + [
    f"july_kaiser9_{k:02}" for k in range(4)
]


def empirical_quantile(x, window):
    """Strictly past, exactly last window finite values; ties use midrank."""
    out = np.full(len(x), np.nan)
    history = []
    queue = deque()
    for i, value in enumerate(x):
        if not np.isfinite(value):
            continue
        if len(history) >= 64:
            out[i] = (
                bisect_left(history, value) + bisect_right(history, value)
            ) / (2 * len(history))
        insort(history, float(value))
        queue.append(float(value))
        if len(queue) > window:
            history.pop(bisect_left(history, queue.popleft()))
    return out


def records(raw, fit_end):
    result = []
    for j in range(raw.status.shape[1]):
        valid = np.isfinite(raw.x["alice"][:, j]) & np.isfinite(
            raw.x["bob"][:, j]
        )
        step = np.flatnonzero(valid)
        if len(step) < 70:
            continue
        values = {p: raw.dtau[p][step, j] * 1e9 for p in ("alice", "bob")}
        fit = step[5:] <= fit_end
        coef = np.zeros(5)
        intercept = 0.0
        ready = fit.sum() >= 64
        if ready:
            xx = np.concatenate(
                [
                    np.lib.stride_tricks.sliding_window_view(a, 6)[:, :5][
                        :, ::-1
                    ][fit]
                    for a in values.values()
                ]
            )
            yy = np.concatenate([a[5:][fit] for a in values.values()])
            mean_x, mean_y = xx.mean(axis=0), yy.mean()
            xx = xx - mean_x
            yy = yy - mean_y
            gram = xx.T @ xx
            # Fixed regularisation, no tuning on future or July outcomes.
            coef = np.linalg.solve(
                gram + np.eye(5) * max(np.trace(gram) * 1e-3 / 5, 1e-12),
                xx.T @ yy,
            )
            intercept = float(mean_y - mean_x @ coef)
        features = {}
        for kind in ("direct", "difference", "ar5"):
            if kind == "ar5" and not ready:
                continue
            features[kind] = {}
            for p, a in values.items():
                if kind == "direct":
                    y = a[5:]
                elif kind == "difference":
                    y = a[5:] - a[4:-1]
                else:
                    xx = np.lib.stride_tricks.sliding_window_view(a, 6)[:, :5][
                        :, ::-1
                    ]
                    y = a[5:] - intercept - xx @ coef
                features[kind][p] = y
        result.append(
            (
                j,
                step[5:],
                features,
                {
                    "profile": j,
                    "coefficients": coef.tolist(),
                    "intercept": intercept,
                    "fit_rows": int(fit.sum()),
                    "ready": bool(ready),
                },
            )
        )
    return result


def sequence(rec, kind, window, bits):
    arrays = []
    for j, step, features, model in rec:
        if kind not in features:
            continue
        u = {
            p: empirical_quantile(a, window) for p, a in features[kind].items()
        }
        finite = np.isfinite(u["alice"]) & np.isfinite(u["bob"])
        gray = {}
        for p, a in u.items():
            words = np.minimum(
                np.floor(np.nan_to_num(a) * (1 << bits)).astype(int),
                (1 << bits) - 1,
            )
            gray[p] = words ^ (words >> 1)
        for plane in range(bits):
            shift = bits - 1 - plane
            arrays.append(
                np.column_stack(
                    [
                        step[finite],
                        np.full(finite.sum(), j),
                        np.zeros(finite.sum(), int),
                        np.full(finite.sum(), plane),
                        (gray["alice"][finite] >> shift) & 1,
                        (gray["bob"][finite] >> shift) & 1,
                    ]
                )
            )
    if not arrays:
        return np.empty((0, 6), int)
    data = np.concatenate(arrays)
    return data[np.lexsort((data[:, 3], data[:, 1], data[:, 0]))]


def pilot_score(data):
    if len(data) < 256:
        return {
            "eligible": False,
            "bits": len(data),
            "BDR": None,
            "dependence": None,
            "violation": 1e6,
        }
    bdr = float(np.mean(data[:, 4] != data[:, 5]))
    dependence = 0.0
    for col in (4, 5):
        bits = data[:, col]
        dependence = max(
            dependence,
            abs(2 * float(bits.mean()) - 1),
            max(
                (abs(a) for a in core.acf(bits, 16) if a is not None),
                default=1,
            ),
        )
        # Check planes too: interleaving can conceal their dependence.
        for j, plane in np.unique(data[:, [1, 3]], axis=0):
            a = bits[(data[:, 1] == j) & (data[:, 3] == plane)]
            if len(a) >= 128:
                dependence = max(
                    dependence,
                    abs(2 * float(a.mean()) - 1),
                    max(
                        (abs(v) for v in core.acf(a, 8) if v is not None),
                        default=1,
                    ),
                )
    violation = max(bdr / 0.02, dependence / 0.10)
    return {
        "eligible": violation <= 1,
        "bits": len(data),
        "BDR": bdr,
        "dependence": dependence,
        "violation": violation,
    }


def worker(case):
    manifest = json.loads(core.PARENT.read_text())
    raw = core.read(manifest["caches"][case]["decor"]["path"])
    raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
    baseline = core.read(core.BASE / f"{case}_calib.pkl")
    pivot = baseline.decision["step"] if baseline.decision else -1
    fit_end = pivot // 2
    print(
        "LOAD",
        case,
        raw.status.shape,
        "fit_end",
        fit_end,
        "pilot_end",
        pivot,
        flush=True,
    )
    rec = records(raw, fit_end)
    del raw
    gc.collect()
    choices = []
    # Fixed search space and scoring; NIST is never queried by selection.
    for kind in ("direct", "difference", "ar5"):
        for window in (128, 512):
            for bits in (1, 2):
                data = sequence(rec, kind, window, bits)
                pilot = data[(data[:, 0] > fit_end) & (data[:, 0] <= pivot)]
                score = pilot_score(pilot)
                choices.append(
                    dict(
                        kind=kind, window=window, bits_per_value=bits, **score
                    )
                )
    feasible = [r for r in choices if r["eligible"]]
    selected = (
        min(feasible, key=lambda r: (-r["bits"], r["dependence"], r["BDR"]))
        if feasible
        else min(choices, key=lambda r: r["violation"])
    )
    core.write(
        OUT / f"{case}_pilot.json",
        {
            "case": case,
            "fit_end": fit_end,
            "pivot": pivot,
            "feasible": bool(feasible),
            "selected": selected,
            "candidates": choices,
            "models": [r[3] for r in rec],
        },
    )
    # Three fixed controls are reported irrespective of how their future looks.
    specs = [
        (
            "selected",
            selected["kind"],
            selected["window"],
            selected["bits_per_value"],
        )
    ]
    specs += [
        (kind + "_fixed", kind, 128, 2)
        for kind in ("direct", "difference", "ar5")
    ]
    for name, kind, window, bits in specs:
        data = sequence(rec, kind, window, bits)
        data = data[data[:, 0] > pivot]
        result = core.evaluate(data, [])
        result.update(
            case=case,
            kind=kind,
            window=window,
            bits_per_value=bits,
            pilot_feasible=bool(feasible),
            purpose=name,
            pilot_end=pivot,
            raw_bits_not_final_key=True,
        )
        stem = OUT / f"{case}_{name}"
        core.write(stem.with_suffix(".json"), result)
        np.savez_compressed(
            stem.with_suffix(".npz"),
            alice=data[:, 4].astype(np.uint8),
            bob=data[:, 5].astype(np.uint8),
            provenance=data[:, :4],
        )
        a = result["party"]["alice"]
        print(
            "RESULT",
            case,
            name,
            kind,
            window,
            bits,
            result["bits"],
            result["BDR"],
            a["max_abs_ACF"],
            [
                (t["name"], t["p_value"])
                for t in a["nist"]
                if t["passed"] is False
            ],
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for case in [args.case] if args.case else CASES:
        worker(case)


if __name__ == "__main__":
    main()
