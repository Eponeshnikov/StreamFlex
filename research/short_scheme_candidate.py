"""Common optional H4 contrast bank plus a single scalar quantizer.

Pilot memory decision is inherited unchanged, not keyed by recording name.
ON: two fixed orthonormal trend-annihilating contrasts per disjoint block4;
zero-threshold sign bits, local magnitude guards, no shared pilot values.
OFF: existing scalar quantizer, with rebalance removed (check0/1 controls).
Research prototype. No claim of final secret length or Gaussianity on data.
"""

import gc
import json
from dataclasses import replace

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import growing_policy as G
from research import linear_key_scheme as core
from research.simple_innovation_scheme import CASES

OUT = core.Path("output_data/notes/keygen_short_candidate_20261105")
H = np.array([[1, -1, -1, 1], [-1, 3, -3, 1]], float) / np.sqrt([[4], [20]])


def from_calibration(c):
    widths = c.widths[c.kept]
    step = np.repeat(c.step[c.kept], widths)
    profile = np.repeat(c.profile[c.kept], widths)
    plane = (
        np.concatenate([np.arange(w) for w in widths])
        if len(widths)
        else np.zeros(0, int)
    )
    bits = {
        p: np.frombuffer("".join(c.bits[p]).encode(), np.uint8) - 48
        for p in ("alice", "bob")
    }
    assert len(step) == len(bits["alice"]) == len(bits["bob"])
    return np.column_stack(
        [
            step,
            profile,
            np.zeros(len(step), int),
            plane,
            bits["alice"],
            bits["bob"],
        ]
    )


def contrasts(raw, pivot, factor):
    arrays = []
    models = []
    for j in range(raw.status.shape[1]):
        ix = np.flatnonzero(
            np.isfinite(raw.x["alice"][:, j]) & np.isfinite(raw.x["bob"][:, j])
        )
        n = len(ix) // 4
        if n < 5:
            continue
        steps = ix[: n * 4].reshape(n, 4)[:, -1]
        y = {
            p: raw.dtau[p][ix[: n * 4], j].reshape(n, 4) * 1e9 @ H.T
            for p in ("alice", "bob")
        }
        pilot = steps <= pivot
        if pilot.sum() < 16:
            continue
        thresholds = {
            p: factor * np.median(np.abs(a[pilot]), axis=0)
            for p, a in y.items()
        }
        models.append(
            {
                "profile": j,
                "pilot_blocks": int(pilot.sum()),
                "local_threshold_ns": {
                    p: a.tolist() for p, a in thresholds.items()
                },
            }
        )
        for k in (0, 1):
            keep = (
                (steps > pivot)
                & (np.abs(y["alice"][:, k]) >= thresholds["alice"][k])
                & (np.abs(y["bob"][:, k]) >= thresholds["bob"][k])
            )
            arrays.append(
                np.column_stack(
                    [
                        steps[keep],
                        np.full(keep.sum(), j),
                        np.full(keep.sum(), k),
                        np.zeros(keep.sum(), int),
                        y["alice"][keep, k] > 0,
                        y["bob"][keep, k] > 0,
                    ]
                )
            )
    data = np.concatenate(arrays) if arrays else np.empty((0, 6), int)
    return data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))], models


def save(case, name, data, **info):
    r = core.evaluate(data, info.pop("models", []))
    r.update(case=case, name=name, **info)
    stem = OUT / f"{case}_{name}"
    core.write(core.Path(str(stem) + ".json"), r)
    np.savez_compressed(
        core.Path(str(stem) + ".npz"),
        alice=data[:, 4].astype(np.uint8),
        bob=data[:, 5].astype(np.uint8),
        provenance=data[:, :4],
    )
    print(
        "RESULT",
        case,
        name,
        r["bits"],
        r["BDR"],
        r["party"]["alice"]["max_abs_ACF"],
        [
            (
                p,
                [
                    (t["name"], t["p_value"])
                    for t in r["party"][p]["nist"]
                    if t["passed"] is False
                ],
            )
            for p in ("alice", "bob")
        ],
        flush=True,
    )


def run_candidate(raw):
    """One scene-independent prototype: inherited pilot decision, fixed .75 guard.

    Retain the existing low-memory branch because removing its rebalance was
    measured to degrade July bit patterns. This is not a cryptographic proof.
    """
    cfg = G.baseline_config()
    decision = G.auto_decision(raw, cfg)
    if decision and decision["on"]:
        data, models = contrasts(raw, decision["step"], 0.75)
        return data, {
            "decision": decision,
            "branch": "H4",
            "guard_factor": 0.75,
            "models": models,
            "H": H.tolist(),
            "no_shared_pilot_values": True,
        }
    calibrated = C.calibrate(raw, cfg)
    return from_calibration(calibrated), {
        "decision": decision,
        "branch": "existing_scalar_quantizer",
        "models": [],
    }


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads(core.PARENT.read_text())
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        baseline = core.read(core.BASE / f"{case}_calib.pkl")
        decision = G.auto_decision(raw, G.baseline_config())
        assert decision == baseline.decision
        save(case, "baseline", from_calibration(baseline), decision=decision)
        if decision and decision["on"]:
            for factor in (0.25, 0.5, 0.75):
                data, models = contrasts(raw, decision["step"], factor)
                save(
                    case,
                    f"H4_local_guard{factor}",
                    data,
                    models=models,
                    decision=decision,
                    guard_factor=factor,
                    H=H.tolist(),
                    no_shared_pilot_values=True,
                    raw_bits_not_final_key=True,
                )
        else:
            for check in (0, 1):
                cfg = replace(
                    G.baseline_config(), rebalance=False, check_bits=check
                )
                c = C.calibrate(raw, cfg)
                save(
                    case,
                    f"one_quantizer_check{check}",
                    from_calibration(c),
                    decision=c.decision,
                    rebalance=False,
                    check_bits=check,
                    raw_bits_not_final_key=True,
                )
        policy, info = run_candidate(raw)
        if not (decision and decision["on"]):
            np.testing.assert_array_equal(policy, from_calibration(baseline))
        save(case, "policy", policy, **info, raw_bits_not_final_key=True)
        del raw
        gc.collect()


if __name__ == "__main__":
    main()
