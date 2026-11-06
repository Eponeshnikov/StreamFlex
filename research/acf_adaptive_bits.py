"""Causal public-check adaptation of the number of key bits per log Δτ."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from research import linear_key_scheme as core
from research.acf_growing_full_budget import audit, calibrate
from research.acf_growing_linear9 import residual
from research.simple_innovation_scheme import CASES

OUT = Path("output_data/notes/keygen_acf_adaptive_bits_20261106")
MIN_HISTORY = 32
UPGRADE = {2: 0.25, 3: 0.22}
DOWNGRADE = {3: 0.40, 4: 0.45}


def profile_metrics(c):
    gray = {p: C.gray(c.bins[p]) for p in ("alice", "bob")}
    rows_out = []
    for profile in np.unique(c.profile):
        selected = c.kept & (c.profile == profile)
        widths = c.widths[selected]
        bits = int(widths.sum())
        a, b = gray["alice"][selected], gray["bob"][selected]
        plane_bdr = []
        for plane in range(c.config.bits):
            mask = 1 << (c.config.bits - plane - 1)
            plane_bdr.append(
                float(np.count_nonzero((a & mask) != (b & mask)) / len(a))
                if len(a)
                else None
            )
        rows_out.append(
            {
                "profile": int(profile),
                "candidates": int(np.count_nonzero(c.profile == profile)),
                "kept": int(selected.sum()),
                "raw_bits": bits,
                "BDR": float(np.bitwise_count(a ^ b).sum() / bits) if bits else None,
                "plane_BDR_MSB_first": plane_bdr,
                "ones": float(np.bitwise_count(a).sum() / bits) if bits else None,
            }
        )
    return rows_out


def adaptive_calibrate(raw):
    original_status = raw.status.copy()
    work = replace(raw, x=residual(raw.x))
    cfg = C.CalibConfig(
        algorithm="standard",
        n_target=1000,
        cdf="pooled",
        bits=4,
        bits_rule="fixed",
        update="growing",
        guard=0.0,
        check_bits=0,
        check_extra=True,
        rebalance=False,
        detrend="off",
    )
    base = C.calibrate(work, cfg)
    n = len(base.step)
    widths = np.zeros(n, np.int64)
    bins = {p: np.zeros(n, np.int64) for p in base.u}
    failed = np.zeros(n, bool)
    rebalanced = {p: a.copy() for p, a in base.u.items()}
    history = {p: [] for p in base.u}
    pending = []
    states = {
        int(profile): {"bits": 2}
        for profile in np.unique(base.profile)
    }
    counts = np.zeros((raw.status.shape[1], 5), np.int64)
    errors = np.zeros((raw.status.shape[1], 5), np.int64)
    events = []
    previous = -1

    for i in range(n):
        step = int(base.step[i])
        if previous >= 0 and step != previous:
            for p, hist in history.items():
                hist.extend(base.u[p][k] for k in pending)
            pending = []
            for profile, state in states.items():
                bits = state["bits"]
                count = int(counts[profile, bits])
                if count < MIN_HISTORY:
                    continue
                q = errors[profile, bits] / count
                new_bits = bits
                if bits < 4 and q <= UPGRADE[bits]:
                    new_bits = bits + 1
                elif bits > 2 and q >= DOWNGRADE[bits]:
                    new_bits = bits - 1
                if new_bits != bits:
                    events.append(
                        {
                            "step": step,
                            "profile": profile,
                            "from": bits,
                            "to": new_bits,
                            "past_values": count,
                            "past_check_disagreement": q,
                            "payload_bits": 0,
                        }
                    )
                    state["bits"] = new_bits
                    counts[profile, new_bits] = 0
                    errors[profile, new_bits] = 0
        previous = step

        profile = int(base.profile[i])
        bits = states[profile]["bits"]
        widths[i] = bits
        if len(history["alice"]) >= cfg.rebalance_min:
            for p, hist in history.items():
                ref = np.asarray(hist)
                value = base.u[p][i]
                rebalanced[p][i] = (
                    np.count_nonzero(ref < value)
                    + np.count_nonzero(ref <= value)
                ) / (2 * ref.size)
        fine = bits + 2
        low = {}
        for p in base.u:
            word = C.quantize(rebalanced[p][i : i + 1], fine, 0.0)[0][0]
            low[p] = int(word & 3)
            bins[p][i] = word >> 2
        failed[i] = low["alice"] != low["bob"]
        counts[profile, bits] += 1
        errors[profile, bits] += failed[i]
        if not failed[i]:
            pending.append(i)

    result_cfg = replace(cfg, check_bits=2, rebalance=True)
    bits_text = {
        p: [
            format(int(word), f"0{int(width)}b")
            for word, width in zip(value[~failed], widths[~failed], strict=True)
        ]
        for p, value in bins.items()
    }
    result = replace(
        base,
        config=result_cfg,
        bins=bins,
        erased=failed,
        bits=bits_text,
        value_bits=widths,
        check_fail=failed,
        cdf_decisions=events,
    )
    np.testing.assert_array_equal(raw.status, original_status)
    return result


def main():
    manifest = json.loads(core.PARENT.read_text())
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        c = adaptive_calibrate(raw)
        result = audit(
            case,
            4,
            calibrated=(raw, c),
            tag="public-check-adaptive-v1",
        )
        result["adaptive_rule"] = {
            "minimum_past_values_per_profile_and_width": MIN_HISTORY,
            "upgrade_max_check_disagreement": UPGRADE,
            "downgrade_min_check_disagreement": DOWNGRADE,
            "transitions": len(c.cdf_decisions),
            "kept_width_histogram": {
                str(bits): int(np.count_nonzero(c.widths[c.kept] == bits))
                for bits in (2, 3, 4)
            },
        }
        OUT.mkdir(exist_ok=True)
        core.write(OUT / f"{case}_adaptive.json", result)
        print(
            "ADAPTIVE",
            case,
            result["raw_bits"],
            result["BDR"],
            result["N"],
            result["final_bits"],
            result["final_bps"],
            result["adaptive_rule"],
            flush=True,
        )


if __name__ == "__main__":
    main()
