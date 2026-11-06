"""Full public-transcript budget for the causal linear9 multi-bit prototype.

The joint predictive envelope includes public checks, deterministic selection,
frame metadata and admission messages. It replaces the standalone frame/check
estimates. It is an empirical diagnostic, not a certified secrecy bound.
"""

import argparse
import gc
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import joint_leakage as J
from pages.offline_keygen_v2 import ldpc as L
from pages.offline_keygen_v2 import privacy as P
from research import linear_key_scheme as core
from research.acf_growing_linear9 import residual
from research.short_scheme_candidate import from_calibration
from research.simple_innovation_scheme import CASES

OUT = Path("output_data/notes/keygen_acf_growing_full_budget_20261106")


def inverse_gray(value, bits):
    value = np.asarray(value, dtype=np.int64)
    out = value.copy()
    for shift in range(1, bits):
        out ^= value >> shift
    return out


def inverse_gray_variable(value, widths):
    value = np.asarray(value, dtype=np.int64)
    out = np.zeros_like(value)
    for bits in np.unique(widths):
        rows = widths == bits
        out[rows] = inverse_gray(value[rows], int(bits))
    return out


def calibrate(raw, bits, check_bits=2):
    original_status = raw.status.copy()
    work = replace(raw, x=residual(raw.x))
    cfg = C.CalibConfig(
        algorithm="standard",
        n_target=1000,
        cdf="pooled",
        bits=bits,
        bits_rule="fixed",
        update="growing",
        guard=0.0,
        check_bits=check_bits,
        check_extra=True,
        rebalance=True,
        detrend="off",
    )
    c = C.calibrate(work, cfg)
    c = replace(
        c,
        bits={
            p: [format(int(word), f"0{bits}b") for word in v[c.kept]]
            for p, v in c.bins.items()
        },
    )
    np.testing.assert_array_equal(raw.status, original_status)
    return c


def audit(
    case, bits, check_bits=2, calibrated=None, tag=None,
    inverse_target=True, transcript=None, ldpc_config=None,
    public_decision_bits=0,
):
    manifest = json.loads(core.PARENT.read_text())
    if calibrated is None:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        c = calibrate(raw, bits, check_bits)
    else:
        raw, c = calibrated
    data = from_calibration(c)
    data = data[np.argsort(data[:, 3], kind="stable")]
    randomness = core.evaluate(data, [])
    target_c = replace(
        c,
        bins={p: inverse_gray_variable(v, c.widths) for p, v in c.bins.items()},
    ) if inverse_target else c
    rec = L.run(
        target_c,
        ldpc_config or L.LdpcConfig(
            block_bits=256, soft=False, public_reliability=False
        ),
        session_id=f"acf-growing-linear9-{tag or f'b{bits}-check{check_bits}'}-{case}",
        times=[s.elapsed_s for s in raw.steps],
    )
    alice, bob = rec.reconciled(), rec.reconciled("bob")
    np.testing.assert_array_equal(alice, bob)
    original = from_calibration(c)[rec.reconciled_index(), 4].astype(np.uint8)
    np.testing.assert_array_equal(alice, original)
    check = J.check_transcript(c) if transcript is None else transcript
    empirical_check = J.empirical_check_information(target_c, check)
    result = {
        "case": case,
        "key_bits_per_interval": bits,
        "rule": tag or "fixed",
        "mean_key_bits_per_kept_value": float(c.widths[c.kept].mean()),
        "config": asdict(c.config),
        "duration_s": rec.duration_s,
        "raw_bits": rec.stream.size,
        "raw_bps": rec.stream.size / rec.duration_s,
        "BDR": randomness["BDR"],
        "NIST_failures": {
            p: [
                {"name": t["name"], "p_value": t["p_value"]}
                for t in randomness["party"][p]["nist"]
                if t["passed"] is False
            ]
            for p in ("alice", "bob")
        },
        "N": len(alice),
        "confirmed_blocks": len(rec.confirmed),
        "blocks": len(rec.blocks),
        "ldpc_leak": rec.leakage,
        "transcript": check.alphabet(),
        "empirical_check": empirical_check,
        "certified_positive_length": False,
    }
    stream = L.stream(target_c)
    index = rec.reconciled_index()
    present = np.zeros(stream.size, bool)
    present[index] = True
    value_row = np.repeat(np.flatnonzero(c.kept), c.widths[c.kept])
    counts = np.bincount(value_row[present], minlength=len(c.step))
    included = c.kept & (counts == c.widths)
    target_n = len(index)
    scored_n = int(c.widths[included].sum())
    partial = target_n - scored_n
    result.update(partial_word_bits=partial, included_words=int(included.sum()))
    if included.sum() < 300:
        result.update(status="insufficient_words", final_bits=None, final_bps=None)
        return result

    print("VIEW", case, bits, flush=True)
    view = core.read(manifest["caches"][case]["view"]["path"])
    metadata, check_features = J.public_features(
        view, c, raw, check, bool(manifest["mpc"]["index_in_frame"])
    )
    del view, raw
    gc.collect()
    print("FIT", case, bits, int(included.sum()), flush=True)
    estimate = J.predict_joint(target_c, metadata, check_features, included)
    result["model"] = estimate
    if estimate["status"] != "estimated":
        result.update(status=estimate["status"], final_bits=None, final_bps=None)
        return result
    gains = estimate["uniform_reference_predictive_disclosure_bits"]
    empirical = max(gains.values()) + partial
    calibration = int(getattr(c, "cdf_control_bits", 0))
    decision = public_decision_bits + (2 if c.decision is not None else 0)
    ell = max(
        0,
        math.floor(
            target_n
            - float(P.delta_h(target_n, 1e-6))
            - rec.leakage
            - calibration
            - empirical
            - P.lhl(1e-6)
        )
        - decision,
    )
    seed = P.uniform_bits(target_n + ell - 1) if ell else np.zeros(0, np.uint8)
    ka = P.toeplitz(alice, ell, seed)
    kb = P.toeplitz(bob, ell, seed)
    np.testing.assert_array_equal(ka, kb)
    result.update(
        status="estimated",
        empirical_joint_envelope_bits=empirical,
        empirical_joint_envelope_per_reconciled_bit=empirical / target_n,
        calibration_leak=calibration,
        decision_leak=decision,
        delta_h=float(P.delta_h(target_n, 1e-6)),
        lhl=float(P.lhl(1e-6)),
        final_bits=ell,
        final_bps=ell / rec.duration_s,
        final_keys_equal=True,
        note="joint envelope replaces standalone frame/check estimates; no certified min-entropy bound",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("bits", type=int, choices=(2, 3, 4))
    parser.add_argument("--check-bits", type=int, choices=(0, 2), default=2)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    result = audit(args.case, args.bits, args.check_bits)
    suffix = "" if args.check_bits == 2 else f"_check{args.check_bits}"
    core.write(OUT / f"{args.case}_b{args.bits}{suffix}.json", result)
    print(
        "RESULT",
        args.case,
        args.bits,
        args.check_bits,
        result.get("N"),
        result.get("empirical_joint_envelope_bits"),
        result.get("final_bits"),
        result.get("final_bps"),
        flush=True,
    )


if __name__ == "__main__":
    main()
