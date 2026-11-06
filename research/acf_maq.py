"""Patwari MAQ: per-component width and public rotated Gray codebook."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import joint_leakage as J
from research import linear_key_scheme as core
from research.acf_growing_full_budget import audit
from research.acf_growing_linear9 import residual
from research.acf_profile_rebalance import correlation_profile_bits
from research.simple_innovation_scheme import CASES

OUT = Path("output_data/notes/keygen_acf_maq_20261106")


def maq_word(fine_bin, bits, flag):
    levels = 1 << (bits + 2)
    group = fine_bin // 4 if flag else ((fine_bin + 2) % levels) // 4
    return int(C.gray(np.array([group]))[0])


def calibrate(raw, profile_bits):
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
    widths = np.array([profile_bits[int(p)] for p in base.profile], np.int64)
    bins = {p: np.zeros(len(base.step), np.int64) for p in base.u}
    flags = np.zeros(len(base.step), np.int64)
    history = {
        p: [[] for _ in range(raw.status.shape[1])]
        for p in base.u
    }
    pending = [[] for _ in range(raw.status.shape[1])]
    previous = -1
    for i in range(len(base.step)):
        step = int(base.step[i])
        profile = int(base.profile[i])
        if previous >= 0 and step != previous:
            for p in base.u:
                for j in range(raw.status.shape[1]):
                    history[p][j].extend(base.u[p][k] for k in pending[j])
                    pending[j] = []
        previous = step
        bits = int(widths[i])
        fine = bits + 2
        words = {}
        for p in base.u:
            value = base.u[p][i]
            hist = history[p][profile]
            if len(hist) >= cfg.rebalance_min:
                ref = np.asarray(hist)
                value = (
                    np.count_nonzero(ref < value)
                    + np.count_nonzero(ref <= value)
                ) / (2 * ref.size)
            words[p] = int(C.quantize(np.array([value]), fine, 0.0)[0][0])
        flag = int((words["alice"] + 1) % 4 >= 2)
        flags[i] = flag
        for p in base.u:
            bins[p][i] = maq_word(words[p], bits, flag)
        pending[profile].append(i)
    result_cfg = replace(cfg, cdf="per_profile")
    bit_text = {
        p: [
            format(int(word), f"0{int(width)}b")
            for word, width in zip(value, widths, strict=True)
        ]
        for p, value in bins.items()
    }
    result = replace(
        base,
        config=result_cfg,
        bins=bins,
        erased=np.zeros(len(base.step), bool),
        bits=bit_text,
        value_bits=widths,
        check_fail=np.zeros(len(base.step), bool),
    )
    transcript = J.CheckTranscript(
        flags,
        flags.copy(),
        np.ones(len(flags), bool),
        np.ones(len(flags), bool),
        1,
    )
    np.testing.assert_array_equal(raw.status, original_status)
    return result, transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--fixed", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    manifest = json.loads(core.PARENT.read_text())
    raw = core.read(manifest["caches"][args.case]["decor"]["path"])
    raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
    if args.fixed:
        profile_bits = {p: args.fixed for p in range(raw.status.shape[1])}
        tag = f"maq-fixed{args.fixed}"
        suffix = f"_fixed{args.fixed}"
    else:
        profile_bits = correlation_profile_bits(raw)
        tag = "maq-rho-oracle"
        suffix = "_rho_maq"
    c, transcript = calibrate(raw, profile_bits)
    result = audit(
        args.case,
        4,
        calibrated=(raw, c),
        tag=tag,
        transcript=transcript,
    )
    result["profile_bits"] = profile_bits
    result["initial_cdf"] = "pooled"
    result["rebalance_cdf"] = "per_profile"
    result["public_codebook_flags"] = len(c.step)
    if not args.fixed:
        result["width_selection"] = "full-record Pearson rho; oracle diagnostic, not an online protocol"
    OUT.mkdir(exist_ok=True)
    core.write(OUT / f"{args.case}{suffix}.json", result)
    print(
        "MAQ",
        args.case,
        result["raw_bits"],
        result["BDR"],
        result["NIST_failures"],
        result["N"],
        result["ldpc_leak"],
        result["empirical_joint_envelope_bits"],
        result["final_bits"],
        result["final_bps"],
        flush=True,
    )


if __name__ == "__main__":
    main()
