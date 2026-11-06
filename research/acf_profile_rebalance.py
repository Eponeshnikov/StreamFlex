"""Pooled growing CDF with a separate public rebalance CDF per Δτ component."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from research import linear_key_scheme as core
from research.acf_growing_full_budget import audit
from research.acf_growing_linear9 import residual
from research.simple_innovation_scheme import CASES

OUT = Path("output_data/notes/keygen_acf_profile_rebalance_20261106")


def calibrate(raw, bits, profile_bits=None, check_bits=2, cdf="per_profile"):
    if cdf not in {"pooled", "per_profile"}:
        raise ValueError("unsupported CDF grouping")
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
        check_bits=0,
        check_extra=True,
        rebalance=False,
        detrend="off",
    )
    base = C.calibrate(work, cfg)
    if profile_bits is None:
        widths = np.full(len(base.step), bits, np.int64)
    else:
        widths = np.array([profile_bits[int(p)] for p in base.profile], np.int64)
    if np.any(widths < 1):
        raise ValueError("profile widths must be positive")
    result_cfg = replace(cfg, cdf=cdf, check_bits=check_bits, rebalance=True)
    groups = base.profile if cdf == "per_profile" else np.zeros_like(base.profile)
    rebalanced = C._rebalanced_u(
        base.u, base.step, groups, widths + check_bits, check_bits, result_cfg
    )
    fine_bins, guard = {}, {}
    for p in base.u:
        fine_bins[p], guard[p] = C.quantize(
            rebalanced[p], widths + check_bits, 0.0
        )
    bins = {p: value >> check_bits for p, value in fine_bins.items()}
    low = (1 << check_bits) - 1
    failed = ((fine_bins["alice"] & low) != (fine_bins["bob"] & low))
    erased = guard["alice"] | guard["bob"] | failed
    bit_text = {
        p: [
            format(int(word), f"0{int(width)}b")
            for word, width in zip(value[~erased], widths[~erased], strict=True)
        ]
        for p, value in bins.items()
    }
    result = replace(
        base,
        config=result_cfg,
        bins=bins,
        erased=erased,
        bits=bit_text,
        value_bits=widths,
        check_fail=failed,
    )
    np.testing.assert_array_equal(raw.status, original_status)
    return result


def correlation_profile_bits(raw):
    """Patwari MAQ widths from full-record ρ; oracle diagnostic, not online."""
    z = residual(raw.x)
    result = {}
    for profile in range(raw.status.shape[1]):
        a, b = z["alice"][:, profile], z["bob"][:, profile]
        rows = (
            np.isfinite(a)
            & np.isfinite(b)
            & (raw.status[:, profile] == C.ADMITTED)
        )
        rho = np.corrcoef(a[rows], b[rows])[0, 1] if rows.sum() >= 20 else 0.0
        bits = 1 if rho < 0.98 else 2 if rho < 0.993 else 3 if rho < 0.998 else 4
        result[profile] = bits
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("bits", type=int, choices=(2, 3, 4))
    parser.add_argument("--rho-maq", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(core.PARENT.read_text())
    raw = core.read(manifest["caches"][args.case]["decor"]["path"])
    raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
    profile_bits = correlation_profile_bits(raw) if args.rho_maq else None
    c = calibrate(raw, args.bits, profile_bits)
    tag = "rho-maq-oracle" if args.rho_maq else f"profile-rebalance-b{args.bits}"
    result = audit(args.case, args.bits, calibrated=(raw, c), tag=tag)
    result["initial_cdf"] = "pooled"
    result["rebalance_cdf"] = "per_profile"
    if profile_bits is not None:
        result["profile_bits"] = profile_bits
        result["width_selection"] = "full-record Pearson rho; oracle diagnostic, not an online protocol"
    OUT.mkdir(exist_ok=True)
    suffix = "_rho_maq" if args.rho_maq else ""
    core.write(OUT / f"{args.case}_b{args.bits}{suffix}.json", result)
    print(
        "PROFILE_REBALANCE",
        args.case,
        args.bits,
        result["raw_bits"],
        result["BDR"],
        result["NIST_failures"],
        result["final_bits"],
        result["final_bps"],
        profile_bits,
        flush=True,
    )


if __name__ == "__main__":
    main()
