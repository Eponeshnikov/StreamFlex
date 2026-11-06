"""Pilot-only selection of component widths and LDPC block size."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import ldpc as L
from research import linear_key_scheme as core
from research.acf_growing_full_budget import audit
from research.acf_growing_linear9 import residual
from research.acf_profile_rebalance import calibrate
from research.simple_innovation_scheme import CASES

OUT = Path("output_data/notes/keygen_acf_causal_selection_20261106")
PILOT_VALUES = 1000
MIN_PROFILE_VALUES = 32
PILOT_BDR_4BIT = 0.03
BLOCK_BDR_THRESHOLDS = (0.01, 0.03)


def select(raw, cdf="per_profile"):
    work = residual(raw.x)
    admitted = (
        (raw.status == C.ADMITTED)
        & np.isfinite(work["alice"])
        & np.isfinite(work["bob"])
    )
    steps, profiles = np.nonzero(admitted)
    if len(steps) < PILOT_VALUES:
        raise ValueError("not enough admitted values for the pilot")
    steps, profiles = steps[:PILOT_VALUES], profiles[:PILOT_VALUES]
    fine = {}
    for party in ("alice", "bob"):
        values = work[party][steps, profiles]
        ordered = np.sort(values)
        u = (
            np.searchsorted(ordered, values, side="left")
            + np.searchsorted(ordered, values, side="right")
        ) / (2 * len(ordered))
        fine[party] = np.floor(u * 64).clip(0, 63).astype(np.int64)
    keep = (fine["alice"] & 3) == (fine["bob"] & 3)
    a = C.gray(fine["alice"][keep] >> 2)
    b = C.gray(fine["bob"][keep] >> 2)
    pilot_bdr = float(np.bitwise_count(a ^ b).sum() / (4 * keep.sum()))
    short_record = len(raw.steps) < 5000
    default_bits = 4 if pilot_bdr <= PILOT_BDR_4BIT and not short_record else 3
    counts = np.bincount(profiles, minlength=raw.status.shape[1])
    profile_bits = {
        int(p): int(
            default_bits
            if default_bits == 3
            or cdf == "pooled"
            or counts[p] >= MIN_PROFILE_VALUES
            else 3
        )
        for p in range(raw.status.shape[1])
    }
    if short_record or pilot_bdr >= BLOCK_BDR_THRESHOLDS[1]:
        block_bits = 768
    elif pilot_bdr < BLOCK_BDR_THRESHOLDS[0]:
        block_bits = 4096
    else:
        block_bits = 5120
    return {
        "profile_bits": profile_bits,
        "block_bits": block_bits,
        "pilot_bdr_4bit": pilot_bdr,
        "pilot_kept_values": int(keep.sum()),
        "pilot_profile_counts": {str(p): int(v) for p, v in enumerate(counts)},
        "pilot_values": PILOT_VALUES,
        "planned_steps": len(raw.steps),
        "short_record": short_record,
        "minimum_profile_values_for_4bit": MIN_PROFILE_VALUES,
        "pilot_bdr_4bit_threshold": PILOT_BDR_4BIT,
        "block_bdr_thresholds": BLOCK_BDR_THRESHOLDS,
        "selection_input": "planned record length and first 1000 admitted residuals; pilot excluded from key generation",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--cdf", choices=("pooled", "per_profile"), default="per_profile")
    args = parser.parse_args()
    manifest = json.loads(core.PARENT.read_text())
    raw = core.read(manifest["caches"][args.case]["decor"]["path"])
    raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
    selection = select(raw, args.cdf)
    c = calibrate(raw, 4, selection["profile_bits"], cdf=args.cdf)
    cfg = L.LdpcConfig(
        block_bits=selection["block_bits"],
        soft=True,
        public_reliability=False,
    )
    decision_bits = 2 * len(selection["profile_bits"]) + 2
    result = audit(
        args.case,
        4,
        calibrated=(raw, c),
        tag=f"pilot-select-{args.cdf}",
        ldpc_config=cfg,
        public_decision_bits=decision_bits,
    )
    result.update(
        selection=selection,
        ldpc_config=asdict(cfg),
        public_decision_bits=decision_bits,
        selection_note="causal data access; thresholds require validation on unseen records",
    )
    OUT.mkdir(exist_ok=True)
    core.write(OUT / f"{args.case}_{args.cdf}.json", result)
    print(
        "PILOT_SELECT",
        args.case,
        selection,
        result.get("raw_bits"),
        result.get("BDR"),
        result.get("NIST_failures"),
        result.get("N"),
        result.get("ldpc_leak"),
        result.get("final_bits"),
        result.get("final_bps"),
        flush=True,
    )


if __name__ == "__main__":
    main()
