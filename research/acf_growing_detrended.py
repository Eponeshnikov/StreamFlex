"""Standard ACF admission, causal detrending and one growing quantizer.

The original admission status is preserved. Each log-delay residual uses the
mean of five earlier observations of the same profile. One pooled growing CDF
uses all earlier admitted residuals; there is no sliding window, RMS scaling,
conditional context, guard band or rebalance. Two bits are quantized per key
candidate: the low bit is public check information, the high bit is the raw
key bit. Raw bits and LDPC balances are not final secret keys.
"""

import gc
import json
from dataclasses import asdict, replace

import numpy as np

from pages.offline_keygen_v2 import calibration as C
from pages.offline_keygen_v2 import ldpc as L
from research import linear_key_scheme as core
from research.short_scheme_candidate import from_calibration
from research.simple_innovation_scheme import CASES

OUT = core.Path("output_data/notes/keygen_acf_growing_detrend_20261105")
MEAN_WINDOW = 5
CDF_MINIMUM = 1000


def residual(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for party, a in x.items():
        z = np.full(a.shape, np.nan)
        for j in range(a.shape[1]):
            ix = np.flatnonzero(np.isfinite(a[:, j]))
            v = a[ix, j]
            if len(v) > MEAN_WINDOW:
                history = np.lib.stride_tricks.sliding_window_view(
                    v[:-1], MEAN_WINDOW
                ).mean(axis=1)
                z[ix[MEAN_WINDOW:], j] = v[MEAN_WINDOW:] - history
        out[party] = z
    return out


def calibrate(raw) -> C.CalibResult:
    original_status = raw.status.copy()
    work = replace(raw, x=residual(raw.x))
    cfg = C.CalibConfig(
        algorithm="standard",
        n_target=CDF_MINIMUM,
        cdf="pooled",
        bits=1,
        bits_rule="fixed",
        update="growing",
        guard=0.0,
        check_bits=1,
        check_extra=True,
        rebalance=False,
        detrend="off",
    )
    result = C.calibrate(work, cfg)
    np.testing.assert_array_equal(raw.status, original_status)
    return result


def reconcile(c: C.CalibResult, case: str) -> dict:
    run = L.run(
        c,
        L.LdpcConfig(
            block_bits=256, soft=False, public_reliability=False
        ),
        session_id=f"acf-growing-detrend-{case}",
    )
    alice, bob = run.reconciled(), run.reconciled("bob")
    np.testing.assert_array_equal(alice, bob)
    np.savez_compressed(OUT / f"{case}_reconciled.npz", alice=alice, bob=bob)
    leak = sum(block.leak for block in run.blocks)
    return {
        "block_bits": 256,
        "reconciled_bits": len(alice),
        "equal": True,
        "confirmed_blocks": len(run.confirmed),
        "blocks": len(run.blocks),
        "reconciliation_leak_upper": leak,
        "public_check_bits_all_candidates": len(c.step),
        "balance_after_EC_only": max(0, len(alice) - leak),
        "final_secret_length": None,
    }


def main():
    import torch

    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    manifest = json.loads(core.PARENT.read_text())
    summary = {}
    for case in CASES:
        raw = core.read(manifest["caches"][case]["decor"]["path"])
        raw.acf_pooled = raw.acf_profiles = raw.cca_curves = None
        c = calibrate(raw)
        data = from_calibration(c)
        result = core.evaluate(data, [])
        result.update(
            case=case,
            algorithm="standard ACF + causal mean5 detrend + growing CDF",
            config=asdict(c.config),
            admitted_values=int((raw.status == C.ADMITTED).sum()),
            key_candidates=len(c.step),
            public_check_bits_all_candidates=len(c.step),
            raw_bits_not_final_key=True,
        )
        result["ldpc"] = reconcile(c, case)
        core.write(OUT / f"{case}.json", result)
        np.savez_compressed(
            OUT / f"{case}.npz",
            alice=data[:, 4].astype(np.uint8),
            bob=data[:, 5].astype(np.uint8),
            provenance=data[:, :4],
        )
        summary[case] = {
            "bits": result["bits"],
            "BDR": result["BDR"],
            "NIST_failures": {
                p: [t["name"] for t in result["party"][p]["nist"] if t["passed"] is False]
                for p in ("alice", "bob")
            },
            "ldpc": result["ldpc"],
        }
        core.write(OUT / "summary.json", summary)
        print("RESULT", case, summary[case], flush=True)
        del raw, c, data
        gc.collect()


if __name__ == "__main__":
    main()
