"""Standard ACF admission, causal linear detrending and two binary bit planes.

The original admission status is preserved. Each log-delay residual subtracts
a line fitted to nine earlier observations of the same profile and evaluated
at the last of those observations. A pooled growing CDF uses all earlier
admitted residuals. Two public check bits are quantized in
addition to two key bits. Binary codewords are emitted one complete bit plane
at a time. Raw bits and LDPC balances are not final secret keys.
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

OUT = core.Path("output_data/notes/keygen_acf_growing_linear9_20261105")
TREND_WINDOW = 9
CDF_MINIMUM = 1000
KEY_BITS = 2
CHECK_BITS = 2


def trend_weights(n=TREND_WINDOW):
    t = np.arange(n, dtype=float)
    centred = t - t.mean()
    return np.full(n, 1 / n) + (n - 1 - t.mean()) * centred / (centred @ centred)


def residual(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    weights = trend_weights()
    for party, a in x.items():
        z = np.full(a.shape, np.nan)
        for j in range(a.shape[1]):
            ix = np.flatnonzero(np.isfinite(a[:, j]))
            v = a[ix, j]
            if len(v) > TREND_WINDOW:
                history = np.lib.stride_tricks.sliding_window_view(
                    v[:-1], TREND_WINDOW
                )
                z[ix[TREND_WINDOW:], j] = v[TREND_WINDOW:] - history @ weights
        out[party] = z
    return out


def binary_words(c: C.CalibResult) -> C.CalibResult:
    return replace(
        c,
        bits={
            p: [format(int(word), f"0{KEY_BITS}b") for word in v[c.kept]]
            for p, v in c.bins.items()
        },
    )


def calibrate(raw) -> C.CalibResult:
    original_status = raw.status.copy()
    work = replace(raw, x=residual(raw.x))
    cfg = C.CalibConfig(
        algorithm="standard",
        n_target=CDF_MINIMUM,
        cdf="pooled",
        bits=KEY_BITS,
        bits_rule="fixed",
        update="growing",
        guard=0.0,
        check_bits=CHECK_BITS,
        check_extra=True,
        rebalance=True,
        detrend="off",
    )
    result = binary_words(C.calibrate(work, cfg))
    np.testing.assert_array_equal(raw.status, original_status)
    return result


def plane_separated(c: C.CalibResult):
    data = from_calibration(c)
    return data[np.argsort(data[:, 3], kind="stable")]


def inverse_gray(value):
    value = np.asarray(value, dtype=np.int64)
    out = value.copy()
    shift = 1
    while shift < KEY_BITS:
        out ^= value >> shift
        shift += 1
    return out


def reconcile(c: C.CalibResult, case: str) -> dict:
    ec_input = replace(c, bins={p: inverse_gray(v) for p, v in c.bins.items()})
    run = L.run(
        ec_input,
        L.LdpcConfig(block_bits=256, soft=False, public_reliability=False),
        session_id=f"acf-growing-linear9-{case}",
    )
    alice, bob = run.reconciled(), run.reconciled("bob")
    np.testing.assert_array_equal(alice, bob)
    original = from_calibration(c)[: len(alice), 4].astype(np.uint8)
    np.testing.assert_array_equal(alice, original)
    np.savez_compressed(OUT / f"{case}_reconciled.npz", alice=alice, bob=bob)
    leak = sum(block.leak for block in run.blocks)
    return {
        "block_bits": 256,
        "reconciled_bits": len(alice),
        "equal": True,
        "confirmed_blocks": len(run.confirmed),
        "blocks": len(run.blocks),
        "reconciliation_leak_upper": leak,
        "public_check_bits_all_candidates": CHECK_BITS * len(c.step),
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
        data = plane_separated(c)
        result = core.evaluate(data, [])
        result.update(
            case=case,
            algorithm="standard ACF + causal linear9 detrend + growing CDF + binary planes",
            config=asdict(c.config),
            trend_weights=trend_weights().tolist(),
            bit_order="all MSB plane, then all second plane; stable within each plane",
            admitted_values=int((raw.status == C.ADMITTED).sum()),
            key_candidates=len(c.step),
            public_check_bits_all_candidates=CHECK_BITS * len(c.step),
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
                p: [
                    {"name": t["name"], "p_value": t["p_value"]}
                    for t in result["party"][p]["nist"]
                    if t["passed"] is False
                ]
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
