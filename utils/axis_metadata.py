"""Duplex metadata, parties, sample spreading and the axis-metadata carrier
with its time selector.
"""

import numpy as np
import streamlit as st


def find_duplex_metadata(value, _seen=None):
    """Find propagated duplex metadata in a result/config source chain."""
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return None
    _seen.add(id(value))
    if isinstance(value, dict):
        direct = value.get("duplex_metadata")
        if isinstance(direct, dict) and direct.get("side_order"):
            return direct
        for key in ("results", "parameters", "config_info", "source_info"):
            if key in value:
                found = find_duplex_metadata(value[key], _seen)
                if found:
                    return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = find_duplex_metadata(item, _seen)
            if found:
                return found
    return None


def duplex_batch_labels(value, batch_size):
    """Label first/second batch halves without changing batch indices."""
    metadata = find_duplex_metadata(value) or {}
    return direction_labels_from_metadata(metadata, batch_size)


def duplex_batch_pair(value, batch_size, selected_batch=0):
    """Return the matching forward/reverse batch indices for a duplex result.

    Duplex data stores all forward batches first and the corresponding reverse
    batches in the second half. For non-duplex data this returns only the
    selected batch, preserving the legacy visualizer behaviour.
    """
    metadata = find_duplex_metadata(value) or {}
    batch_size = int(batch_size)
    selected_batch = max(0, min(int(selected_batch), batch_size - 1))
    side_order = metadata.get("side_order") or []
    base = int(metadata.get("base_batch_size") or 0)
    if len(side_order) == 2 and not base and batch_size % 2 == 0:
        base = batch_size // 2
    if len(side_order) != 2 or base <= 0 or base * 2 != batch_size:
        return [(selected_batch, f"Batch {selected_batch}")]
    local = selected_batch % base
    return [
        (local, "Base → Mobile"),
        (base + local, "Mobile → Base"),
    ]


def direction_labels_from_metadata(metadata, batch_size):
    """Per-batch-index duplex direction labels from a ``duplex_metadata`` dict.

    Returns a list of length ``batch_size``. When the metadata describes a
    two-sided duplex channel (``side_order`` of length 2), the first
    ``base_batch_size`` indices are the forward (base-station → mobile) side and
    the rest the reverse (mobile → base-station) side; otherwise every index is
    a plain ``Batch i``.
    """
    metadata = metadata or {}
    batch_size = int(batch_size)
    side_order = metadata.get("side_order") or []
    base = int(metadata.get("base_batch_size") or 0)
    if len(side_order) == 2 and not base and batch_size % 2 == 0:
        base = batch_size // 2
    if len(side_order) != 2 or base * 2 != batch_size:
        return [f"Batch {i}" for i in range(batch_size)]
    first = (
        "Base → Mobile"
        if str(side_order[0]).startswith("forward")
        else "First side"
    )
    second = (
        "Mobile → Base"
        if str(side_order[1]).startswith("reverse")
        else "Second side"
    )
    return [
        first if base == 1 else f"{first} — batch {i}" for i in range(base)
    ] + [second if base == 1 else f"{second} — batch {i}" for i in range(base)]


# ---------------------------------------------------------------------------
# Axis metadata — a stage-agnostic descriptor of a 7-dim CIR/signal tensor.
#
# The pipeline tensors share the leading-5-axis layout
# ``[batch, rx, rx_ant, tx, tx_ant, ...]``. ``batch`` encodes the duplex
# direction (TX→RX / RX→TX), and — for trajectory JSONs where the RX walk was
# moved onto the time axis — the time axis is a concatenation of several
# per-TX walks. ``axis_metadata`` records both so any visualizer (in a plugin
# or a page) can split the time axis back into individual trajectories, filter
# by the owning TX, and know which direction a batch index represents. It rides
# through the pipeline via the standard ``config_info``/``source_info`` chain
# and is dim-agnostic (plain JSON-able dict), so ResultsSaver serializes it too.
# ---------------------------------------------------------------------------
AXIS_METADATA_VERSION = 1
CIR_AXIS_NAMES = ["batch", "rx", "rx_ant", "tx", "tx_ant", "path", "time"]


# ---- The eavesdropper as a pipeline participant -------------------------
#
# A CIR-Generator config given an Eve positions JSON (Position Selector,
# *Eavesdropper (Eve) walk*) emits a twin of every combination with her RX
# list in place of Bob's. Both twins carry ``party`` and ``party_pair`` in
# their parameters; every later stage treats Eve's twin exactly like Bob's
# except for the random realisations that belong to a *receiver* -- her
# thermal noise and her front-end impairments are hers, not a copy of his.
# With the same seed her AWGN would be his sample for sample, and on a walk
# near his that reads as an eavesdropper far better than any real one.

PARTY_BOB = "bob"
PARTY_EVE = "eve"


def cir_party(item) -> str:
    """``"eve"`` for an item descended from an Eve twin, else ``"bob"``.

    Walks ``source_info`` to the CIR-Generator's parameters, so it works on
    a CIR item and on anything downstream of one.
    """

    def _find(node):
        if not isinstance(node, dict):
            return None
        info = node.get("config_info", node)
        if not isinstance(info, dict):
            return None
        if info.get("plugin_key") == "cir_generator":
            return (info.get("parameters") or {}).get("party")
        sources = info.get("source_info") or []
        if isinstance(sources, dict):
            sources = [sources]
        for source in sources:
            found = _find(source)
            if found is not None:
                return found
        return None

    return PARTY_EVE if _find(item) == PARTY_EVE else PARTY_BOB


def party_seed(seed, party: str):
    """The seed a receiver's random realisation uses for ``party``.

    Bob keeps the configured seed unchanged, so runs without an Eve are
    bit-identical to before. Eve's is derived from it -- reproducible, and
    independent of Bob's stream.
    """
    if party != PARTY_EVE or seed is None:
        return seed
    state = np.random.SeedSequence([int(seed), 0x45564531]).generate_state(1)
    return int(state[0] & 0x7FFFFFFF)


def spread_sample_indices(points, k, rng=None, existing=None):
    """Pick ``k`` indices of ``points`` spread over the map, k-means++ style.

    Each next point is drawn with probability proportional to the squared
    distance to the nearest already chosen one (D² seeding), so picks
    cover the area instead of clustering the way a uniform draw does,
    while staying random. ``existing`` -- coordinates already selected --
    count as chosen, so a second press fills the gaps. ``rng`` is a
    ``random.Random`` (or anything with ``random()``/``randrange()``).
    """
    import random as _random

    rng = rng if rng is not None else _random.Random()
    pts = np.asarray(points, dtype=float).reshape(len(points), -1)
    n = len(pts)
    k = max(0, min(int(k), n))
    if k == 0:
        return []
    d2 = np.full(n, np.inf)
    if existing is not None and len(existing):
        ex = np.asarray(existing, dtype=float).reshape(len(existing), -1)
        dim = min(ex.shape[1], pts.shape[1])
        for e in ex[:, :dim]:
            d2 = np.minimum(d2, ((pts[:, :dim] - e) ** 2).sum(axis=1))
    chosen: list[int] = []
    for _ in range(k):
        w = d2.copy()
        w[chosen] = 0.0
        if np.isinf(w).any():
            # Nothing chosen yet: uniform over the remaining points.
            pool = [i for i in range(n) if i not in chosen]
            idx = pool[rng.randrange(len(pool))]
        elif w.sum() <= 0:
            pool = [i for i in range(n) if i not in chosen]
            idx = pool[rng.randrange(len(pool))]
        else:
            c = np.cumsum(w)
            idx = int(np.searchsorted(c, rng.random() * c[-1], side="right"))
            idx = min(idx, n - 1)
        chosen.append(idx)
        d2 = np.minimum(d2, ((pts - pts[idx]) ** 2).sum(axis=1))
    return chosen


def build_axis_metadata(
    *,
    batch_size,
    duplex_metadata=None,
    trajectory_segments=None,
    num_time_steps=None,
    tx_index=None,
    dt=None,
    axes=None,
):
    """Assemble the standardized ``axis_metadata`` dict (see module note).

    ``trajectory_segments`` is the per-combo, RX-list-rebased segment list
    (each with ``rx_start_index`` / ``n_points`` / ``tx_index`` / ``dt`` / ...)
    stored by the CIR generator after the RX→time swap. Its segments are mapped
    onto the time axis as ``time_segments``.
    """
    batch_size = int(batch_size)
    directions = direction_labels_from_metadata(
        duplex_metadata or {}, batch_size
    )
    time_segments = []
    for i, seg in enumerate(trajectory_segments or []):
        try:
            start = int(seg.get("rx_start_index", 0))
            count = int(seg.get("n_points", 0))
        except (TypeError, ValueError):
            continue
        entry = {
            "trajectory_id": i,
            "tx_index": seg.get("tx_index"),
            "start": start,
            "count": count,
            "dt": seg.get("dt"),
            "seed": seg.get("seed"),
            "mode": seg.get("mode"),
            "v_min": seg.get("v_min"),
            "v_max": seg.get("v_max"),
        }
        # A leg of a multi-tower walk: which walk, which leg, and where it
        # sits on that walk's clock -- what lets an analysis stitch the
        # legs back together or keep them apart. Only on such segments, so
        # every other run's metadata (and parquet schema) stays as it was.
        handover = seg.get("handover")
        if isinstance(handover, dict) and handover.get("walk_id") is not None:
            entry["time_start_index"] = int(seg.get("time_start_index") or 0)
            entry["handover_walk_id"] = str(handover["walk_id"])
            entry["handover_leg"] = int(handover.get("leg", 0))
            entry["handover_n_legs"] = int(handover.get("n_legs", 0))
        time_segments.append(entry)
    meta = {
        "version": AXIS_METADATA_VERSION,
        "axes": list(axes) if axes else list(CIR_AXIS_NAMES),
        "batch": {
            "kind": "duplex_direction",
            "directions": directions,
            "side_order": (duplex_metadata or {}).get("side_order") or [],
            "base_batch_size": (duplex_metadata or {}).get("base_batch_size"),
        },
        "time": {
            "kind": "trajectory" if time_segments else "static",
            "dt": dt,
            "num_time_steps": (
                int(num_time_steps) if num_time_steps is not None else None
            ),
            "segments": time_segments,
        },
        "tx_index": tx_index,
    }
    return meta


def find_axis_metadata(value, _seen=None):
    """Locate a propagated ``axis_metadata`` dict in a result/config chain."""
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return None
    _seen.add(id(value))
    if isinstance(value, dict):
        direct = value.get("axis_metadata")
        if isinstance(direct, dict) and "axes" in direct:
            return direct
        for key in ("results", "parameters", "config_info", "source_info"):
            if key in value:
                found = find_axis_metadata(value[key], _seen)
                if found:
                    return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = find_axis_metadata(item, _seen)
            if found:
                return found
    return None


def axis_metadata_time_selector(
    axis_metadata,
    num_time_steps,
    *,
    key_prefix,
    container=None,
    tx_filter=None,
):
    """Render trajectory / time-point selectors for a time-axis-bearing tensor.

    Splits the concatenated time axis into the walks recorded in
    ``axis_metadata['time']['segments']`` and lets the user pick a trajectory
    (optionally filtered to one TX) and a point within it. Returns
    ``(time_index, info)`` where ``info`` carries the resolved trajectory id,
    owning ``tx_index`` and the segment's ``(start, count)`` for callers that
    want to slice the whole walk.

    Falls back to a plain time-step slider when no trajectory metadata exists,
    so it is safe to call unconditionally.
    """
    ui = container if container is not None else st
    num_time_steps = int(num_time_steps)
    segments = ((axis_metadata or {}).get("time") or {}).get("segments") or []
    if tx_filter is not None:
        segments = [
            s for s in segments if s.get("tx_index") in (None, tx_filter)
        ]
    if not segments or num_time_steps <= 1:
        t_idx = 0
        if num_time_steps > 1:
            t_idx = ui.slider(
                "Time step",
                0,
                num_time_steps - 1,
                0,
                key=f"{key_prefix}_tstep",
            )
        return int(t_idx), {
            "trajectory_id": None,
            "tx_index": None,
            "start": 0,
            "count": num_time_steps,
        }
    labels = []
    for s in segments:
        tx = s.get("tx_index")
        tx_txt = f"TX {tx}" if tx is not None else "TX ?"
        labels.append(
            f"Trajectory {s['trajectory_id']} ({tx_txt}, "
            f"{s.get('count', 0)} pts)"
        )
    sel = ui.selectbox(
        "Select trajectory",
        list(range(len(segments))),
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_traj",
    )
    seg = segments[int(sel)]
    start = int(seg.get("start", 0))
    count = max(1, int(seg.get("count", 1)))
    end = min(start + count, num_time_steps)
    local = 0
    if end - start > 1:
        local = ui.slider(
            "Point in trajectory",
            0,
            end - start - 1,
            0,
            key=f"{key_prefix}_pt",
        )
    return start + int(local), {
        "trajectory_id": seg.get("trajectory_id"),
        "tx_index": seg.get("tx_index"),
        "start": start,
        "count": end - start,
    }
