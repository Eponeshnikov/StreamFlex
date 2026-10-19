"""Row and N-dim chunk planning from memory budgets."""

import os

from utils.memory import available_memory_bytes, free_memory_bytes


def parquet_uncompressed_bytes_per_row(file_path):
    """Average uncompressed Parquet payload per top-level row.

    File size is a poor proxy for nested numeric data because Parquet encoding
    and compression disappear as soon as Polars materialises it. Metadata gives
    a cheap estimate without reading the payload itself.
    """
    import glob

    import pyarrow.parquet as pq

    paths = (
        sorted(glob.glob(os.path.join(file_path, "*.parquet")))
        if os.path.isdir(file_path)
        else [file_path]
    )
    total_rows = 0
    total_bytes = 0
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        total_rows += int(metadata.num_rows)
        for row_group_idx in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_idx)
            for column_idx in range(row_group.num_columns):
                total_bytes += int(
                    row_group.column(column_idx).total_uncompressed_size
                )
    if total_rows <= 0 or total_bytes <= 0:
        raise ValueError(f"No Parquet row-size metadata found in {file_path}")
    return max(1, (total_bytes + total_rows - 1) // total_rows)


def plan_axis_chunks(
    shape,
    splittable_axes,
    bytes_per_element,
    *,
    device="cpu",
    n_workers=1,
    margin=0.65,
    max_bytes=None,
    min_chunk=1,
) -> dict[int, int]:
    """Greedy memory-aware chunk sizes for an N-dim tensor.

    Returns ``{axis: chunk_size}`` for every axis in ``splittable_axes`` such
    that one block — ``prod(chunk over splittable) * prod(full over the rest)
    * bytes_per_element`` — fits the budget. Non-splittable axes stay whole.
    The largest current chunk is halved each step, so blocks stay roughly
    balanced instead of collapsing one axis to 1. ``max_bytes`` (when > 0)
    overrides the auto budget; otherwise the budget is
    ``free_memory_bytes(device) * margin / n_workers``.
    """
    shape = [max(1, int(s)) for s in shape]
    splittable = [ax for ax in splittable_axes if 0 <= ax < len(shape)]
    if max_bytes is not None and max_bytes > 0:
        budget = int(max_bytes)
    else:
        budget = int(
            free_memory_bytes(device) * float(margin) / max(1, int(n_workers))
        )
    bytes_per_element = max(1, int(bytes_per_element))
    budget = max(budget, bytes_per_element)
    fixed = 1
    for ax in range(len(shape)):
        if ax not in splittable:
            fixed *= shape[ax]
    chunks = {ax: shape[ax] for ax in splittable}

    def block_bytes():
        prod = fixed
        for ax in splittable:
            prod *= chunks[ax]
        return prod * bytes_per_element

    guard = 0
    while block_bytes() > budget and guard < 4096:
        guard += 1
        candidates = [ax for ax in splittable if chunks[ax] > min_chunk]
        if not candidates:
            break
        ax = max(candidates, key=lambda a: chunks[a])
        chunks[ax] = max(int(min_chunk), chunks[ax] // 2)
    return chunks


def iter_nd_blocks(shape, chunks):
    """Yield tuples of slices tiling ``shape`` into ``chunks`` blocks.

    ``chunks`` is ``{axis: size}``; axes absent from it are taken whole.
    """
    import itertools as _it

    ranges = []
    for ax, dim in enumerate(shape):
        dim = int(dim)
        step = int(chunks.get(ax, dim)) or dim
        step = max(1, step)
        ranges.append(
            [(s, min(s + step, dim)) for s in range(0, dim, step)]
            or [(0, dim)]
        )
    for combo in _it.product(*ranges):
        yield tuple(slice(a, b) for (a, b) in combo)


def suggest_row_batch_size(
    bytes_per_row,
    *,
    n_workers=1,
    margin=0.65,
    min_rows=1,
    max_rows=None,
):
    """Auto row/config batch size for the Parquet-streaming pages.

    ``bytes_per_row`` is the caller's estimate of the peak working-set cost of
    one row/config (exploded columns + intermediate frames). Returns how many
    rows fit the effective free RAM (host and cgroup limit) × margin ÷
    workers, clamped to ``[min_rows, max_rows]``.
    """
    bytes_per_row = max(1, int(bytes_per_row))
    free = available_memory_bytes()
    budget = int(free * float(margin) / max(1, int(n_workers)))
    rows = max(int(min_rows), budget // bytes_per_row)
    if max_rows is not None:
        rows = min(int(max_rows), rows)
    return int(rows)
