"""Free / available memory (cgroup aware), solver headroom and OOM kills."""

import os


# ---------------------------------------------------------------------------
# Memory-aware N-dimensional chunk planner.
#
# The heavy stages (SignalChannelizer delay-and-sum, OptiReceiver matched
# filter / peak finder) used to chunk only the batch and time axes. With
# trajectory JSONs the other axes (rx, tx, ...) can also be large, so we split
# the tensor across *all* requested axes into blocks sized to a fraction of the
# free RAM/VRAM (divided across parallel workers). ``free_memory_bytes`` picks
# the right pool for the device; ``plan_axis_chunks`` returns a per-axis chunk
# size; ``iter_nd_blocks`` walks the resulting blocks as tuples of slices.
# ---------------------------------------------------------------------------
def free_memory_bytes(device="cpu"):
    """Free memory (bytes) for ``device`` — VRAM for cuda/gpu, else host RAM.

    ``auto`` resolves the same way the heavy blocks pick their compute
    device: CUDA when available, host otherwise. Budgeting host RAM while
    computing on the GPU sizes blocks far past VRAM and OOMs.
    """
    dev = str(device).lower()
    if "cuda" in dev or "gpu" in dev or "auto" in dev:
        try:
            import torch

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return int(free)
        except Exception:
            pass
    return available_memory_bytes()


def _cgroup_v2_available_bytes(
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    proc_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
):
    """Return the remaining memory in this process' cgroup v2 hierarchy.

    A systemd scope can impose ``MemoryMax`` below the host's available RAM.
    Check the current cgroup and all its parents because a parent slice may be
    the limiting one. ``None`` means that no finite readable limit was found.
    """
    try:
        with open(proc_cgroup, encoding="utf-8") as fh:
            rel = next(
                line.rstrip("\n").split("::", 1)[1]
                for line in fh
                if "::" in line
            )
    except (OSError, StopIteration):
        return None

    current = os.path.normpath(os.path.join(cgroup_root, rel.lstrip("/")))
    root = os.path.normpath(cgroup_root)
    remaining = []
    while current == root or current.startswith(root + os.sep):
        try:
            with open(
                os.path.join(current, "memory.max"), encoding="utf-8"
            ) as fh:
                limit_text = fh.read().strip()
            if limit_text != "max":
                with open(
                    os.path.join(current, "memory.current"), encoding="utf-8"
                ) as fh:
                    used = int(fh.read().strip())
                remaining.append(max(0, int(limit_text) - used))
        except (OSError, ValueError):
            pass
        if current == root:
            break
        current = os.path.dirname(current)
    return min(remaining) if remaining else None


def available_memory_bytes():
    """Memory currently available to this process, including cgroup limits."""
    try:
        import psutil

        host_available = int(psutil.virtual_memory().available)
    except Exception:
        host_available = 4 * 1024**3
    cgroup_available = _cgroup_v2_available_bytes()
    if cgroup_available is not None:
        return min(host_available, cgroup_available)
    return host_available


def _cgroup_v2_dirs(
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    proc_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
):
    """This process' cgroup v2 directory and its parents, innermost first."""
    try:
        with open(proc_cgroup, encoding="utf-8") as fh:
            rel = next(
                line.rstrip("\n").split("::", 1)[1]
                for line in fh
                if "::" in line
            )
    except (OSError, StopIteration):
        return []
    current = os.path.normpath(os.path.join(cgroup_root, rel.lstrip("/")))
    root = os.path.normpath(cgroup_root)
    dirs = []
    while current == root or current.startswith(root + os.sep):
        dirs.append(current)
        if current == root:
            break
        current = os.path.dirname(current)
    return dirs


def solver_memory_headroom(
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    proc_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
):
    """``(headroom, limit)`` in bytes before the OOM killer acts.

    Unlike :func:`available_memory_bytes`, a cgroup's inactive page cache
    is not counted as used: the kernel drops it before it kills anything,
    and a run that has written gigabytes of parquet sits at ``memory.max``
    on cache alone. ``limit`` is the binding total, for a relative margin.
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        headroom, limit = int(vm.available), int(vm.total)
    except Exception:
        return None, None
    for directory in _cgroup_v2_dirs(cgroup_root, proc_cgroup):
        try:
            with open(os.path.join(directory, "memory.max"), encoding="utf-8") as fh:
                limit_text = fh.read().strip()
            if limit_text == "max":
                continue
            with open(
                os.path.join(directory, "memory.current"), encoding="utf-8"
            ) as fh:
                used = int(fh.read().strip())
            inactive_file = 0
            with open(os.path.join(directory, "memory.stat"), encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("inactive_file "):
                        inactive_file = int(line.split()[1])
                        break
        except (OSError, ValueError):
            continue
        room = max(0, int(limit_text) - (used - inactive_file))
        if room < headroom:
            headroom, limit = room, int(limit_text)
    return headroom, limit


def cgroup_oom_kills(
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    proc_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
):
    """OOM kills so far in this process' cgroup (``memory.events``), or 0.

    The scope ``run_streamlit.sh`` starts is where a solver child dies when
    the run outgrows ``MemoryMax``; the counter going up between two reads
    is how a dead child is told apart from one that crashed on its own.
    """
    dirs = _cgroup_v2_dirs(cgroup_root, proc_cgroup)
    if not dirs:
        return 0
    try:
        with open(os.path.join(dirs[0], "memory.events"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("oom_kill "):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0
