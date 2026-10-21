"""Joblib scratch, payload spilling and worker-count planning."""

import os
from typing import Any, Literal, overload

from loguru import logger

from utils.gpu import (
    CUDA_CONTEXT_VRAM_BYTES,
    GPU_WORKER_VRAM_BYTES,
    WORKER_WORKING_SET_FACTOR,
)
from utils.memory import available_memory_bytes, free_memory_bytes
from utils.storage import resolve_scratch_root, tmp_dir_for

#: Where joblib parks the array arguments it hands to loky workers. Left
#: alone it picks ``/dev/shm``, which on this box is tmpfs -- i.e. RAM, and
#: the one kind of RAM nothing accounts for: it belongs to no process, so
#: it shows up neither in the app's RSS nor in the cgroup's own usage, and
#: it survives the process that made it. Measured on 2026-09-22, two
#: Streamlit processes that had already died were still holding **50 GiB**
#: of it, which the next run then started out against.
JOBLIB_TEMP_FOLDER_ENV = "JOBLIB_TEMP_FOLDER"
#: The directories joblib may have used. ``/tmp`` is tmpfs here too.
JOBLIB_SCRATCH_CANDIDATES = ("/dev/shm", "/tmp")


def configure_joblib_temp_folder(scratch_root=None):
    """Point joblib's worker spill at a disk, not at RAM.

    Returns the directory it set, or ``None`` when the caller has already
    chosen one — an explicit ``JOBLIB_TEMP_FOLDER`` is a decision, not a
    default to overwrite.
    """
    existing = os.environ.get(JOBLIB_TEMP_FOLDER_ENV)
    if existing:
        return None
    root = scratch_root or resolve_scratch_root()
    folder = os.path.join(tmp_dir_for(root), "joblib")
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        return None
    os.environ[JOBLIB_TEMP_FOLDER_ENV] = folder
    return folder


def sweep_orphaned_joblib_folders(directories=None):
    """Delete joblib spill folders whose owning process is gone.

    joblib removes its folder when the executor shuts down cleanly. When
    the parent is killed instead — which is exactly what happens to a run
    that runs the box out of memory — nobody removes it, and on tmpfs the
    memory is not returned until somebody does. So this runs at start-up,
    the only moment after such a death when anything can.

    The owning pid is in the name (``joblib_memmapping_folder_<pid>_...``)
    and a folder is removed only when that pid is not alive **and** the
    folder belongs to this user. A live pid is left strictly alone: another
    session of this app may be mid-run.
    """
    import shutil

    removed_bytes = 0
    removed = 0
    for parent in directories or (
        *JOBLIB_SCRATCH_CANDIDATES,
        os.environ.get(JOBLIB_TEMP_FOLDER_ENV) or "",
    ):
        if not parent or not os.path.isdir(parent):
            continue
        try:
            names = os.listdir(parent)
        except OSError:
            continue
        for name in names:
            if not name.startswith("joblib_memmapping_folder_"):
                continue
            path = os.path.join(parent, name)
            parts = name.split("_")
            if len(parts) < 4 or not parts[3].isdigit():
                continue
            pid = int(parts[3])
            try:
                os.kill(pid, 0)
                continue  # alive: another run may own it
            except ProcessLookupError:
                pass
            except PermissionError:
                continue  # someone else's process, and someone else's files
            try:
                if os.stat(path).st_uid != os.getuid():
                    continue
                size = sum(
                    os.path.getsize(os.path.join(root, f))
                    for root, _dirs, files in os.walk(path)
                    for f in files
                )
                shutil.rmtree(path)
            except OSError:
                continue
            removed_bytes += size
            removed += 1
    return removed, removed_bytes


#: Arrays smaller than this stay in RAM: a file costs an inode, an open
#: and a page-table setup, and below a few megabytes that is more than the
#: memory it saves.
SPILL_MIN_ARRAY_BYTES = 8 * 1024**2
#: Moves that floor, the way ``STREAMFLEX_EXPERIMENT_ROOT`` moves the scan
#: library. Read per call rather than at import, so a test can force every
#: array through the memmap path and see the consumers meet one.
SPILL_MIN_BYTES_ENV = "STREAMFLEX_SPILL_MIN_BYTES"


def spill_min_array_bytes():
    """The floor under spilling, env override included."""
    raw = os.environ.get(SPILL_MIN_BYTES_ENV)
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return SPILL_MIN_ARRAY_BYTES


@overload
def spill_payload_arrays[DictPayload: dict[Any, Any]](
    obj: DictPayload,
    directory: str | os.PathLike[str],
    *,
    min_bytes: int | None = None,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "c",
    _counter: list[int] | None = None,
) -> tuple[DictPayload, int]: ...


@overload
def spill_payload_arrays(
    obj: object,
    directory: str | os.PathLike[str],
    *,
    min_bytes: int | None = None,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "c",
    _counter: list[int] | None = None,
) -> tuple[Any, int]: ...


def spill_payload_arrays(
    obj: object,
    directory: str | os.PathLike[str],
    *,
    min_bytes: int | None = None,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "c",
    _counter: list[int] | None = None,
) -> tuple[Any, int]:
    """Move an in-memory payload's big arrays to disk, read back as memmaps.

    The bus between plugins keeps every stage's payload alive at once, and
    a ray-tracing run puts 120 CIR sets on it: measured, 326 MiB each and
    883 MiB at the worst, which is ~45 GiB of *anonymous* memory. Anonymous
    memory cannot be reclaimed -- the kernel's only move under pressure is
    to kill the process, and on 2026-09-22 it did exactly that at the
    scope's 100 GiB ceiling, taking a finished four-hour trajectory run
    with it.

    The same bytes in a file are page cache: the reader sees an ordinary
    array, the pages are faulted in on demand, and under pressure the
    kernel drops the clean ones instead of the process. Nothing downstream
    changes, because a memmap *is* an ``ndarray`` -- it slices, it does
    arithmetic, and ``np.asarray`` on it is free.

    The mode is copy-on-write rather than read-only, and that is not
    caution: ``torch.from_numpy`` on a read-only array returns a tensor
    PyTorch itself warns is undefined to write to, which is a worse
    failure than the one being avoided. Under copy-on-write an untouched
    page is still clean page cache and still reclaimable -- only a page
    somebody writes turns private, and nothing downstream writes.

    Returns ``(payload, bytes_spilled)``. Arrays already backed by a file
    are left alone, and so is anything under ``min_bytes``.
    """
    import numpy as _np

    if min_bytes is None:
        min_bytes = spill_min_array_bytes()
    if _counter is None:
        _counter = [0, 0]

    if isinstance(obj, _np.ndarray):
        if isinstance(obj, _np.memmap) or obj.nbytes < int(min_bytes):
            # Already on disk (from an earlier pass over the same payload),
            # or too small to be worth a file. Either way it is left where
            # it is -- and `payload_spill_directories` is what keeps the
            # file it already lives in from being swept away.
            return obj, _counter[0]
        path = os.path.join(directory, f"bus_{_counter[1]:06d}.npy")
        _counter[1] += 1
        try:
            os.makedirs(directory, exist_ok=True)
            # ``np.save`` writes the logical array, so the non-contiguous
            # ``re``/``im`` views that complex payloads travel as come back
            # as ordinary contiguous arrays rather than as halves of
            # something.
            _np.save(path, _np.ascontiguousarray(obj))
            spilled = _np.load(path, mmap_mode=mmap_mode)
        except (OSError, ValueError):
            # Spilling is an optimisation, never a correctness
            # requirement: a full disk, a swept directory or a racing
            # rerun must not throw away a run that has already been
            # computed. Keep the array in memory and carry on.
            logger.warning(f"Could not spill an array to {path}; keeping it in RAM.")
            return obj, _counter[0]
        _counter[0] += int(obj.nbytes)
        return spilled, _counter[0]

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key], _ = spill_payload_arrays(
                value,
                directory,
                min_bytes=min_bytes,
                mmap_mode=mmap_mode,
                _counter=_counter,
            )
        return obj, _counter[0]

    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i], _ = spill_payload_arrays(
                value,
                directory,
                min_bytes=min_bytes,
                mmap_mode=mmap_mode,
                _counter=_counter,
            )
        return obj, _counter[0]

    if isinstance(obj, tuple):
        spilled = [
            spill_payload_arrays(
                value,
                directory,
                min_bytes=min_bytes,
                mmap_mode=mmap_mode,
                _counter=_counter,
            )[0]
            for value in obj
        ]
        return tuple(spilled), _counter[0]

    return obj, _counter[0]


def payload_spill_directories(obj, _seen=None):
    """Every directory the payload's memmapped arrays actually live in.

    What a payload references is *not* the directory of the most recent
    spill. A second pass over an already-spilled payload writes nothing --
    its arrays are memmaps, and memmaps are left alone -- so the newest
    directory can be empty while the bus reads entirely out of an older
    one. Keeping only the newest then deletes the live data: measured, a
    re-run wiped the very files the visualizers were drawing from, and the
    next pass died on `No such file or directory: bus_000006.npy`.
    """
    import numpy as _np

    if _seen is None:
        _seen = set()
    if isinstance(obj, _np.memmap):
        name = getattr(obj, "filename", None)
        if name:
            _seen.add(os.path.dirname(os.path.abspath(name)))
    elif isinstance(obj, dict):
        for value in obj.values():
            payload_spill_directories(value, _seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            payload_spill_directories(value, _seen)
    return _seen


def drop_spill_directories(parent, keep=()):
    """Remove spill directories under ``parent`` that nothing references.

    ``keep`` is the set of directories the *current* payload reads from --
    from `payload_spill_directories`, not a guess at which one is newest.
    Called after the bus has been handed its new payload and never before:
    the previous payload is read right up until the moment it is replaced.
    """
    import shutil

    if isinstance(keep, str):
        keep = (keep,)
    keep_abs = {os.path.abspath(k) for k in keep if k}
    removed = 0
    try:
        entries = os.listdir(parent)
    except OSError:
        return 0
    for name in entries:
        path = os.path.abspath(os.path.join(parent, name))
        if path in keep_abs or not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError:
            pass
    return removed


def process_rss_bytes():
    """Resident memory of this process, or 0 when it cannot be read.

    The one measurement that does not care how the data is *represented*.
    Counting numpy arrays is fine right up until something has already
    turned them into Python lists -- which is exactly what the saver's
    "Save as object" mode does to every 1-D array before anything
    downstream gets to look, so an array-based estimate reads a row that
    costs gigabytes as weighing nothing.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def payload_array_bytes(obj):
    """Bytes of every numpy array under ``obj``.

    Complex arrays travel split as ``{"re": ..., "im": ...}``, and those two
    are *views* of one buffer at half its dtype width, so they add back up
    to the array rather than to twice it. A payload whose arrays have been
    spilled to memmap files carries paths rather than arrays and is counted
    at what it actually occupies, which is nearly nothing.
    """
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(payload_array_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(payload_array_bytes(v) for v in obj)
    return 0


def payload_file_bytes(obj):
    """Bytes of every saved array file (``.bin`` memmap, ``.pickle``) that a
    string under ``obj`` names.

    With "Use np.memmap" or "Save Results in File" a stage hands the next
    one a *path* instead of an array, and :func:`payload_array_bytes` reads
    that as nothing -- while the worker that receives it loads the whole
    file and computes on it. An OptiReceiver run of 2026-10-10 (and again
    2026-10-21) planned 4 workers on that zero and took Streamlit down at
    the cgroup's 100 GiB.
    """
    if isinstance(obj, str):
        if obj.endswith((".bin", ".pickle")) and os.path.isfile(obj):
            return int(os.path.getsize(obj))
        return 0
    if isinstance(obj, dict):
        return sum(payload_file_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(payload_file_bytes(v) for v in obj)
    return 0


def worker_payload_bytes(*payload_lists, working_set_factor=None):
    """RAM one configuration costs a worker, measured from the data itself.

    The *largest* entry of each list rather than the average: configurations
    are handed out without regard to their size, so any worker can draw the
    big one, and a budget built on the average is wrong exactly when it
    matters.
    """
    factor = (
        WORKER_WORKING_SET_FACTOR
        if working_set_factor is None
        else float(working_set_factor)
    )
    carried = 0
    for payloads in payload_lists:
        carried += max(
            (payload_array_bytes(item) for item in (payloads or ())),
            default=0,
        )
    return max(1, int(carried * factor))


def plan_worker_processes(
    n_tasks,
    *,
    host_bytes_per_task,
    device_bytes_per_task=0,
    device="cpu",
    requested=0,
    host_margin=0.5,
    device_margin=0.7,
):
    """How many config workers this machine can actually feed, and why.

    A worker count is not a preference, it is whatever the scarcest of four
    things allows, and on this box the scarce one is rarely the CPU: a
    SignalChannelizer run with ``n_jobs=16`` lost four configurations to
    CUDA OOM (two workers holding 8.7 and 9.3 GiB of a 24 GiB card) and
    then took the whole app down on host RAM, because sixteen copies of a
    572 MiB payload is 9 GiB before any of them computes anything.

    So the count is measured from what a task actually carries rather than
    set: ``host_bytes_per_task`` is the payload the worker is handed plus
    whatever it builds that stays in RAM, and ``device_bytes_per_task`` is
    what it puts on the card *on top of* its CUDA context. ``requested``
    is a user ceiling (0 = none), never a floor -- asking for sixteen on a
    machine that fits three is how this went wrong in the first place.

    Returns ``(n_workers, plan)``, where ``plan`` names every limit and the
    one that bound, so a run that goes sequential says why rather than just
    being slow.
    """
    n_tasks = max(1, int(n_tasks))
    limits = {"tasks": n_tasks, "cpus": max(1, os.cpu_count() or 1)}

    host_need = max(1, int(host_bytes_per_task))
    limits["host_memory"] = max(
        1, int(available_memory_bytes() * float(host_margin) // host_need)
    )

    dev = str(device).lower()
    if "cuda" in dev or "gpu" in dev or "auto" in dev:
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if has_cuda:
            per_worker = CUDA_CONTEXT_VRAM_BYTES + max(
                0, int(device_bytes_per_task)
            )
            limits["device_memory"] = max(
                1,
                int(
                    free_memory_bytes("cuda")
                    * float(device_margin)
                    // per_worker
                ),
            )

    if requested:
        limits["requested"] = max(1, int(requested))

    n_workers = max(1, min(limits.values()))
    return n_workers, {
        "n_workers": n_workers,
        "limits": limits,
        "bound_by": min(limits, key=lambda k: limits[k]),
        "host_bytes_per_task": host_need,
        "device_bytes_per_task": max(0, int(device_bytes_per_task)),
    }


def gpu_worker_slots(group, size, *, vram_margin=0.7):
    """How many solver children this group may actually run in parallel.

    Capped by free device memory rather than by core count: the work is
    host-side and single-threaded (a path solve leaves the GPU at ~0% while
    one core rebuilds the megakernel), so more processes really is more
    throughput — right up to the point where the next scene does not fit
    and the driver starts thrashing.
    """
    size = max(1, int(size))
    try:
        free = int(free_memory_bytes("cuda"))
    except Exception:
        return size
    if free <= 0:
        return size
    allowed = int(free * float(vram_margin) // GPU_WORKER_VRAM_BYTES)
    return max(1, min(size, allowed))
