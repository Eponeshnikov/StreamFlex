"""GPU memory release, the keep-warm GPU job / worker pools and the batched
solver submission with requeue.
"""

import os
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal, overload

import streamlit as st
from loguru import logger

from utils.memory import cgroup_oom_kills, solver_memory_headroom


def _cuda_driver():
    """The CUDA driver API through ctypes, or None (no driver, no GPU)."""
    try:
        import ctypes

        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            return None
        return cuda
    except OSError:
        return None


def _cuda_devices(cuda) -> list[int]:
    """Devices this process already has a CUDA context on.

    Retaining a primary context that is not active would *create* one
    (~300 MiB) just to measure it, so only active ones are touched.
    """
    import ctypes

    count = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return []
    active = []
    for device in range(count.value):
        flags, state = ctypes.c_uint(0), ctypes.c_int(0)
        if cuda.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(state)) == 0 and state.value:
            active.append(device)
    return active


def _device_free_bytes(cuda, devices: list[int]) -> int:
    """Free VRAM summed over ``devices`` (device-wide, all processes)."""
    import ctypes

    total_free = 0
    for device in devices:
        ctx = ctypes.c_void_p()
        if cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device) != 0:
            continue
        try:
            if cuda.cuCtxPushCurrent_v2(ctx) != 0:
                continue
            free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
            if cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) == 0:
                total_free += int(free.value)
            cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        finally:
            cuda.cuDevicePrimaryCtxRelease_v2(device)
    return total_free


def _trim_cuda_pools(cuda, devices: list[int]) -> None:
    """Hand every unused byte of the stream-ordered pools to the driver.

    Dr.Jit allocates with ``cuMemAllocAsync``: ``flush_malloc_cache`` frees
    its cache into the device's memory pool, and the pool keeps the pages
    (release threshold 0 only applies at a synchronisation the pool sees).
    """
    import ctypes

    for device in devices:
        for getter in (cuda.cuDeviceGetMemPool, cuda.cuDeviceGetDefaultMemPool):
            pool = ctypes.c_void_p()
            if getter(ctypes.byref(pool), device) == 0 and pool.value:
                cuda.cuMemPoolTrimTo(pool, ctypes.c_size_t(0))


def release_gpu_memory():
    """Return this process' cached VRAM to the driver.

    Two caching allocators are in play and freeing one does nothing for the
    other: torch's, which the heavy plugins allocate through, and Dr.Jit's,
    which Sionna RT (and the Position Selector's in-process radio maps and
    ray casts) allocate through.

    ``dr.flush_malloc_cache()`` alone looked like a no-op: Dr.Jit (1.5)
    frees its cache with ``cuMemFreeAsync``, into the device's
    stream-ordered pool, and the pool keeps every page. Measured: 1 GiB
    freed by the flush, 0 back; after ``dr.sync_thread()`` and
    ``cuMemPoolTrimTo(pool, 0)`` the process is down to its bare context
    (386 MiB). Memory still referenced -- a loaded scene's geometry and
    acceleration structure -- stays; dropping that is the caller's business
    (``utils.release_scene_memory``).

    Every call is best-effort: a CPU-only box, a driver that is already
    gone, or a Dr.Jit built without the CUDA backend must not take a plugin
    run down on the way out. Torch and Dr.Jit are only touched when already
    imported, and only CUDA contexts that exist are asked, so the call never
    creates one. Returns the bytes handed back (device-wide free memory
    before vs after), or 0 when there is no GPU to ask.
    """
    import sys

    cuda = _cuda_driver()
    devices = _cuda_devices(cuda) if cuda is not None else []
    if not devices:
        return 0
    before = _device_free_bytes(cuda, devices)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    dr = sys.modules.get("drjit")
    if dr is not None:
        try:
            dr.flush_malloc_cache()
            dr.sync_thread()
        except Exception:
            pass
    try:
        _trim_cuda_pools(cuda, devices)
    except Exception:
        pass
    return max(0, _device_free_bytes(cuda, devices) - before)


def release_worker_pool():
    """Shut the idle joblib/loky pool down instead of waiting it out.

    joblib keeps its executor warm for reuse, and every worker holds a CUDA
    context of its own — ~420 MiB, so an ``n_jobs=16`` stage parks ~6.5 GiB
    of pure overhead on the card. The pool *is* reaped on joblib's idle
    timeout, but that timeout is minutes, which is exactly when the next
    stage of the chain starts and finds the card full.

    Reads loky's module-level singleton rather than calling
    ``get_reusable_executor()``: the public accessor *starts* a pool when
    none exists, which would spawn the very workers this is meant to drop.
    Returns True when a pool was actually shut down.
    """
    try:
        from joblib.externals.loky import reusable_executor

        executor = getattr(reusable_executor, "_executor", None)
        if executor is None:
            return False
        executor.shutdown(wait=True)
        return True
    except Exception:
        return False


def gpu_memory_held_bytes():
    """VRAM this process holds, or None when it cannot be read.

    Torch only knows about its own allocator, and the memory that matters
    here is Dr.Jit's — which torch reports as zero and cannot free. So ask
    the driver for the per-process figure instead. Best-effort: no GPU, no
    ``nvidia-smi``, or a container without visibility all return None rather
    than raising in a ``finally``.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    me = os.getpid()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == me:
            try:
                return int(float(parts[1])) * 1024**2
            except ValueError:
                return None
    return 0


GPU_JOB_CACHE_ENTRIES = 4


@st.cache_resource(show_spinner=False)
def _gpu_job_store():
    """Results of work sent to a child process, shared per server.

    Same shape as ``_scene_mesh_store``: a plain dict under a lock, with the
    cache decorator only making it a singleton. The executor itself is
    loky's — the two heavy plugins already drive it under Streamlit, and it
    solves the spawn/``__main__`` problem that bare
    ``multiprocessing`` would re-run the page over.
    """
    return {
        "done": OrderedDict(),
        "running": {},
        "warm": set(),
        "lock": threading.Lock(),
    }


@overload
def submit_gpu_job[T](
    key: str,
    fn: Callable[..., T],
    *args: Any,
    background: Literal[False],
    warm_group: str | None = ...,
    **kwargs: Any,
) -> T: ...


@overload
def submit_gpu_job[T](
    key: str,
    fn: Callable[..., T],
    *args: Any,
    background: bool = ...,
    warm_group: str | None = ...,
    **kwargs: Any,
) -> T | None: ...


def submit_gpu_job(
    key, fn, *args, background=True, warm_group=None, **kwargs
):
    """Run ``fn`` in a child process and reclaim its VRAM when it is done.

    This is the only way to get GPU memory back from Sionna RT: Dr.Jit's
    allocator never returns device memory to the driver (see
    ``release_gpu_memory``), so the process that ray-traces has to be a
    process that exits. Measured on an LOD3 city, spawning the child,
    importing ``sionna.rt`` and loading the scene costs 0.8 s total, against
    minutes for the solve it replaces.

    ``submit`` returns in ~1 ms, so with ``background=True`` the script
    thread is never blocked: the first call returns ``None`` and the caller
    draws a placeholder and polls, the way ``scene_mesh_parts`` does. Set
    ``background=False`` to wait for the result inline (tests, workers).

    ``fn`` must be importable by qualified name in a fresh interpreter, so
    it belongs in a module like ``pages/rt_solver_worker.py`` — not in a
    plugin loaded through ``importlib`` under a synthetic name, and not in a
    closure.
    """
    from joblib.externals.loky import get_reusable_executor

    store = _gpu_job_store()
    with store["lock"]:
        if key in store["done"]:
            store["done"].move_to_end(key)
            return store["done"][key]
        pending = store["running"].get(key)
        if pending is None:
            pending = get_reusable_executor(max_workers=1).submit(
                fn, *args, **kwargs
            )
            store["running"][key] = pending
        if background and not pending.done():
            return None

    try:
        result = pending.result()
    except Exception:
        # Drop the failed job so the next rerun retries instead of
        # re-raising the same stale future forever.
        with store["lock"]:
            store["running"].pop(key, None)
        _reclaim_idle_gpu_worker(store)
        raise

    with store["lock"]:
        store["running"].pop(key, None)
        store["done"][key] = result
        while len(store["done"]) > GPU_JOB_CACHE_ENTRIES:
            store["done"].popitem(last=False)
        if warm_group is not None:
            store["warm"].add(warm_group)
    _reclaim_idle_gpu_worker(store)
    return result


#: VRAM one solver child holds once it has a scene and has solved once:
#: 1794 MiB measured on LOD3 Kazan (the CUDA context is 420 MiB of that,
#: and none of it ever comes back inside a live process). Used to cap how
#: many of them may run at once.
GPU_WORKER_VRAM_BYTES = 1800 * 1024**2


@st.cache_resource(show_spinner=False)
def _gpu_pool_store():
    """Dedicated single-worker executors, by group.

    Not `get_reusable_executor(max_workers=N)`: that is loky's *singleton*,
    shared with the heavy plugins, and resizing it shuts the pool down —
    including the child that is holding a scene warm. It also hands a task
    to whichever worker is free, and this needs the opposite: a frame must
    land on the same process every time, or every worker loads its own copy
    of the city and the card fills with duplicates. One single-worker
    executor per slot gives that affinity for nothing.
    """
    return {"pools": {}, "lock": threading.Lock()}


#: VRAM a loky worker of the heavy plugins parks before it has computed
#: anything: its own CUDA context. Measured at ~420 MiB, which is why
#: ``n_jobs=16`` puts 6.5 GiB of pure overhead on the card before a single
#: tensor. Rounded up, because a context that is one page over the estimate
#: costs a worker and a context that is under it costs the run.
CUDA_CONTEXT_VRAM_BYTES = 440 * 1024**2


#: What a worker holds beyond the payload it was handed: the output it is
#: filling, the tensors it uploaded, the intermediates. Measured as roughly
#: the payload again, so a task is budgeted at twice what it carries. Being
#: wrong high costs a worker; being wrong low cost this box its whole
#: Streamlit process to the kernel's OOM killer.
WORKER_WORKING_SET_FACTOR = 2.0


#: A parallel solve gives up a process when the memory left before the OOM
#: killer is below the larger of these: absolute, and as a share of the limit.
SOLVER_TRIM_HEADROOM_BYTES = 6 * 1024**3
SOLVER_TRIM_HEADROOM_FRACTION = 0.05
#: How often the headroom is read while a batch runs.
SOLVER_MEMORY_POLL_S = 5.0
#: Runs of one job (the first included) before its failure is the batch's.
SOLVER_JOB_ATTEMPTS = 3


def _solver_memory_low():
    headroom, limit = solver_memory_headroom()
    if headroom is None:
        return False
    floor = max(
        SOLVER_TRIM_HEADROOM_BYTES,
        int(SOLVER_TRIM_HEADROOM_FRACTION * int(limit or 0)),
    )
    return headroom < floor


def _is_device_oom(exc):
    from concurrent.futures import BrokenExecutor

    if isinstance(exc, BrokenExecutor):
        return False
    text = str(exc).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text


def submit_gpu_batch(
    group, jobs, *, size, on_result=None, on_wait=None, on_notice=None
):
    """Run several solver jobs across a group's own children, in parallel.

    ``jobs`` is a sequence of ``(fn, args, kwargs)``. Results come back in
    the order given, and ``on_result(index, elapsed)`` is called as each one
    lands so the caller can report progress — out of order, because that is
    the order they finish in. ``on_wait()`` is called about every 0.4 s
    while anything is still running, which is how a caller draws a bar for
    work that reports from inside the children rather than by finishing:
    one job here can be a quarter of a run.

    The children are kept between calls exactly as ``submit_gpu_job``'s warm
    worker is: each one loads the scene once and answers every later job
    from it. ``release_gpu_worker(group)`` ends them.

    A child that dies does not take the batch with it. Its job goes back to
    the head of the queue for a fresh process, and every job already done
    is kept -- a window of a walk is hours, and losing finished ones to a
    sibling's crash used to cost the whole combination. When the death is
    memory (the cgroup's OOM counter moved, CUDA ran out, or the headroom
    is low when it happens), the slot is closed instead of refilled: the
    run continues with one process fewer. The same trimming happens before
    it comes to that -- the youngest job is withdrawn and requeued once the
    headroom drops below ``SOLVER_TRIM_HEADROOM_*``, so the OOM killer
    never gets to pick a victim (it may pick the Streamlit server). A job
    that fails ``SOLVER_JOB_ATTEMPTS`` times raises, as does any ordinary
    exception from a job. ``on_notice(text)`` hears each of these events.
    """
    import time as _time
    from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, wait

    from joblib.externals.loky import ProcessPoolExecutor

    jobs = list(jobs)
    if not jobs:
        return []
    size = max(1, min(int(size), len(jobs)))
    # `size` is how many run at once; `jobs` may be many more than that.
    store = _gpu_pool_store()

    def _new_pool():
        # An explicit idle timeout, because loky's default reaps a worker
        # between frames and the next one then reloads the city. These die
        # when the view is switched off, which is `release_gpu_worker`'s job
        # and the only thing that returns their VRAM anyway.
        return ProcessPoolExecutor(max_workers=1, timeout=1800)

    with store["lock"]:
        pools = store["pools"].setdefault(group, [])
        while len(pools) < size:
            pools.append(_new_pool())
        slots: list[Any] = pools[:size]

    def _notice(text):
        logger.warning(text)
        if on_notice is not None:
            on_notice(text)

    def _drop(slot, replace):
        """Kill a slot's child; give the slot a fresh one or close it."""
        old = slots[slot]
        fresh = _new_pool() if replace else None
        slots[slot] = fresh
        with store["lock"]:
            shared = store["pools"].get(group, [])
            if old in shared:
                if fresh is not None:
                    shared[shared.index(old)] = fresh
                else:
                    shared.remove(old)
        try:
            old.shutdown(wait=False, kill_workers=True)
        except Exception:
            pass

    # Handed out one at a time, not dealt round-robin up front. Jobs here
    # are not equal: a window of a walk costs what its receivers cost, and
    # a blocked point is cheap where a point with rich multipath is not.
    # Dealt in advance, the run ends with one worker still grinding
    # through its share while the rest sit idle -- the load falls off and
    # so does the rate. A worker that finishes takes the next window
    # instead, and keeps the scene it already has.
    queue = list(range(len(jobs)))
    results: list[Any] = [None] * len(jobs)
    attempts = [0] * len(jobs)
    started = _time.time()
    inflight = {}
    oom_kills = cgroup_oom_kills()
    next_memory_check = started + SOLVER_MEMORY_POLL_S

    def _feed(slot):
        if not queue or slots[slot] is None:
            return
        index = queue.pop(0)
        fn, args, kwargs = jobs[index]
        try:
            future = slots[slot].submit(fn, *args, **dict(kwargs or {}))
        except BrokenExecutor:
            # Kept from an earlier call whose child has died since (or was
            # killed idle): it refuses work before running any.
            _drop(slot, replace=True)
            future = slots[slot].submit(fn, *args, **dict(kwargs or {}))
        inflight[future] = (index, slot, _time.time())

    def _live():
        return sum(pool is not None for pool in slots)

    def _idle():
        busy = {slot for _i, slot, _t in inflight.values()}
        return [
            slot
            for slot, pool in enumerate(slots)
            if pool is not None and slot not in busy
        ]

    def _fill():
        for slot in _idle():
            _feed(slot)

    _fill()
    while inflight:
        landed, _ = wait(
            set(inflight), timeout=0.4, return_when=FIRST_COMPLETED
        )
        for future in landed:
            index, slot, _t0 = inflight.pop(future)
            try:
                results[index] = future.result()
            except Exception as exc:
                died = isinstance(exc, BrokenExecutor)
                device_oom = _is_device_oom(exc)
                if not (died or device_oom):
                    raise
                attempts[index] += 1
                if attempts[index] >= SOLVER_JOB_ATTEMPTS:
                    raise
                kills = cgroup_oom_kills()
                memory = (
                    device_oom or kills > oom_kills or _solver_memory_low()
                )
                oom_kills = kills
                queue.insert(0, index)
                shrink = memory and _live() > 1
                _notice(
                    f"Job {index + 1}/{len(jobs)} lost its process "
                    f"({type(exc).__name__}"
                    f"{', memory' if memory else ''}); requeued, attempt "
                    f"{attempts[index] + 1}/{SOLVER_JOB_ATTEMPTS}"
                    + (
                        f", running on {_live() - 1} process(es) from now"
                        if shrink
                        else ""
                    )
                    + "."
                )
                _drop(slot, replace=not shrink)
                continue
            if on_result is not None:
                on_result(index, _time.time() - started)
        now = _time.time()
        if now >= next_memory_check:
            next_memory_check = now + SOLVER_MEMORY_POLL_S
            if _live() > 1 and _solver_memory_low():
                idle = _idle()
                if idle:
                    # An idle child still holds its scene; it goes first.
                    _drop(idle[-1], replace=False)
                    _notice(
                        "Memory headroom low: an idle solver process was "
                        f"closed, {_live()} process(es) left."
                    )
                else:
                    # The youngest job has the least to lose.
                    future, (index, slot, _t0) = max(
                        inflight.items(), key=lambda item: item[1][2]
                    )
                    del inflight[future]
                    queue.insert(0, index)
                    _drop(slot, replace=False)
                    _notice(
                        "Memory headroom low: job "
                        f"{index + 1}/{len(jobs)} withdrawn and requeued, "
                        f"running on {_live()} process(es) from now."
                    )
        # Every live slot without a job takes the next one -- including a
        # job just requeued while the slot that would have refilled was
        # closed, which would otherwise end the loop with it never run.
        _fill()
        if on_wait is not None and inflight:
            on_wait()
    return results


def release_gpu_worker(warm_group):
    """Drop a keep-warm claim and kill the worker once none are left.

    The counterpart to ``submit_gpu_job(warm_group=...)``: while a group is
    registered the child survives between calls, so the scene it loaded stays
    warm and the next render is just a figure build. Releasing the last group
    ends the process, and *that* is what returns its VRAM — deleting the
    scene inside it would not (measured: load 892 MiB, delete, flush, still
    892 MiB).
    """
    store = _gpu_job_store()
    with store["lock"]:
        store["warm"].discard(warm_group)
        # Cached figures belong to the process that made them; keeping them
        # would hand back results for a scene that no longer exists.
        for cached in [k for k in store["done"] if _in_group(k, warm_group)]:
            del store["done"][cached]
    pool_store = _gpu_pool_store()
    with pool_store["lock"]:
        pools = pool_store["pools"].pop(warm_group, [])
    for pool in pools:
        # Each of these holds a scene of its own, which is the whole reason
        # they exist and the whole reason they must not outlive the view.
        pool.shutdown(wait=False, kill_workers=True)
    return _reclaim_idle_gpu_worker(store) or bool(pools)


def _in_group(key, warm_group):
    return isinstance(key, str) and key.startswith(f"{warm_group}:")


def _reclaim_idle_gpu_worker(store):
    """Kill the worker once nothing is queued — that is what frees the VRAM.

    Only when the store is idle: the executor has one worker, so shutting it
    down while another job is pending would abort that job too — and only
    when nothing has claimed the worker with ``warm_group``, since killing it
    would throw away the scene that claim exists to keep loaded.
    """
    with store["lock"]:
        if store["running"] or store["warm"]:
            return False
    return release_worker_pool()
