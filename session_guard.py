"""Keep a running script alive when its browser tab goes away.

Streamlit ties script execution to the websocket. ``WebsocketSessionManager.
disconnect_session`` calls ``AppSession.request_script_stop()``
unconditionally, so a closed tab, a laptop that went to sleep or a client that
fails to drain a large forward message aborts whatever the script was doing.
There is no config option for it — the call is not guarded by anything.

For this app that means a multi-hour plugin chain dies between stages: on
2026-09-07 an OptiReceiver stage was killed four minutes in, right after the
Signal Channelizer had pushed a 361 MB delta at the browser, and the run never
reached the ResultsSaver parquet. The 8 GB of memmaps it had already written
became unreferenced garbage.

The guard suppresses that one call, and only while the script is actually
running. Everything else ``disconnect_session`` does is left alone: the
session still moves into storage (``server.disconnectedSessionTTL`` decides
how long it can be reattached to), its file watchers still stop, its
session-scoped caches are still cleared — this app declares none, all its
caches are global.

A page that still has the old session in memory *does* come back to it: the
reconnect id travels in the third entry of ``Sec-WebSocket-Protocol``, and a
tab whose page survived (a browser restored with its tabs, a laptop that
woke up, a dropped network) reattaches to the session it had. The first
thing such a client sends is a ``rerun_script``, and with
``runner.fastReruns`` (on by default) a rerun does not wait for the running
script: it stops it on the spot and starts over -- through
``ScriptRunner.request_stop``, not ``AppSession.request_script_stop``, so the
disconnect guard never sees it. On 2026-10-23 that killed a ray-tracing run
25 minutes in, and because the Generate trigger was still set the new run
started the same trace from zero. So the rerun a reattached client sends
while the script is running is held back and replayed once the script has
finished: the browser gets its full page then, and the run gets to end.
Only that one rerun is held -- anything the user clicks afterwards
interrupts the run as it always does.

A fresh page load cannot ask for the old session (the frontend keeps the
id in page memory only) and gets a new one. The run finishes and writes its
artifacts to disk; the session state it leaves behind is recovered from the
autosave snapshot (``StateManager.restore_autosave``), not from here.
:func:`detached_runs` tells the new session that a run is still going, so
the main page can keep it from starting a second copy of the same work.

Cost while detached: the run keeps rendering into the session's
``ForwardMsgQueue`` with no client to flush it. That is bounded by one full
render of the page — the queue coalesces deltas by delta path and is cleared
at the start of every script run — not by how long the run lasts.
"""

import threading
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from streamlit.runtime.app_session import AppSession

_INSTALLED_ATTR = "_streamflex_session_guard"
_KEEP_RUNNING_ATTR = "_streamflex_keep_running"
#: Set on a session a client reattached to while its script was running:
#: the next rerun request is the reconnect's own, and is held back.
_HOLD_RERUN_ATTR = "_streamflex_hold_rerun"
#: The held rerun's client state (``None`` stands for "previous state"),
#: replayed when the script ends. A list so "nothing held" and "held with
#: no client state" stay distinguishable.
_HELD_RERUN_ATTR = "_streamflex_held_rerun"

#: Sessions whose script was running when their client went away, with the
#: time it went, until they are reattached or the script ends.
_DETACHED: dict[str, tuple["AppSession", float]] = {}
_DETACHED_LOCK = threading.Lock()

#: Detached sessions whose script was still running when the session storage
#: let go of them, kept alive here until the script ends.
_PARKED: dict[int, "AppSession"] = {}
_PARKED_LOCK = threading.Lock()
_PARK_POLL_S = 30.0
_watcher: threading.Thread | None = None


def _is_running(session) -> bool:
    from streamlit.runtime.app_session import AppSessionState

    return getattr(session, "_state", None) == AppSessionState.APP_IS_RUNNING


def detached_runs(exclude: str | None = None) -> list[dict]:
    """Scripts still running for a client that has gone away.

    ``[{"session_id", "since"}]`` (``since`` is a ``time.time()``),
    excluding the session ``exclude`` -- the caller's own. A fresh page
    load gets a new session while the old one keeps running detached;
    without this the new one restores the autosave, finds Generate set and
    starts the same work again next to it.
    """
    with _DETACHED_LOCK:
        for session_id, (session, _since) in list(_DETACHED.items()):
            if not _is_running(session):
                del _DETACHED[session_id]
        return [
            {"session_id": session_id, "since": since}
            for session_id, (_session, since) in _DETACHED.items()
            if session_id != exclude
        ]


def _release_finished_sessions() -> int:
    """Shut down parked sessions whose script has ended; return how many."""
    with _PARKED_LOCK:
        done = [(k, s) for k, s in _PARKED.items() if not _is_running(s)]
        for key, _ in done:
            del _PARKED[key]
    for _, session in done:
        try:
            # __del__ already ran once (PEP 442: never again), so the
            # shutdown it skipped is done here explicitly.
            session.shutdown()
        except Exception as exc:  # a finished run must not take the app down
            logger.bind(class_name="streamlit").warning(
                "parked session {} shutdown failed: {}",
                getattr(session, "id", "?"),
                exc,
            )
        else:
            logger.bind(class_name="streamlit").info(
                "parked session {} finished its script and was shut down",
                getattr(session, "id", "?"),
            )
    return len(done)


def _watch_parked() -> None:
    import time

    while True:
        time.sleep(_PARK_POLL_S)
        _release_finished_sessions()


def _ensure_watcher() -> None:
    global _watcher
    if _watcher is None or not _watcher.is_alive():
        _watcher = threading.Thread(
            target=_watch_parked, name="streamflex-parked-sessions", daemon=True
        )
        _watcher.start()


def install_session_guards() -> None:
    """Patch streamlit so a disconnect does not stop a running script.

    Idempotent, and safe to call from every entry point: ``app.py`` calls it
    so the guard also covers the offline pages (peaks processing and model
    training are just as long-running as the plugin chain), and
    ``streamflex_app.py`` calls it again in case the module is run directly.
    """
    from streamlit.runtime.app_session import AppSession, AppSessionState
    from streamlit.runtime.scriptrunner import ScriptRunnerEvent
    from streamlit.runtime.websocket_session_manager import (
        WebsocketSessionManager,
    )

    if getattr(WebsocketSessionManager, _INSTALLED_ATTR, False):
        return

    original_request_script_stop = AppSession.request_script_stop

    def request_script_stop(self, *args, **kwargs):
        # The flag is set for the duration of one disconnect_session call, so
        # only the stop that the disconnect itself triggers is swallowed. A
        # later stop — the client's own stop button, session shutdown, a
        # second disconnect once the script is idle — goes through.
        if getattr(self, _KEEP_RUNNING_ATTR, False):
            logger.bind(class_name="streamlit").warning(
                "session {} lost its client mid-run: keeping the script "
                "alive instead of stopping it",
                getattr(self, "id", "?"),
            )
            return None
        return original_request_script_stop(self, *args, **kwargs)

    AppSession.request_script_stop = request_script_stop

    original_disconnect_session = WebsocketSessionManager.disconnect_session

    def disconnect_session(self, session_id: str) -> None:
        session_info = self._active_session_info_by_id.get(session_id)
        session = getattr(session_info, "session", None)
        running = (
            session is not None
            and getattr(session, "_state", None) == AppSessionState.APP_IS_RUNNING
        )
        if running:
            setattr(session, _KEEP_RUNNING_ATTR, True)
            with _DETACHED_LOCK:
                _DETACHED[session_id] = (session, time.time())
        try:
            return original_disconnect_session(self, session_id)
        finally:
            if running:
                setattr(session, _KEEP_RUNNING_ATTR, False)

    WebsocketSessionManager.disconnect_session = disconnect_session

    original_connect_session = WebsocketSessionManager.connect_session

    def connect_session(self, *args, **kwargs):
        session_id = original_connect_session(self, *args, **kwargs)
        with _DETACHED_LOCK:
            detached = _DETACHED.pop(session_id, None)
        session_info = self._active_session_info_by_id.get(session_id)
        session = getattr(session_info, "session", None)
        if detached is not None and session is not None and _is_running(session):
            setattr(session, _HOLD_RERUN_ATTR, True)
            logger.bind(class_name="streamlit").warning(
                "session {} reattached mid-run: its client's rerun waits for "
                "the script to finish",
                session_id,
            )
        return session_id

    WebsocketSessionManager.connect_session = connect_session

    original_request_rerun = AppSession.request_rerun

    def request_rerun(self, client_state):
        if getattr(self, _HOLD_RERUN_ATTR, False):
            setattr(self, _HOLD_RERUN_ATTR, False)
            if _is_running(self):
                setattr(self, _HELD_RERUN_ATTR, [client_state])
                logger.bind(class_name="streamlit").warning(
                    "session {}: rerun from the reattached client held until "
                    "the running script finishes",
                    getattr(self, "id", "?"),
                )
                return None
        return original_request_rerun(self, client_state)

    AppSession.request_rerun = request_rerun

    original_handle_event = AppSession._handle_scriptrunner_event_on_event_loop

    def _handle_scriptrunner_event_on_event_loop(self, sender, event, *args, **kwargs):
        # Read before the original handles it: SHUTDOWN clears the runner.
        current = sender is not None and sender is getattr(self, "_scriptrunner", None)
        result = original_handle_event(self, sender, event, *args, **kwargs)
        # SHUTDOWN of the current runner is its thread ending -- after the
        # script's own stop event, once nothing of the old run can still
        # interleave. A stale runner's SHUTDOWN says nothing about the run.
        if event == ScriptRunnerEvent.SHUTDOWN and current:
            held = getattr(self, _HELD_RERUN_ATTR, None)
            if held is not None:
                setattr(self, _HELD_RERUN_ATTR, None)
                if self._state != AppSessionState.SHUTDOWN_REQUESTED:
                    logger.bind(class_name="streamlit").info(
                        "session {}: script finished, replaying the held rerun",
                        getattr(self, "id", "?"),
                    )
                    original_request_rerun(self, held[0])
        return result

    AppSession._handle_scriptrunner_event_on_event_loop = (
        _handle_scriptrunner_event_on_event_loop
    )

    # The disconnect is only the first way a detached run dies. The session
    # then sits in the session storage for `disconnectedSessionTTL`; once
    # that has passed, the next storage write (any other tab disconnecting)
    # evicts it, the last reference goes, and `AppSession.__del__` calls
    # `shutdown()` -- which stops the script. A three-hour training whose
    # tab was closed died the moment the page was opened and closed again an
    # hour later. A running session is parked instead and shut down by the
    # watcher once its script has finished.
    original_del = AppSession.__del__

    def __del__(self):
        if _is_running(self):
            with _PARKED_LOCK:
                _PARKED[id(self)] = self
            _ensure_watcher()
            logger.bind(class_name="streamlit").warning(
                "session {} expired from storage mid-run: parked until its "
                "script finishes",
                getattr(self, "id", "?"),
            )
            return
        original_del(self)

    AppSession.__del__ = __del__
    setattr(WebsocketSessionManager, _INSTALLED_ATTR, True)

    logger.bind(class_name="streamlit").info(
        "session guard installed: a websocket disconnect no longer stops a "
        "running script"
    )
