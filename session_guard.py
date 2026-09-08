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

What the guard does *not* do is bring the browser back to the run. A fresh
page load has no way to ask for the old session: the reconnect id travels in
the third entry of ``Sec-WebSocket-Protocol`` and the frontend keeps it in
page memory only (there is no ``localStorage`` entry for it anywhere in the
bundle), so a reload always gets a new session. The run finishes and writes
its artifacts to disk; the session state it leaves behind is recovered from
the autosave snapshot (``StateManager.restore_autosave``), not from here.

Cost while detached: the run keeps rendering into the session's
``ForwardMsgQueue`` with no client to flush it. That is bounded by one full
render of the page — the queue coalesces deltas by delta path and is cleared
at the start of every script run — not by how long the run lasts.
"""

from loguru import logger

_INSTALLED_ATTR = "_streamflex_session_guard"
_KEEP_RUNNING_ATTR = "_streamflex_keep_running"


def install_session_guards() -> None:
    """Patch streamlit so a disconnect does not stop a running script.

    Idempotent, and safe to call from every entry point: ``app.py`` calls it
    so the guard also covers the offline pages (peaks processing and model
    training are just as long-running as the plugin chain), and
    ``streamflex_app.py`` calls it again in case the module is run directly.
    """
    from streamlit.runtime.app_session import AppSession, AppSessionState
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
        try:
            return original_disconnect_session(self, session_id)
        finally:
            if running:
                setattr(session, _KEEP_RUNNING_ATTR, False)

    WebsocketSessionManager.disconnect_session = disconnect_session
    setattr(WebsocketSessionManager, _INSTALLED_ATTR, True)

    logger.bind(class_name="streamlit").info(
        "session guard installed: a websocket disconnect no longer stops a "
        "running script"
    )
