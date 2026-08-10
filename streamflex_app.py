import os
import shutil
from datetime import datetime

import psutil
import streamlit as st
from loguru import logger
from streamlit.components.v1 import html

from data_manager import DataManager
from plugin_manager import PluginManager
from state_manager import StateManager
from utils import get_colored_logs, logger_init
from widget_manager import WidgetManager


class SnapshotError(RuntimeError):
    """A snapshot could not be saved or deleted."""


class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder = self._placeholder.container()
            self._placeholder.subheader("Table of Contents", divider="blue")
            self._placeholder.markdown(
                "\n".join(self._items), unsafe_allow_html=True
            )

    def _markdown(self, text, level, space=""):
        key = "".join([c if c.isalnum() else "-" for c in text]).lower()

        st.markdown(
            f"<{level} id='{key}' style='color: #5DADE2; font-style: italic;'>{text}</{level}><hr style='margin: 15px 0; background-color: #5DADE2; height: 1px; border: none;'>",
            unsafe_allow_html=True,
        )
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


toc = Toc()


def get_dir_size(directory):
    """
    Calculate total size of all files in a directory (recursively) in bytes

    Args:
        directory (str): Path to directory

    Returns:
        int: Total size in bytes (0 if directory doesn't exist)
    """
    total_size = 0
    if not os.path.exists(directory):
        return 0

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                continue
    return total_size


def format_size(size_bytes):
    """
    Convert size in bytes to human-readable format

    Args:
        size_bytes (int): Size in bytes

    Returns:
        str: Formatted size string (e.g., "1.23 MB")
    """
    if size_bytes == 0:
        return "0 bytes"

    units = ["bytes", "KB", "MB", "GB", "TB"]
    unit_index = 0

    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1

    return f"{size_bytes:.2f} {units[unit_index]}"


def get_cache_sizes():
    """
    Get sizes of all cache directories

    Returns:
        dict: Dictionary with sizes of each directory in bytes
              Format: {
                  'tmp': {'size_bytes': int, 'formatted': str},
                  'cache': {'size_bytes': int, 'formatted': str},
                  'output': {'size_bytes': int, 'formatted': str}
              }
    """
    sizes = {
        "tmp": {
            "size_bytes": get_dir_size(".tmp"),
            "formatted": format_size(get_dir_size(".tmp")),
        },
        "cache": {
            "size_bytes": get_dir_size(".cache"),
            "formatted": format_size(get_dir_size(".cache")),
        },
        "output": {
            "size_bytes": get_dir_size("output_data"),
            "formatted": format_size(get_dir_size("output_data")),
        },
    }
    return sizes


def clear_cache_dirs(
    clear_cache=False, clear_output=False, cache_categories=None
):
    """
    Clear specified cache directories with options for selective clearing.
    Always clears .tmp directory when called.

    Args:
        clear_cache (bool): Clear all files in .cache directory (if True) or selected categories (if False)
        clear_output (bool): Clear all files in output_data directory
        cache_categories (list): List of categories to clear from .cache directory

    Returns:
        tuple: (success, message) where success is boolean and message is status string
    """
    try:
        cleared = []

        # Always clear .tmp directory
        tmp_dir = ".tmp"
        if os.path.exists(tmp_dir):
            for filename in os.listdir(tmp_dir):
                file_path = os.path.join(tmp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            cleared.append("temporary files")

        # Clear .cache directory
        if clear_cache:
            cache_dir = ".cache"
            if os.path.exists(cache_dir):
                if cache_categories:
                    # Clear only selected categories
                    for filename in os.listdir(cache_dir):
                        # Split on '^' to get category (new format: Category^hash.extension)
                        file_category = filename.split("^")[0]
                        if file_category in cache_categories:
                            file_path = os.path.join(cache_dir, filename)
                            try:
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                            except Exception as e:
                                logger.error(
                                    f"Failed to delete {file_path}: {e}"
                                )
                    cleared.append(
                        f"selected cache categories: {', '.join(cache_categories)}"
                    )
                else:
                    # Clear entire cache
                    for filename in os.listdir(cache_dir):
                        file_path = os.path.join(cache_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path}: {e}")
                    cleared.append("entire cache")

        # Clear output_data directory
        if clear_output:
            output_dir = "output_data"
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
                cleared.append("output files")

        if not cleared:
            return False, "Nothing was cleared (no options selected)"

        return True, f"Successfully cleared: {', '.join(cleared)}"

    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        return False, f"Failed to clear: {e!s}"


PLUGIN_OUTPUT_ROOT = os.path.join("output_data", "plugins")


def list_plugin_outputs(root=PLUGIN_OUTPUT_ROOT):
    """
    Enumerate the per-run artifacts each plugin writes under output_data/plugins.

    Plugins with a "save results to file" option (SignalChannelizer,
    OptiReceiver) dump their heavy payloads into
    ``output_data/plugins/<plugin>/<timestamp>/``, and ResultsSaver writes its
    parquets next to them. Those are the only parts of output_data that are
    disposable once the results have been processed — everything else
    (processed/, dataset/, models/, first_component/) must stay.

    Args:
        root (str): Base folder holding the per-plugin subfolders.

    Returns:
        dict: ``{plugin_name: [(entry_name, path, size_bytes), ...]}`` with the
              entries of every plugin sorted newest first.
    """
    outputs = {}
    if not os.path.isdir(root):
        return outputs

    for plugin_name in sorted(os.listdir(root)):
        plugin_dir = os.path.join(root, plugin_name)
        if not os.path.isdir(plugin_dir):
            continue

        entries = []
        for entry_name in os.listdir(plugin_dir):
            entry_path = os.path.join(plugin_dir, entry_name)
            try:
                size = (
                    get_dir_size(entry_path)
                    if os.path.isdir(entry_path)
                    else os.path.getsize(entry_path)
                )
                mtime = os.path.getmtime(entry_path)
            except OSError:
                continue
            entries.append((entry_name, entry_path, size, mtime))

        entries.sort(key=lambda item: item[3], reverse=True)
        outputs[plugin_name] = [
            (name, path, size) for name, path, size, _ in entries
        ]
    return outputs


def delete_plugin_outputs(paths, root=PLUGIN_OUTPUT_ROOT):
    """
    Delete plugin run folders/files, refusing anything outside output_data/plugins.

    Args:
        paths (list): Paths previously reported by :func:`list_plugin_outputs`.
        root (str): Guard directory — nothing outside it is ever removed.

    Returns:
        tuple: ``(removed, freed_bytes, errors)``
    """
    root_abs = os.path.abspath(root)
    removed, errors = [], []
    freed = 0

    for path in paths:
        path_abs = os.path.abspath(path)
        if (
            path_abs == root_abs
            or os.path.commonpath([root_abs, path_abs]) != root_abs
        ):
            errors.append(f"{path}: outside {root}, skipped")
            continue
        try:
            if os.path.isdir(path_abs):
                size = get_dir_size(path_abs)
                shutil.rmtree(path_abs)
            elif os.path.exists(path_abs):
                size = os.path.getsize(path_abs)
                os.unlink(path_abs)
            else:
                continue
            freed += size
            removed.append(path)
            logger.info(f"Removed plugin output {path} ({format_size(size)})")
        except OSError as e:
            logger.error(f"Failed to delete {path}: {e}")
            errors.append(f"{path}: {e}")

    return removed, freed, errors


def plugin_output_cleanup_ui():
    """Per-plugin cleanup of the run folders under output_data/plugins."""
    st.caption(
        "Remove the per-run files written by the plugins' *save results to "
        "file* option. Only `output_data/plugins/` is touched — datasets, "
        "models and processed results stay."
    )
    outputs = list_plugin_outputs()
    if not outputs:
        st.info("No plugin output folders yet.")
        return

    plugin_names = sorted(outputs)
    plugin_choice = st.selectbox(
        "Plugin",
        plugin_names,
        format_func=lambda name: (
            f"{name} ({format_size(sum(e[2] for e in outputs[name]))})"
        ),
        key="plugin_output_cleanup_plugin",
    )
    entries = outputs.get(plugin_choice, [])
    if not entries:
        st.info(f"'{plugin_choice}' has no output files.")
        return

    labels = {
        f"{name} — {format_size(size)}": path for name, path, size in entries
    }
    sizes = {
        f"{name} — {format_size(size)}": size for name, _, size in entries
    }

    select_all = st.checkbox(
        f"Select all {len(entries)} runs",
        key="plugin_output_cleanup_all",
    )
    if select_all:
        chosen = list(labels)
    else:
        chosen = st.multiselect(
            "Runs (newest first)",
            list(labels),
            key="plugin_output_cleanup_runs",
        )

    selected_paths = [labels[label] for label in chosen]
    selected_size = sum(sizes[label] for label in chosen)

    if st.button(
        f"🗑️ Delete selected ({format_size(selected_size)})",
        key="plugin_output_cleanup_delete",
        disabled=not selected_paths,
    ):
        removed, freed, errors = delete_plugin_outputs(selected_paths)
        if removed:
            st.success(
                f"Deleted {len(removed)} entries, freed {format_size(freed)}"
            )
        if errors:
            st.error("\n\n".join(errors))
        if not removed and not errors:
            st.info("Nothing to delete.")


@st.fragment
def cache_management_ui():
    """Streamlit UI for cache management operations"""
    with st.expander(" Clear Cache Directories", icon="🧹"):
        # col1, col2 = st.columns([9, 1], vertical_alignment='top')
        # col1.subheader("Clear Cache Directories")
        st.button(
            "Refresh",
            key="refresh_cache_btn",
            icon=":material/restart_alt:",
            type="primary",
        )
        sizes = get_cache_sizes()
        col1, col2, col3, col4 = st.tabs(
            [".tmp", ".cache", "Output", "Plugins"]
        )
        with col1:
            st.metric(".tmp Size", sizes["tmp"]["formatted"])
        with col2:
            st.metric(".cache Size", sizes["cache"]["formatted"])
        with col3:
            st.metric("Output Size", sizes["output"]["formatted"])
        with col4:
            st.metric(
                "Plugin Outputs Size",
                format_size(get_dir_size(PLUGIN_OUTPUT_ROOT)),
            )
        # Cache files section
        clear_select = st.pills(
            "Select folder for clearing",
            [".cache + .tmp", "output_data"],
            selection_mode="multi",
            key="clear_trash_select",
        )
        clear_cache_box = ".cache + .tmp" in clear_select
        # cache_categories = set()
        # cache_dir = ".cache"
        # if os.path.exists(cache_dir):
        #     for filename in os.listdir(cache_dir):
        #         if '^' in filename:  # Changed to handle new format
        #             cache_categories.add(filename.split('^')[0])

        # clear_all_cache = st.toggle("Clear ALL cached data", key="clear_all_cache")
        # if cache_categories:
        #     selected_categories = st.multiselect(
        #         "Or select categories to clear:",
        #         sorted(cache_categories),
        #         disabled=clear_all_cache
        #     )

        clear_output = "output_data" in clear_select

        if st.button(
            "🗑️ Execute Clearing",
            key="execute_clearing",
            disabled=len(clear_select) == 0,
        ):
            success, msg = clear_cache_dirs(
                clear_cache=clear_cache_box,
                clear_output=clear_output,
                cache_categories=None,
            )
            if success:
                st.success(msg)
            else:
                st.error(msg)

    with st.expander(" Clear Plugin Outputs", icon="🔌"):
        plugin_output_cleanup_ui()


@st.fragment
def session_state_manager():
    toggle_state = st.toggle("Show Session State", key="show_session_state")
    if toggle_state:
        st.json(st.session_state, expanded=3)
        st.button("Refresh")
    else:
        st.info("Session state hidden")


# Logging configuration
logger_init()


def _trace_session_events():
    """Record who asks streamlit to stop a running script.

    A stop request raises StopException, which derives from BaseException and
    therefore leaves no trace in the app's own handlers. This version of
    streamlit does not log back messages either, so wrap the single funnel all
    three sources go through — the client's `stop_script` message, session
    shutdown, and websocket disconnect — and log the caller.
    """
    import traceback

    from streamlit.runtime.app_session import AppSession

    if getattr(AppSession, "_streamflex_traced", False):
        return

    original_request_script_stop = AppSession.request_script_stop

    def request_script_stop(self, *args, **kwargs):
        callers = " <- ".join(
            f"{frame.name}:{frame.lineno}"
            for frame in reversed(traceback.extract_stack()[-6:-1])
        )
        logger.bind(class_name="streamlit").warning(
            f"request_script_stop for session "
            f"{getattr(self, 'id', '?')} via {callers}"
        )
        return original_request_script_stop(self, *args, **kwargs)

    AppSession.request_script_stop = request_script_stop
    AppSession._streamflex_traced = True  # type: ignore[attr-defined]

    # The disconnect itself is decided one layer lower: uvicorn/websockets know
    # the close code (1009 = message too big, 1006 = abnormal, ...). Those are
    # stdlib-logging records that only reach stderr, so mirror them into the
    # loguru file.
    import logging

    class _ToLoguru(logging.Handler):
        def emit(self, record):
            # Routine chatter (session lifecycle, runtime states) goes to TRACE
            # so it stays out of the DEBUG log; anything at WARNING or above
            # keeps its own level, since that is where close codes show up.
            level = (
                record.levelname
                if record.levelno >= logging.WARNING
                else "TRACE"
            )
            logger.bind(class_name=record.name).log(level, record.getMessage())

    handler = _ToLoguru()
    for name in (
        "uvicorn.error",
        "websockets",
        "websockets.server",
        "streamlit.runtime.runtime",
        "streamlit.runtime.websocket_session_manager",
    ):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, _ToLoguru) for h in stdlib_logger.handlers):
            stdlib_logger.addHandler(handler)

    from streamlit.runtime.runtime_util import get_max_message_size_bytes
    from streamlit.web.server.starlette.starlette_server import (
        _get_websocket_settings,
    )

    logger.bind(class_name="streamlit").info(
        f"websocket settings: ping(interval, timeout)="
        f"{_get_websocket_settings()}, "
        f"max_message_size={get_max_message_size_bytes() / 1024**2:.0f}MB"
    )

    # How much is actually pushed at the client, and whether one message gets
    # near the size limit right before a disconnect.
    original_enqueue = AppSession._enqueue_forward_msg
    big_message_mb = 5

    def _enqueue_forward_msg(self, msg):
        try:
            size = msg.ByteSize()
            total = getattr(self, "_streamflex_bytes", 0) + size
            self._streamflex_bytes = total
            if size > big_message_mb * 1024**2:
                logger.bind(class_name="streamlit").warning(
                    f"forward msg {size / 1024**2:.1f}MB "
                    f"(type '{msg.WhichOneof('type')}'), "
                    f"session total {total / 1024**2:.1f}MB"
                )
        except Exception:  # never break rendering over instrumentation
            pass
        return original_enqueue(self, msg)

    AppSession._enqueue_forward_msg = _enqueue_forward_msg

    # Streamlit disconnects a client whose send queue overflows ("slow clients
    # (bad network, paused tabs)"), and that disconnect is what stops the run.
    # Watch the queue depth, and widen the limit: a single remote user over a
    # ~9 Mbit link cannot drain a heavy CIR render within 500 pending messages.
    from streamlit.web.server.starlette import (
        starlette_websocket as _ws_module,
    )

    _ws_module.WEBSOCKET_MAX_SEND_QUEUE_SIZE = 20000  # type: ignore[attr-defined]

    original_write = _ws_module.StarletteSessionClient.write_forward_msg

    def write_forward_msg(self, msg):
        queue = getattr(self, "_send_queue", None)
        if queue is not None:
            depth = queue.qsize()
            if depth and depth % 250 == 0:
                logger.bind(class_name="streamlit").warning(
                    f"client send queue depth {depth}/{queue.maxsize} "
                    "— client is not draining"
                )
        try:
            return original_write(self, msg)
        except Exception:
            logger.bind(class_name="streamlit").error(
                "write_forward_msg failed: closed="
                f"{getattr(self, '_closed', None) and self._closed.is_set()}, "
                f"queue={queue.qsize() if queue else '?'}/"
                f"{queue.maxsize if queue else '?'}"
            )
            raise

    _ws_module.StarletteSessionClient.write_forward_msg = write_forward_msg


_trace_session_events()


def load_monitor():
    """
    Display real-time CPU and memory usage metrics using Streamlit components.

    This function creates a two-column layout to show:
    1. CPU usage as a percentage and progress bar
    2. Memory usage as a percentage and progress bar

    It also displays warning messages if CPU or memory usage exceeds 90%.

    Note:
    This function relies on the Streamlit (st) and psutil libraries to be imported and available.
    """
    # CPU Usage
    cpu_col, mem_col = st.columns(2)
    with cpu_col:
        cpu_percent = psutil.cpu_percent()
        st.metric("CPU Usage", f"{cpu_percent}%")
        st.progress(cpu_percent / 100)

    # Memory Usage
    with mem_col:
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        st.metric("Memory Usage", f"{mem_percent}%")
        st.progress(mem_percent / 100)

    # Optional: Add warning thresholds
    if cpu_percent > 90:
        st.error("High CPU usage detected!")
    if mem_percent > 90:
        st.error("High Memory usage detected!")


def rerun_ui():
    rerun_default_scope = st.segmented_control(
        "Rerun Scope",
        ["app", "fragment"],
        default="fragment",
        help="Rerun scope for widgets",
    )
    return rerun_default_scope


@st.fragment
def rerun_bttn():
    rerun_all = st.button(
        "Rerun all", type="primary", icon=":material/autorenew:"
    )
    if rerun_all:
        st.rerun(scope="app")


def global_trigger_onclick():
    st.session_state["global_trigger"] = True


def global_trigger():
    """Styled global trigger with visual feedback and enhanced UI"""
    st.markdown("### ⚡ Global Control")

    if st.button(
        "Execute Global Trigger",
        type="primary",
        width="stretch",
        icon="⚡",
        help="Trigger action across all plugins simultaneously",
        on_click=global_trigger_onclick,
    ):
        st.toast("Trigger activated system-wide!", icon=":material/flash_on:")


def main():
    # Plugins read this key by index, so it must exist on the very first run of
    # a session. setdefault (not a plain assignment) keeps the value written by
    # the trigger's on_click callback, which fires before this rerun.
    st.session_state.setdefault("global_trigger", False)
    st.session_state["timestemp"] = (
        datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    st.set_page_config(page_title="Dynamic Plugin System", layout="wide")

    st.title("🎛️ StreamFlex")

    # Initialize managers with logging
    try:
        logger.info("Initializing application managers")
        data_mgr = DataManager()
        widget_mgr = WidgetManager()
        plugin_mgr = PluginManager()
        state_mgr = StateManager()
        logger.success("Managers initialized successfully")
    except Exception as e:
        logger.error(f"Manager initialization failed: {e}")
        st.error("Failed to initialize application components")
        return

    # Load plugins with error handling
    with st.spinner("🔌 Loading plugins..."):
        try:
            plugin_mgr.load_plugins()
            logger.info(f"Loaded {len(plugin_mgr.plugins)} plugins")
        except Exception as e:
            logger.error(f"Plugin loading failed: {e}")
            st.error("Failed to load plugins")
            return

    # Sidebar Section
    with st.sidebar:
        st.header("📸 Snapshot Management")

        # Save Snapshot
        with st.expander("💾 Save Snapshot", expanded=True):
            snapshot_name = st.text_input("Name your snapshot")
            if st.button("💾 Save", key="save_btn"):
                if snapshot_name:
                    try:
                        selected_plugins = st.session_state.get(
                            "selected_plugins", []
                        )
                        if state_mgr.save_snapshot(
                            snapshot_name,
                            data_mgr,
                            widget_mgr,
                            selected_plugins,
                        ):
                            logger.info(f"Saved snapshot: {snapshot_name}")
                            st.success(f"✅ Saved: {snapshot_name}")
                        else:
                            raise SnapshotError("Snapshot save failed")
                    except Exception as e:
                        logger.error(f"Save error: {e}")
                        st.error("❌ Failed to save snapshot")
                else:
                    st.warning("⚠️ Please enter a snapshot name")

        # Load/Delete Snapshots
        with st.expander("📂 Manage Snapshots", expanded=True):
            snapshots = state_mgr.list_snapshots()
            selected_snapshot = st.selectbox(
                "Available snapshots", snapshots, key="snap_sel"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Load", key="load_btn"):
                    try:
                        selected_plugins = state_mgr.load_snapshot(
                            selected_snapshot, data_mgr, widget_mgr
                        )
                        if selected_plugins is not None:
                            st.session_state.selected_plugins = (
                                selected_plugins
                            )
                            logger.info(
                                f"Loaded snapshot: {selected_snapshot}"
                            )
                            st.success(f"✅ Loaded: {selected_snapshot}")
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Load error: {e}")
                        st.error("❌ Failed to load snapshot")

            with col2:
                if st.button("🗑️ Delete", key="del_btn"):
                    try:
                        if state_mgr.delete_snapshot(selected_snapshot):
                            logger.info(
                                f"Deleted snapshot: {selected_snapshot}"
                            )
                            st.success(f"✅ Deleted: {selected_snapshot}")
                            st.rerun()
                        else:
                            raise SnapshotError("Snapshot delete failed")
                    except Exception as e:
                        logger.error(f"Delete error: {e}")
                        st.error("❌ Failed to delete snapshot")
        # st.divider()
        global_trigger()
        # st.divider()
        toc.placeholder(sidebar=True)
        cache_management_ui()
        with st.expander(
            "Rerun Control",
            expanded=False,
            icon=":material/published_with_changes:",
        ):
            rerun_scope = rerun_ui()
            rerun_bttn()

    # Plugin Selection
    available_plugins = [p.get_name() for p in plugin_mgr.get_plugins()]
    selected_plugins = st.multiselect(
        "🔌 Select Active Plugins",
        available_plugins,
        key="selected_plugins",
        default=st.session_state.get("selected_plugins", []),
        help="Select multiple plugins to activate them",
    )

    # Plugin Execution
    try:
        if selected_plugins:
            st.subheader("🚀 Active Plugins")
            for plugin_name in selected_plugins:
                with st.container():
                    plugin = plugin_mgr.plugins.get(plugin_name)
                    if plugin:
                        try:
                            toc.header(f"{plugin_name}")
                            if rerun_scope is not None:
                                plugin.global_rerun_scope = rerun_scope
                            plugin.run_notification(data_mgr, widget_mgr)
                            logger.info(f"Executed plugin: {plugin_name}")
                        except Exception as e:
                            logger.error(f"Plugin {plugin_name} failed: {e}")
                            st.error(f"❌ Error in {plugin_name}: {e!s}")
                        except BaseException as e:
                            # Streamlit's StopException/RerunException derive from
                            # BaseException, so an interrupted run leaves no trace
                            # in the handler above. Name it, then let it through.
                            logger.warning(
                                f"Plugin {plugin_name} interrupted by "
                                f"{type(e).__module__}.{type(e).__name__}: {e!r}"
                            )
                            raise
            toc.generate()
        else:
            st.info(
                "ℹ️ No plugins selected. Choose plugins from the dropdown above."
            )
    finally:
        # Must run even when the script is interrupted: otherwise the trigger
        # stays latched and every later rerun regenerates everything.
        st.session_state["global_trigger"] = False
    # Enhanced Debug Section
    with st.sidebar.expander("🔍 Debug Console"):
        st.subheader("📊 System Resources (Beta)")
        real_time_monitor = st.checkbox(
            "Enable real time monitor", key="real_time_monitor"
        )
        if real_time_monitor:
            st.fragment(run_every=1)(load_monitor)()
        else:
            load_monitor()
        tab1, tab2 = st.tabs(["📝 Session State", "📟 Terminal Output"])

        with tab1:
            session_state_manager()

        with tab2:
            # Colored log display with auto-scroll (existing code)
            html(
                f"""
                <div id="logContainer" 
                    style="
                        height: 300px;
                        overflow-y: auto;
                        background-color: #262730;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-family: monospace;
                        white-space: pre-wrap;
                    ">
                    {get_colored_logs(100)}
                </div>
                <script>
                    // Auto-scroll to bottom
                    var container = document.getElementById('logContainer');
                    container.scrollTop = container.scrollHeight;
                </script>
                """,
                height=300,
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh All"):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Logs"):
                latest_log = max(
                    [
                        os.path.join("logs", f)
                        for f in os.listdir("logs")
                        if f.endswith(".log")
                    ],
                    key=os.path.getmtime,
                )
                open(latest_log, "w").close()
                st.rerun()


if __name__ == "__main__":
    main()
