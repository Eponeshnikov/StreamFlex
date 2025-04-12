import os

import psutil
import streamlit as st
from loguru import logger
from streamlit.components.v1 import html

from data_manager import DataManager
from plugin_manager import PluginManager
from state_manager import StateManager
from utils import get_colored_logs, logger_init
from widget_manager import WidgetManager


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
        return False, f"Failed to clear: {str(e)}"


def cache_management_ui():
    """Streamlit UI for cache management operations"""
    with st.expander("🧹 Cache Management"):
        st.subheader("Clear Cache Directories")

        # Cache files section
        clear_select = st.pills(
            "Select folder for clearing",
            [".cache", "output_data"],
            selection_mode="multi",
            key='clear_trash_select'
        )
        clear_cache_box = ".cache" in clear_select
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

        if st.button("🗑️ Execute Clearing", key="execute_clearing", disabled=len(clear_select)==0):
            success, msg = clear_cache_dirs(
                clear_cache=clear_cache_box,
                clear_output=clear_output,
                cache_categories=None,
            )
            if success:
                st.success(msg)
                st.rerun()  # Refresh to show changes
            else:
                st.error(msg)


# Logging configuration
logger_init()


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


def main():
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
                            raise Exception("Snapshot save failed")
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
                            raise Exception("Snapshot delete failed")
                    except Exception as e:
                        logger.error(f"Delete error: {e}")
                        st.error("❌ Failed to delete snapshot")
        st.fragment(cache_management_ui)()

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
    if selected_plugins:
        st.subheader("🚀 Active Plugins")
        for plugin_name in selected_plugins:
            with st.container():
                plugin = plugin_mgr.plugins.get(plugin_name)
                if plugin:
                    try:
                        st.subheader(f"_:blue[{plugin_name}]_", divider="blue")
                        plugin.run(data_mgr, widget_mgr)
                        logger.info(f"Executed plugin: {plugin_name}")
                    except Exception as e:
                        logger.error(f"Plugin {plugin_name} failed: {e}")
                        st.error(f"❌ Error in {plugin_name}: {str(e)}")
    else:
        st.info(
            "ℹ️ No plugins selected. Choose plugins from the dropdown above."
        )

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
            st.json(st.session_state, expanded=3)

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
