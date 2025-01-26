import os
import streamlit as st
from loguru import logger
from data_manager import DataManager
from widget_manager import WidgetManager
from plugin_manager import PluginManager
from state_manager import StateManager
from streamlit.components.v1 import html
from utils import get_colored_logs, logger_init

# Logging configuration
logger_init()


def main():
    st.set_page_config(page_title="Dynamic Plugin System", layout="wide")
    
    # Add Streamlit logo to sidebar
    with st.sidebar:
        st.logo("https://streamlit.io/images/brand/streamlit-mark-color.png", link="https://streamlit.io")
    
    st.title("🎛️ Dynamic Plugin System")
    
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
                        selected_plugins = st.session_state.get("selected_plugins", [])
                        if state_mgr.save_snapshot(snapshot_name, data_mgr, widget_mgr, selected_plugins):
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
            selected_snapshot = st.selectbox("Available snapshots", snapshots, key="snap_sel")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Load", key="load_btn"):
                    try:
                        selected_plugins = state_mgr.load_snapshot(selected_snapshot, data_mgr, widget_mgr)
                        if selected_plugins is not None:
                            st.session_state.selected_plugins = selected_plugins
                            logger.info(f"Loaded snapshot: {selected_snapshot}")
                            st.success(f"✅ Loaded: {selected_snapshot}")
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Load error: {e}")
                        st.error("❌ Failed to load snapshot")
            
            with col2:
                if st.button("🗑️ Delete", key="del_btn"):
                    try:
                        if state_mgr.delete_snapshot(selected_snapshot):
                            logger.info(f"Deleted snapshot: {selected_snapshot}")
                            st.success(f"✅ Deleted: {selected_snapshot}")
                            st.rerun()
                        else:
                            raise Exception("Snapshot delete failed")
                    except Exception as e:
                        logger.error(f"Delete error: {e}")
                        st.error("❌ Failed to delete snapshot")

    # Main Content Area
    st.header("🔌 Plugin Dashboard")
    
    # Plugin Selection
    available_plugins = [p.get_name() for p in plugin_mgr.get_plugins()]
    selected_plugins = st.multiselect(
        "🔌 Select Active Plugins",
        available_plugins,
        key="selected_plugins",
        default=st.session_state.get("selected_plugins", []),
        help="Select multiple plugins to activate them"
    )
    
    # Plugin Execution
    if selected_plugins:
        st.subheader("🚀 Active Plugins")
        for plugin_name in selected_plugins:
            with st.container():
                plugin = plugin_mgr.plugins.get(plugin_name)
                if plugin:
                    try:
                        st.markdown(f"#### {plugin_name}")
                        plugin.run(data_mgr, widget_mgr)
                        logger.info(f"Executed plugin: {plugin_name}")
                    except Exception as e:
                        logger.error(f"Plugin {plugin_name} failed: {e}")
                        st.error(f"❌ Error in {plugin_name}: {str(e)}")
    else:
        st.info("ℹ️ No plugins selected. Choose plugins from the dropdown above.")

    # Enhanced Debug Section
    with st.sidebar.expander("🔍 Debug Console"):
        st.write("### 📝 Session State")
        st.json(st.session_state)
        
        st.write("### 📟 Terminal Output")
        
        # Colored log display with auto-scroll
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
            height=300
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh All"):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Logs"):
                latest_log = max([os.path.join("logs", f) for f in os.listdir("logs") if f.endswith(".log")], 
                               key=os.path.getmtime)
                open(latest_log, "w").close()
                st.rerun()

if __name__ == "__main__":
    main()