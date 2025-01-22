import streamlit as st
import logging
from data_manager import DataManager
from config_manager import ConfigManager
from widget_manager import WidgetManager
from plugin_manager import PluginManager
from state_manager import StateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("Dynamic Plugin System")
    
    # Initialize managers
    data_mgr = DataManager()
    config_mgr = ConfigManager()
    widget_mgr = WidgetManager()
    plugin_mgr = PluginManager()
    state_mgr = StateManager()
    
    # Load plugins
    plugin_mgr.load_plugins()
    
    # Snapshot management UI
    with st.sidebar:
        st.header("Snapshot Management")
        
        # Save snapshot
        snapshot_name = st.text_input("Snapshot name")
        if st.button("Save Snapshot"):
            selected_plugins = st.session_state.get("selected_plugins", [])
            if snapshot_name:
                success = state_mgr.save_snapshot(
                    snapshot_name,
                    data_mgr,
                    config_mgr,
                    widget_mgr,
                    selected_plugins
                )
                if success:
                    st.success(f"Saved snapshot: {snapshot_name}")
                else:
                    st.error("Failed to save snapshot")
            else:
                st.warning("Please enter a snapshot name")
        
        # Load snapshot
        snapshots = state_mgr.list_snapshots()
        selected_snapshot = st.selectbox("Available snapshots", snapshots)
        if selected_snapshot and st.button("Load Snapshot"):
            selected_plugins = state_mgr.load_snapshot(
                selected_snapshot,
                data_mgr,
                config_mgr,
                widget_mgr
            )
            if selected_plugins is not None:
                st.session_state.selected_plugins = selected_plugins
                st.success(f"Loaded snapshot: {selected_snapshot}")
                st.rerun()
        
        # Delete snapshot
        if selected_snapshot and st.button("Delete Snapshot"):
            if state_mgr.delete_snapshot(selected_snapshot):
                st.success(f"Deleted snapshot: {selected_snapshot}")
                st.rerun()
            else:
                st.error("Failed to delete snapshot")

    # Plugin selection
    available_plugins = [p.get_name() for p in plugin_mgr.get_plugins()]
    selected_plugins = st.multiselect(
        "Select Plugins",
        available_plugins,
        key="selected_plugins",
        default=st.session_state.get("selected_plugins", [])
    )
    
    # Run selected plugins
    for plugin_name in selected_plugins:
        plugin = plugin_mgr.plugins.get(plugin_name)
        if plugin:
            st.subheader(plugin_name)
            plugin.run(data_mgr, config_mgr, widget_mgr)

if __name__ == "__main__":
    main()