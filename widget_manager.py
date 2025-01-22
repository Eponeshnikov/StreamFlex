import logging
import streamlit as st

class WidgetManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if 'widget_states' not in st.session_state:
            st.session_state.widget_states = {}

    def save_widget_state(self, plugin_name, widget_key, value):
        plugin_states = st.session_state.widget_states.get(plugin_name, {})
        plugin_states[widget_key] = value
        st.session_state.widget_states[plugin_name] = plugin_states
        self.logger.info(f"Saved widget state: {plugin_name}.{widget_key}")

    def load_widget_state(self, plugin_name, widget_key, default=None):
        return st.session_state.widget_states.get(plugin_name, {}).get(widget_key, default)

    def export_state(self):
        return st.session_state.widget_states.copy()

    def import_state(self, state):
        st.session_state.widget_states = state.copy()
        self.logger.info("Imported widget states")