import logging
import streamlit as st

class DataManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if 'shared_data' not in st.session_state:
            st.session_state.shared_data = {}

    def set_data(self, key, value):
        self.logger.info(f"Setting data: {key}={value}")
        st.session_state.shared_data[key] = value

    def get_data(self, key):
        value = st.session_state.shared_data.get(key)
        self.logger.info(f"Retrieving data: {key}={value}")
        return value

    def clear_data(self, key):
        if key in st.session_state.shared_data:
            self.logger.info(f"Clearing data: {key}")
            del st.session_state.shared_data[key]

    def export_state(self):
        return st.session_state.shared_data.copy()

    def import_state(self, state):
        st.session_state.shared_data = state.copy()
        self.logger.info("Imported data state")