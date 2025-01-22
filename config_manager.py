import logging
import os
import yaml
import json
import toml
import streamlit as st

class ConfigManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if 'configs' not in st.session_state:
            st.session_state.configs = {}

    def load_config(self, plugin_name, config_path):
        try:
            if not os.path.exists(config_path):
                self.logger.warning(f"Config file not found: {config_path}")
                return

            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith('.toml'):
                    config = toml.load(f)
                else:
                    raise ValueError("Unsupported config format")

            self.validate_config(plugin_name, config)
            st.session_state.configs[plugin_name] = config
            self.logger.info(f"Loaded config for {plugin_name}")
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")

    def validate_config(self, plugin_name, config):
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        self.logger.info(f"Validated config for {plugin_name}")

    def get_config(self, plugin_name):
        return st.session_state.configs.get(plugin_name, {})

    def export_state(self):
        return st.session_state.configs.copy()

    def import_state(self, state):
        st.session_state.configs = state.copy()
        self.logger.info("Imported config state")