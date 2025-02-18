import os
from pathlib import Path

import streamlit as st
import toml
from loguru import logger


class Plugin:
    def __init__(self):
        self.logger = logger.bind(class_name=self.__class__.__name__)
        try:
            self.file = self.file if self.file is not None else __file__
            self.logger.debug(
                f"Determined plugin file path: {self.file}", file=self.file
            )
            self.default_path = False
        except AttributeError:
            self.file = __file__
            self.logger.warning(
                f"Unable to determine plugin file path, use default {self.file}",
                file=self.file,
            )
            self.default_path = True

        if self.__class__ is Plugin:
            self.logger.error("Cannot instantiate base Plugin class")
            raise RuntimeError("Base Plugin class cannot be instantiated")

        self._load_config()

    def _load_config(self):
        """Robust config loading with project-level fallback"""
        # Get project root relative to this file location
        project_root = Path(__file__).resolve().parent
        plugin_dir = (
            project_root / "plugins" / self.__class__.__name__
            if self.default_path
            else Path(os.path.dirname(self.file)).resolve()
        )
        config_dir = plugin_dir / "configs"
        config_path = config_dir / "config.toml"

        # Default values
        default_name = (
            self.__class__.__name__
            if self.default_path
            else os.path.basename(os.path.dirname(self.file))
        )
        default_version = "0.0.0"

        # Ensure directory structure exists
        config_dir.mkdir(parents=True, exist_ok=True)

        # Load or create config
        if config_path.exists():
            with open(config_path, "r") as f:
                config = toml.load(f)
            self._name = config.get("name", default_name)
            self._version = config.get("version", default_version)
            self.logger.debug(f"Loaded config from {config_path}")
        else:
            self._name = default_name
            self._version = default_version
            with open(config_path, "w") as f:
                toml.dump({"name": self._name, "version": self._version}, f)
            self.logger.info(f"Created new config at {config_path}")

        self.logger.success(
            f"Initialized plugin: {self.get_name()} v{self.get_version()}"
        )

    def get_name(self):
        return self._name

    def get_version(self):
        return self._version

    def run(self, data_manager, widget_manager):
        pass

    def create_widget(
        self,
        widget_manager,
        widget_type,
        widget_name,
        default_value=None,
        value_param="value",
        args=(),
        kwargs={},
        value_serializer=lambda x: x,
        value_deserializer=lambda x: x,
    ):
        """
        Universal widget creator with state persistence.

        This function creates a widget of the specified type and persists its state.
        It loads the saved state, prepares the widget parameters, creates the widget,
        serializes and saves the widget value if it changes, and finally returns the widget value.

        Parameters
        ----------
        - widget_manager: An instance of the widget manager responsible for state persistence.
        - widget_type: The type of the widget to be created.
        - widget_name: The name of the widget.
        - default_value: The default value of the widget.
        - value_param (optional): The parameter name for the widget value. Default is 'value'.
        - args (optional): Additional positional arguments for the widget creation. Default is an empty tuple.
        - kwargs (optional): Additional keyword arguments for the widget creation. Default is an empty dictionary.
        - value_serializer (optional): A function to serialize the widget value. Default is a lambda function that returns the input value.
        - value_deserializer (optional): A function to deserialize the saved widget value. Default is a lambda function that returns the input value.

        Returns
        -------
        The created and potentially updated widget value.
        """
        full_key = f"{self.get_name()}_{widget_name}"

        # Load and deserialize saved state
        saved_raw_value = widget_manager.load_widget_state(
            self.get_name(), widget_name, default_value
        )
        saved_value = value_deserializer(saved_raw_value)

        # Prepare widget parameters
        if value_param is not None:
            widget_params = {**kwargs, value_param: saved_value, "key": full_key}
        else:
            widget_params = {**kwargs, "key": full_key}
        # Create widget
        widget_value = widget_type(*args, **widget_params)

        # Serialize and save if value changes
        serialized_value = value_serializer(widget_value)
        if serialized_value != saved_raw_value:
            widget_manager.save_widget_state(
                self.get_name(), widget_name, serialized_value
            )
            st.rerun()

        return widget_value
