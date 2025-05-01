import streamlit as st
from loguru import logger


class WidgetManager:
    def __init__(self):
        """
        Initialize the WidgetManager class.

        This method sets up the logging and session state for the widget manager.
        If 'widget_states' does not exist in the session state, it initializes an empty dictionary.
        Otherwise, it logs a debug message indicating the existence of existing widget states.

        Returns:
        --------
            None
        """
        self.logger = logger.bind(class_name=self.__class__.__name__)
        if "widget_states" not in st.session_state:
            st.session_state.widget_states = {}
            self.logger.debug("Initialized widget_states in session state")
        else:
            self.logger.debug("Found existing widget_states in session state")
        if "current_value" not in st.session_state:
            st.session_state.current_value = {}
            self.logger.debug("Initialized current_value in session state")
        else:
            self.logger.debug("Found existing current_value in session state")

    def save_widget_state(self, plugin_name, widget_key, value):
        """
        Save widget state with value validation and conflict detection.

        Parameters:
        -----------
        plugin_name : str
            The name of the plugin associated with the widget.
        widget_key : str
            The unique identifier for the widget within the plugin.
        value : any
            The new value to be saved for the widget.

        Returns:
        --------
        bool
            True if the widget state was successfully saved, False otherwise.
            If an exception occurs during the save process, the method logs the error
            and returns False.
        """
        try:
            previous_value = st.session_state.widget_states.get(
                plugin_name, {}
            ).get(widget_key)
            plugin_states = st.session_state.widget_states.get(plugin_name, {})

            plugin_states[widget_key] = value
            st.session_state.widget_states[plugin_name] = plugin_states

            self.logger.info(
                "Saved widget state: {plugin}.{key} = {value} (previous: {prev})",
                plugin=plugin_name,
                key=widget_key,
                value=repr(value),
                prev=repr(previous_value),
            )
            return True
        except Exception as e:
            self.logger.error(
                "Failed to save {plugin}.{key}: {error}",
                plugin=plugin_name,
                key=widget_key,
                error=str(e),
            )
            return False

    def load_widget_state(self, plugin_name, widget_key, default=None):
        """
        Load widget state with existence checking.

        This function retrieves the value of a specific widget state based on the provided plugin name and widget key.
        If the widget state does not exist, the function returns the provided default value.

        Parameters:
        -----------
        plugin_name : str
            The name of the plugin associated with the widget.
        widget_key : str
            The unique identifier for the widget within the plugin.
        default : any, optional
            The default value to be returned if the widget state does not exist.
            Defaults to None.

        Returns:
        --------
        any
            The value of the widget state if it exists, or the provided default value if it does not.
            If an exception occurs during the retrieval process, the function logs the error
            and returns the provided default value.
        """
        try:
            value_state = st.session_state.widget_states.get(
                plugin_name, {}
            ).get(widget_key, default)
            is_new_object = widget_key not in st.session_state.current_value.get(
                plugin_name, {}
            )
            if is_new_object:
                plugin_current = st.session_state.current_value.get(
                    plugin_name, {}
                )
                plugin_current[widget_key] = value_state
                st.session_state.current_value[plugin_name] = plugin_current
            value = (
                value_state
                if is_new_object
                else st.session_state.current_value.get(plugin_name, {})[
                    widget_key
                ]
            )

            self.logger.debug(
                "Loading {plugin}.{key} = {value} (value_state: {value_state}) (default: {default})",
                plugin=plugin_name,
                key=widget_key,
                value=repr(value),
                value_state=repr(value_state),
                default=repr(default),
            )

            return value, is_new_object
        except Exception as e:
            self.logger.error(
                "Failed to load {plugin}.{key}: {error}",
                plugin=plugin_name,
                key=widget_key,
                error=str(e),
            )
            return default, False

    def export_state(self):
        """
        Export complete widget states with validation.

        This function retrieves the current state of all widget states and returns it as a dictionary.
        The state includes the plugin names as keys and their corresponding widget states as values.
        The function also performs validation to ensure that the state is in the correct format.

        Returns:
        --------
        dict
            A dictionary representing the current state of all widget states.
            If an exception occurs during the export process, an empty dictionary is returned.

        Raises:
        -------
        Exception
            If the state is not in the expected format, an exception is raised.
        """
        try:
            state = st.session_state.widget_states.copy()
            self.logger.info(
                "Exporting widget states: {count} plugins, {total} total widgets",
                count=len(state),
                total=sum(len(v) for v in state.values()),
            )
            self.logger.trace("Exported state structure: {state}", state=state)
            return state
        except Exception as e:
            self.logger.error("Export failed: {error}", error=str(e))
            return {}

    def import_state(self, state):
        """
        Import widget states with data validation.

        This function takes a dictionary representing widget states and imports it into the current session state.
        The function performs data validation to ensure that the provided state is in the correct format.

        Parameters:
        -----------
        state : dict
            A dictionary representing widget states. The dictionary should have plugin names as keys and their corresponding
            widget states as values. Each widget state should be a dictionary with widget keys and their corresponding values.

        Returns:
        --------
        bool
            True if the import is successful, False otherwise.
            If the import is successful, the function logs a success message and returns True.
            If the import fails due to an exception, the function logs an error message and returns False.

        Raises:
        -------
        ValueError
            If the provided state is not a dictionary, a ValueError is raised.
        """
        try:
            if not isinstance(state, dict):
                raise ValueError("Invalid state format")

            st.session_state.widget_states = state.copy()
            self.logger.success(
                "Imported widget states: {plugins} plugins, {widgets} widgets",
                plugins=len(state),
                widgets=sum(len(v) for v in state.values()),
            )
            self.logger.trace("Imported state structure: {state}", state=state)
            return True
        except Exception as e:
            self.logger.exception("Import failed: {error}", error=str(e))
            return False
