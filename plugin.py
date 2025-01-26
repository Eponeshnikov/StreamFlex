import logging
from abc import ABC, abstractmethod
import streamlit as st

class Plugin(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def run(self, data_manager, config_manager, widget_manager):
        pass
    
    @abstractmethod
    def get_name(self):
        pass
    
    @abstractmethod
    def get_version(self):
        pass
    
    def create_widget(self, widget_manager, widget_type, widget_name, 
                     default_value, value_param='value', args=(), kwargs={},
                     value_serializer=lambda x: x, value_deserializer=lambda x: x):
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
            self.get_name(), 
            widget_name, 
            default_value
        )
        saved_value = value_deserializer(saved_raw_value)

        # Prepare widget parameters
        widget_params = {
            **kwargs,
            value_param: saved_value,
            "key": full_key
        }
        # Create widget
        widget_value = widget_type(*args, **widget_params)

        # Serialize and save if value changes
        serialized_value = value_serializer(widget_value)
        if serialized_value != saved_raw_value:
            widget_manager.save_widget_state(
                self.get_name(), 
                widget_name, 
                serialized_value
            )
            st.rerun()

        return widget_value