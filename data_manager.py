from loguru import logger
import streamlit as st


class DataManager:
    def __init__(self):
        """
        Initialize a new instance of DataManager.

        This class manages data storage and retrieval using Streamlit's session state.
        It also logs relevant events using the Loguru logger.

        Attributes
        ----------
        - logger: A Loguru logger instance bound to the class name.
        - shared_data: A dictionary to store data in Streamlit's session state.

        Methods
        -------
        - set_data: Store a key-value pair in shared_data.
        - get_data: Retrieve a value from shared_data using a key.
        - clear_data: Remove a key-value pair from shared_data using a key.
        - export_state: Export the current state of shared_data as a dictionary.
        - import_state: Import a state dictionary into shared_data.

        """
        self.logger = logger.bind(class_name=self.__class__.__name__)
        if "shared_data" not in st.session_state:
            self.logger.debug("Initializing new shared_data session state")
            st.session_state.shared_data = {}

    def set_data(self, key, value):
        """
        Store a key-value pair in the shared data dictionary.

        This method adds or updates a key-value pair in the Streamlit session state's
        shared_data dictionary. It also logs the action using the info level.

        Parameters:
        -----------
            key: The key under which to store the value. Can be any hashable type.
            value: The value to be stored. Can be of any type.

        Returns:
        --------
            None
        """
        self.logger.info(f"Setting data: {key}={value}")
        st.session_state.shared_data[key] = value

    def get_data(self, key, default=None):
        """
        Retrieve a value from the shared data dictionary using the provided key.

        This method attempts to fetch a value from the Streamlit session state's
        shared_data dictionary using the given key. It logs the retrieval action
        using the debug level.

        Parameters:
        -----------
            key: The key to look up in the shared data dictionary. Can be any hashable type.

        Returns:
        --------
            The value associated with the given key if it exists in the shared data dictionary,
            or None if the key is not found.
        """
        value = st.session_state.shared_data.get(key, default)
        self.logger.debug(f"Retrieving data: {key}={value}")
        return value

    def clear_data(self, key):
        """
        Remove a key-value pair from the shared data dictionary.

        This method attempts to remove a key-value pair from the Streamlit session state's
        shared_data dictionary using the given key. If the key exists, it logs the removal
        action using the warning level.

        Parameters:
        -----------
            key: The key to remove from the shared data dictionary. Can be any hashable type.

        Returns:
        --------
            None
        """
        if key in st.session_state.shared_data:
            self.logger.warning(f"Clearing data: {key}")
            del st.session_state.shared_data[key]

    def export_state(self):
        """
        Export the current state of the shared data dictionary.

        This method creates and returns a copy of the current shared_data dictionary
        from the Streamlit session state. It logs the export action using the debug level.

        Returns:
        --------
            dict: A copy of the current shared_data dictionary, containing all key-value
                  pairs stored in the session state.
        """
        self.logger.debug("Exporting data state")
        return st.session_state.shared_data.copy()

    def import_state(self, state):
        """
        Import a state dictionary into the shared data dictionary.

        This method takes a state dictionary as input, copies its contents into the
        Streamlit session state's shared_data dictionary, and logs the import action
        using the info level. It also logs a success message indicating the number of
        data items imported.

        Parameters:
        -----------
            state : dict
                A dictionary containing key-value pairs to be imported into the shared data.
                The keys and values can be of any type.

        Returns:
        --------
            None
        """
        self.logger.info("Importing data state")
        st.session_state.shared_data = state.copy()
        self.logger.success(f"Imported {len(state)} data items")
