import hashlib
import os
import pickle
import sys
import numbers
import ast
from loguru import logger


def generate_unique_filename(plugin_name, data, *args, **kwargs):
    """
    Generate a unique filename by combining the plugin_name with a hash of the serialized data.

    Parameters:
        plugin_name (str): The name of the plugin.
        data: The data to be serialized and hashed.

    Returns:
        str: A unique filename in the format "plugin_name_hash.pickle".
    """
    combined_data = (plugin_name, data, args, kwargs)
    serialized_data = pickle.dumps(combined_data)
    hash_object = hashlib.sha256(serialized_data)
    filename = f"{plugin_name}^" + hash_object.hexdigest()
    return filename


def save_to_pickle(data, filename, folder="cache"):
    """
    Save data to a pickle file and return the file path.

    Parameters:
        data: The data to be saved.
        filename (str): The name of the pickle file.
        folder (str): The folder where the pickle file will be saved.

    Returns:
        str: The full path to the saved pickle file.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(folder, filename)  # type: ignore

    # Save the data to the pickle file
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    return file_path


# Helper function to safely parse values, especially for None and numbers
def safe_literal_eval(value_str, expected_type=None, allow_none=False):
    """
    Safely evaluate a string literal, handling None and basic types.

    Parameters:
        value_str (str): The string to be evaluated. It can represent a literal value like a number, string, or None.
        expected_type (str, optional): The expected type of the evaluated value.
            Can be "int", "float", or "str". If provided, the function will enforce type checking.
        allow_none (bool, optional): Whether to allow the string "None" to be evaluated as None.
            If False, a ValueError will be raised if "None" is encountered.

    Returns:
        The evaluated value, which can be an int, float, str, list, or None, depending on the input.

    Raises:
        ValueError: If the input string cannot be evaluated, or if the evaluated value does not match the expected type,
                   or if "None" is encountered but `allow_none` is False.
    """
    # self.logger.debug(f"Attempting safe_literal_eval on '{value_str}' (expected: {expected_type}, allow_none: {allow_none})") # Cannot log here as it's a global function
    try:
        # Handle direct None string
        if isinstance(value_str, str) and value_str.strip().lower() == "none":
            if allow_none:
                # self.logger.debug("Evaluated 'None' string as None.")
                return None
            else:
                # self.logger.warning(f"Disallowed 'None' string encountered for value '{value_str}'.")
                raise ValueError("None is not allowed for this parameter.")

        # Evaluate other literals using ast.literal_eval, which safely evaluates strings to Python literals
        val = ast.literal_eval(value_str)
        # self.logger.debug(f"ast.literal_eval result: {val} (type: {type(val)})")

        # Type checking for single values (won't apply directly to list strings)
        if expected_type == "int" and not isinstance(val, int):
            # Special case: Check if it's a list where evaluation happened
            if not isinstance(val, list):
                # self.logger.warning(f"Type mismatch: Expected int, got {type(val)} for '{value_str}'.")
                raise ValueError(f"Expected an integer, got {type(val)}")
        if expected_type == "float" and not isinstance(
            val, numbers.Number
        ):  # Allow int to be treated as float
            # Special case: Check if it's a list where evaluation happened
            if not isinstance(val, list):
                # self.logger.warning(f"Type mismatch: Expected float, got {type(val)} for '{value_str}'.")
                raise ValueError(f"Expected a float, got {type(val)}")
        if expected_type == "str" and not isinstance(val, str):
            # Special case: Check if it's a list where evaluation happened
            if not isinstance(val, list):
                # self.logger.warning(f"Type mismatch: Expected str, got {type(val)} for '{value_str}'.")
                raise ValueError(f"Expected a string, got {type(val)}")

        # Check for None if not allowed (after evaluation)
        if val is None and not allow_none:
            # self.logger.warning(f"Disallowed None value encountered after evaluation for '{value_str}'.")
            raise ValueError("None is not allowed for this parameter.")

        # self.logger.debug(f"Successfully evaluated '{value_str}' to: {val}")
        return val
    except (ValueError, SyntaxError, TypeError) as e:
        # self.logger.error(f"Evaluation failed for '{value_str}': {e}")
        raise ValueError(f"Invalid input format '{value_str}': {e}")


def calculate_eta(total_work, completed_work, time_elapsed):
    """
    Calculate the estimated time remaining (eta) in seconds.

    Parameters:
        total_work (float or int): Total amount of work to be done (e.g., bytes, tasks).
        completed_work (float or int): Amount of work already completed.
        time_elapsed (float): Time elapsed so far in seconds.

    Returns:
        float: Estimated time remaining in seconds, or 0.0 if work is complete.
        None: If eta cannot be estimated due to insufficient data.

    Raises:
        ValueError: If total_work is not positive, or if completed_work or time_elapsed is negative.
    """
    # Input validation
    if total_work <= 0:
        raise ValueError("total_work must be positive")
    if completed_work < 0:
        raise ValueError("completed_work cannot be negative")
    if time_elapsed < 0:
        raise ValueError("time_elapsed cannot be negative")

    # If work is complete or overdone, no time remains
    if completed_work >= total_work:
        return 0.0

    # If no work is done or no time has elapsed, eta cannot be estimated
    if completed_work == 0 or time_elapsed == 0:
        return "Estimated..."

    # Calculate eta: time_elapsed * (remaining_work / completed_work)
    return time_elapsed * (total_work - completed_work) / completed_work


def get_colored_logs(lines=100, log_dir="logs"):
    """
    Retrieve logs and add color based on log level.

    This function reads the most recent log file from the specified directory,
    retrieves the specified number of lines, and applies HTML color formatting
    based on the log level of each line.
    Parameters:
    -----------
    lines : int, optional
        The number of lines to retrieve from the log files. Default is 100.
    log_dir : str, optional
        The directory where log files are stored. Default is 'logs'.

    Returns:
    --------
    str
        A string containing the colored log lines in HTML format.
        If the logs directory is not found, returns an error message in red.
        If no log files are available, returns a warning message in yellow.
        If an error occurs during processing, returns an error message in red.
    """
    try:
        if not os.path.exists(log_dir):
            return "<span style='color: red'>Logs directory not found</span>"

        log_files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.endswith(".log")
        ]
        if not log_files:
            return "<span style='color: yellow'>No log files available</span>"

        latest_file = max(log_files, key=os.path.getmtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.readlines()[-lines:]
            colored_lines = []
            for line in content:
                # Add color based on log level
                if "ERROR" in line:
                    colored_lines.append(
                        f"<span style='color: #ff4b4b'>{line}</span>"
                    )
                elif "WARNING" in line:
                    colored_lines.append(
                        f"<span style='color: #faca2b'>{line}</span>"
                    )
                elif "INFO" in line:
                    # Changed INFO to white
                    colored_lines.append(
                        f"<span style='color: #FFFFFF'>{line}</span>"
                    )
                elif "DEBUG" in line:
                    # Changed DEBUG to light blue
                    colored_lines.append(
                        f"<span style='color: #4DCFFF'>{line}</span>"
                    )
                else:
                    colored_lines.append(
                        f"<span style='color: white'>{line}</span>"
                    )
            return "".join(colored_lines)
    except Exception as e:
        return f"<span style='color: red'>Error reading logs: {str(e)}</span>"


def logger_init(log_dir="logs"):
    """
    Initialize and configure the logger for both file and console output.

    This function sets up logging to both a file and the console. It creates
    a log directory if it doesn't exist, configures file logging with rotation
    and retention policies, and sets up console logging with color output.

    Parameters:
    -----------
    log_dir : str, optional
        The directory where log files will be stored. Default is 'logs'.

    Returns:
    --------
    None
    """
    # Create logs directory if not exists
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()

    # Configure file logging
    logger.add(
        f"{log_dir}"
        + "/{time:YYYY-MM-DD}.log",  # Now in logs folder with date pattern
        rotation="00:00",
        retention="1 week",
        level="DEBUG",
        enqueue=True,
        compression="zip",  # Optional: compress rotated files
    )

    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
    )


CACHE_FOLDER = ".cache"


def generate_cache_filename(func, *args, **kwargs):
    """
    Generate a unique cache filename based on the function name, arguments, and keyword arguments.

    This function takes a function object, positional arguments, and keyword arguments as input.
    It combines the function name, arguments, and keyword arguments into a tuple, serializes the tuple,
    and computes the SHA-256 hash of the serialized data. The hash is then used to generate a unique
    filename with a '.pickle' extension. The filename is returned as a string.

    Parameters
    ----------
    func (function): The function object for which the cache filename is being generated.
    *args (tuple): Positional arguments passed to the function.
    **kwargs (dict): Keyword arguments passed to the function.

    Returns:
    str: A unique cache filename based on the function name, arguments, and keyword arguments.
    """
    combined_data = (func.__name__, args, kwargs)
    serialized_data = pickle.dumps(combined_data)
    hash_object = hashlib.sha256(serialized_data)
    filename = f"{func.__name__}^" + hash_object.hexdigest() + ".pickle"
    return os.path.join(CACHE_FOLDER, filename)


def cache_result(reset=False):
    """
    A decorator function that caches the results of a function and stores them in a cache file.

    Parameters
    ----------
    reset (bool): If True, the cache file will be deleted and the function will be executed again.
                  If False (default), the function will attempt to load the result from the cache file.

    Returns:
    function: The decorated function, which will either return the cached result or execute the function
              and store the result in the cache file.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_file = generate_cache_filename(func, *args, **kwargs)
            if not reset:
                try:
                    with open(cache_file, "rb") as file:
                        cached_data = pickle.load(file)
                    cached_result = cached_data
                    return cached_result
                except (IOError, pickle.PickleError, EOFError):
                    pass
            result = func(*args, **kwargs)
            cached_data = result
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            with open(cache_file, "wb") as file:
                pickle.dump(cached_data, file)

            return result

        return wrapper

    return decorator
