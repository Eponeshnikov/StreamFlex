import os
import sys
from loguru import logger

def get_colored_logs(lines=100, log_dir='logs'):
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

        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log")]
        if not log_files:
            return "<span style='color: yellow'>No log files available</span>"

        latest_file = max(log_files, key=os.path.getmtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.readlines()[-lines:]
            colored_lines = []
            for line in content:
                # Add color based on log level
                if "ERROR" in line:
                    colored_lines.append(f"<span style='color: #ff4b4b'>{line}</span>")
                elif "WARNING" in line:
                    colored_lines.append(f"<span style='color: #faca2b'>{line}</span>")
                elif "INFO" in line:
                    # Changed INFO to white
                    colored_lines.append(f"<span style='color: #FFFFFF'>{line}</span>")
                elif "DEBUG" in line:
                    # Changed DEBUG to light blue
                    colored_lines.append(f"<span style='color: #4DCFFF'>{line}</span>")
                else:
                    colored_lines.append(f"<span style='color: white'>{line}</span>")
            return "".join(colored_lines)
    except Exception as e:
        return f"<span style='color: red'>Error reading logs: {str(e)}</span>"
    
    
def logger_init(log_dir='logs'):
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
        f"{log_dir}" + "/{time:YYYY-MM-DD}.log",  # Now in logs folder with date pattern
        rotation="10 MB",
        retention="12 month",
        level="DEBUG",
        enqueue=True,
        compression="zip"  # Optional: compress rotated files
    )

    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
    )