import os
import sys
from loguru import logger

def get_colored_logs(lines=100):
    """Retrieve logs and add color based on log level"""
    log_dir = "logs"
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
    
    
def logger_init():
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)

    logger.remove()

    # Configure file logging
    logger.add(
        "logs/{time:YYYY-MM-DD}.log",  # Now in logs folder with date pattern
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