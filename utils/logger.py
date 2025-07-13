import logging
import sys
import os
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Purple
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logger(name='DBNet', log_dir='logs'):
    """
    Set up logger with both file and console output
    
    Args:
        name (str): Logger name
        log_dir (str): Directory to store log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # File handler - detailed logging to file
    log_file = os.path.join(log_dir, f'dbnet_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - colored output for better visibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create a global logger instance
logger = setup_logger()

def log_exception(e: Exception, context: str = None):
    """
    Log an exception with detailed information
    
    Args:
        e (Exception): The exception to log
        context (str, optional): Additional context about where/why the error occurred
    """
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    
    error_details = {
        'type': type(e).__name__,
        'message': str(e),
        'file': tb[-1].filename,
        'line': tb[-1].lineno,
        'function': tb[-1].name,
        'context': context
    }
    
    error_msg = f"""
    {'='*80}
    Error Type: {error_details['type']}
    Message: {error_details['message']}
    Location: {error_details['file']}:{error_details['line']} in {error_details['function']}
    {'Context: ' + error_details['context'] if error_details['context'] else ''}
    {'='*80}
    Full traceback:
    {''.join(traceback.format_tb(e.__traceback__))}
    """
    
    logger.error(error_msg) 