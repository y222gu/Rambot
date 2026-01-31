"""
Logging configuration for the Raman control system.

Provides structured logging with file and console handlers.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.DEBUG,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure structured logging for the Raman control system.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level for file handler
        console_level: Logging level for console handler

    Returns:
        Configured root logger for the raman module
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('raman')
    logger.setLevel(log_level)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler - detailed logging with rotation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"raman_control_{timestamp}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)

    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.

    Args:
        name: Name of the module (e.g., 'serial_protocol', 'motion_thread')

    Returns:
        Logger instance for the module
    """
    return logging.getLogger(f'raman.{name}')
