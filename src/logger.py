"""Logging manager for AI Document Assistant.

This module provides a centralized logging system with configurable log levels,
multiple output destinations (console and file), and structured logging format.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from src.runtime_paths import LOG_DIR, ensure_runtime_directories


class LoggerManager:
    """Centralized logging manager for the AI Document Assistant application.

    Features:
        - Multiple log handlers (console + file)
        - Configurable log levels
        - Structured logging with timestamps
        - Automatic log rotation
        - Support for different log formats
    """

    # Log level mapping for easy configuration
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Default log format
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    VERBOSE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"

    def __init__(
        self,
        log_dir: str = str(LOG_DIR),
        log_level: str = "INFO",
        console_enabled: bool = True,
        file_enabled: bool = True,
        verbose: bool = False,
    ):
        """Initialize the logger manager.

        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_enabled: Whether to log to console
            file_enabled: Whether to log to file
            verbose: Whether to use verbose log format
        """
        ensure_runtime_directories()
        self.log_dir = Path(log_dir)
        self.log_level = self._parse_log_level(log_level)
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.verbose = verbose
        self.loggers: Dict[str, logging.Logger] = {}

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create formatters
        self.formatters = {
            "standard": logging.Formatter(self.DEFAULT_FORMAT),
            "verbose": logging.Formatter(self.VERBOSE_FORMAT),
        }

    def _parse_log_level(self, level_str: str) -> int:
        """Parse log level string to logging constant.

        Args:
            level_str: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            Logging level constant
        """
        return self.LOG_LEVELS.get(level_str.upper(), logging.INFO)

    def _create_file_handler(self, logger_name: str) -> logging.FileHandler:
        """Create a file handler for logging.

        Args:
            logger_name: Name of the logger

        Returns:
            Configured file handler
        """
        # Generate log file name with date
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{logger_name}_{today}.log"

        # Create file handler with UTF-8 encoding
        handler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8",
        )
        handler.setLevel(self.log_level)
        handler.setFormatter(self.formatters["verbose"])

        return handler

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create a console handler for logging.

        Returns:
            Configured console handler
        """
        handler = logging.StreamHandler()
        handler.setLevel(self.log_level)
        handler.setFormatter(self.formatters["standard"])

        return handler

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name.

        Args:
            name: Name of the logger (typically module name)

        Returns:
            Configured logger instance
        """
        # Return cached logger if exists
        if name in self.loggers:
            return self.loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        logger.propagate = False  # Prevent duplicate logging

        # Add handlers
        if self.console_enabled:
            console_handler = self._create_console_handler()
            logger.addHandler(console_handler)

        if self.file_enabled:
            file_handler = self._create_file_handler(name)
            logger.addHandler(file_handler)

        # Cache the logger
        self.loggers[name] = logger

        return logger

    def log_method_call(self, logger: logging.Logger, method_name: str, **kwargs):
        """Log a method call with parameters.

        Args:
            logger: Logger instance to use
            method_name: Name of the method being called
            **kwargs: Parameters passed to the method
        """
        params = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        logger.debug(f"Calling {method_name}({params})")

    def log_method_result(self, logger: logging.Logger, method_name: str, result: Any, duration: float = None):
        """Log a method result.

        Args:
            logger: Logger instance to use
            method_name: Name of the method that returned
            result: Result of the method
            duration: Execution duration in seconds (optional)
        """
        result_str = str(result)
        if len(result_str) > 100:
            result_str = result_str[:100] + "..."

        if duration is not None:
            logger.debug(f"{method_name} completed in {duration:.2f}s: {result_str}")
        else:
            logger.debug(f"{method_name} completed: {result_str}")

    def log_error(self, logger: logging.Logger, error: Exception, context: str = ""):
        """Log an error with context.

        Args:
            logger: Logger instance to use
            error: Exception that occurred
            context: Additional context about where the error occurred
        """
        if context:
            logger.error(f"Error in {context}: {error}", exc_info=True)
        else:
            logger.error(f"Error: {error}", exc_info=True)

    def log_info(self, logger: logging.Logger, message: str, **kwargs):
        """Log an informational message.

        Args:
            logger: Logger instance to use
            message: Message to log
            **kwargs: Additional context (will be logged as key=value pairs)
        """
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info(f"{message} | {context}")
        else:
            logger.info(message)

    def log_warning(self, logger: logging.Logger, message: str, **kwargs):
        """Log a warning message.

        Args:
            logger: Logger instance to use
            message: Warning message to log
            **kwargs: Additional context
        """
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.warning(f"{message} | {context}")
        else:
            logger.warning(message)

    def get_log_file_path(self, logger_name: str) -> Path:
        """Get the path to the log file for a specific logger.

        Args:
            logger_name: Name of the logger

        Returns:
            Path to the log file
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{logger_name}_{today}.log"

    def configure_global_logging(self):
        """Configure the root logger with global settings."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Add console handler
        if self.console_enabled:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)

        # Add file handler for root logger
        if self.file_enabled:
            file_handler = self._create_file_handler("app")
            root_logger.addHandler(file_handler)


# Create a global logger instance for convenience
_global_logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger from the global manager.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    return _global_logger_manager.get_logger(name)


def log_info(message: str, **kwargs):
    """Convenience function to log an informational message.

    Args:
        message: Message to log
        **kwargs: Additional context
    """
    logger = get_logger("app")
    if kwargs:
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{message} | {context}")
    else:
        logger.info(message)


def log_error(error: Exception, context: str = ""):
    """Convenience function to log an error.

    Args:
        error: Exception that occurred
        context: Additional context about where the error occurred
    """
    logger = get_logger("app")
    if context:
        logger.error(f"Error in {context}: {error}", exc_info=True)
    else:
        logger.error(f"Error: {error}", exc_info=True)


def log_warning(message: str, **kwargs):
    """Convenience function to log a warning.

    Args:
        message: Warning message to log
        **kwargs: Additional context
    """
    logger = get_logger("app")
    if kwargs:
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.warning(f"{message} | {context}")
    else:
        logger.warning(message)


def log_debug(message: str, **kwargs):
    """Convenience function to log a debug message.

    Args:
        message: Debug message to log
        **kwargs: Additional context
    """
    logger = get_logger("app")
    if kwargs:
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"{message} | {context}")
    else:
        logger.debug(message)
