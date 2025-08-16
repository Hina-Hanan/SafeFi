"""
Logging configuration and utilities for SafeFi DeFi Risk Assessment Agent.

This module provides centralized logging configuration using loguru
with support for file rotation, structured logging, and different log levels.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from ..config.settings import get_settings


class LoggerSetup:
    """
    Logger setup and configuration class.
    
    This class handles the initialization and configuration of the application
    logging system using loguru.
    """
    
    def __init__(self):
        """Initialize logger setup."""
        self.settings = get_settings()
        self._is_configured = False
    
    def setup(self) -> None:
        """
        Setup logger configuration.
        
        Configures loguru with file rotation, appropriate log levels,
        and structured formatting.
        """
        if self._is_configured:
            return
        
        # Remove default handler
        logger.remove()
        
        # Console handler
        logger.add(
            sys.stdout,
            level=self.settings.log_level,
            format=self._get_console_format(),
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File handler
        log_path = Path(self.settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            level=self.settings.log_level,
            format=self._get_file_format(),
            rotation=self.settings.log_rotation,
            retention=self.settings.log_retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Error file handler
        error_log_path = log_path.parent / "error.log"
        logger.add(
            str(error_log_path),
            level="ERROR",
            format=self._get_file_format(),
            rotation=self.settings.log_rotation,
            retention=self.settings.log_retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        self._is_configured = True
        logger.info("Logger configuration completed")
    
    def _get_console_format(self) -> str:
        """
        Get console log format.
        
        Returns:
            Console log format string
        """
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    def _get_file_format(self) -> str:
        """
        Get file log format.
        
        Returns:
            File log format string
        """
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    def add_custom_handler(
        self,
        sink: str,
        level: str = "INFO",
        format_string: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Add custom log handler.
        
        Args:
            sink: Log sink (file path, function, etc.)
            level: Log level for this handler
            format_string: Custom format string
            **kwargs: Additional loguru.add() parameters
        """
        if format_string is None:
            format_string = self._get_file_format()
        
        logger.add(
            sink,
            level=level,
            format=format_string,
            **kwargs
        )
        
        logger.info(f"Added custom log handler: {sink}")


# Global logger setup instance
_logger_setup = LoggerSetup()


def setup_logger() -> None:
    """Setup application logger."""
    _logger_setup.setup()


def get_logger(name: Optional[str] = None) -> Any:
    """
    Get logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if not _logger_setup._is_configured:
        setup_logger()
    
    if name:
        return logger.bind(name=name)
    return logger


def log_function_call(func_name: str, args: Dict[str, Any], result: Any = None) -> None:
    """
    Log function call with arguments and result.
    
    Args:
        func_name: Function name
        args: Function arguments
        result: Function result (optional)
    """
    logger.debug(f"Function call: {func_name}", extra={"args": args, "result": result})


def log_api_request(endpoint: str, method: str, status_code: int, response_time: float) -> None:
    """
    Log API request details.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status_code: Response status code
        response_time: Response time in seconds
    """
    logger.info(
        f"API Request: {method} {endpoint}",
        extra={
            "status_code": status_code,
            "response_time": response_time,
            "endpoint": endpoint,
            "method": method
        }
    )


def log_error_with_context(error: Exception, context: Dict[str, Any]) -> None:
    """
    Log error with additional context.
    
    Args:
        error: Exception instance
        context: Additional context information
    """
    logger.error(
        f"Error occurred: {str(error)}",
        extra={"error_type": type(error).__name__, "context": context}
    )
