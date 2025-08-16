"""
Settings and configuration management for SafeFi DeFi Risk Assessment Agent.

This module handles all application settings, environment variables,
and configuration validation using Pydantic.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from functools import lru_cache

from pydantic import BaseSettings, validator, Field
from loguru import logger


class Settings(BaseSettings):
    """
    Application settings and configuration.
    
    This class manages all configuration settings for the SafeFi application,
    including API keys, database connections, and logging configuration.
    """
    
    # Application Settings
    app_name: str = Field(default="SafeFi DeFi Risk Assessment Agent", description="Application name")
    debug: bool = Field(default=False, description="Debug mode flag")
    environment: str = Field(default="development", description="Application environment")
    
    # API Keys
    defi_llama_api_key: Optional[str] = Field(default=None, description="DeFi Llama API key")
    moralis_api_key: Optional[str] = Field(default=None, description="Moralis API key")
    alchemy_api_key: Optional[str] = Field(default=None, description="Alchemy API key")
    coingecko_api_key: Optional[str] = Field(default=None, description="CoinGecko API key")
    etherscan_api_key: Optional[str] = Field(default=None, description="Etherscan API key")
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    
    # Database
    database_url: str = Field(default="sqlite:///data/safefi.db", description="Database connection URL")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/safefi.log", description="Log file path")
    log_rotation: str = Field(default="1 day", description="Log rotation period")
    log_retention: str = Field(default="30 days", description="Log retention period")
    
    # API Rate Limiting
    api_rate_limit: int = Field(default=100, description="API requests per minute")
    api_timeout: int = Field(default=30, description="API request timeout in seconds")
    
    # Data Collection
    data_collection_interval: int = Field(default=300, description="Data collection interval in seconds")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent API requests")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """
        Validate environment setting.
        
        Args:
            v: Environment value
            
        Returns:
            Validated environment value
            
        Raises:
            ValueError: If environment is not valid
        """
        valid_environments = ["development", "testing", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """
        Validate log level setting.
        
        Args:
            v: Log level value
            
        Returns:
            Validated log level value
            
        Raises:
            ValueError: If log level is not valid
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def get_api_endpoints(self) -> Dict[str, str]:
        """
        Get API endpoints configuration.
        
        Returns:
            Dictionary of API endpoints
        """
        return {
            "defi_llama": "https://api.llama.fi",
            "moralis": "https://deep-index.moralis.io/api/v2",
            "alchemy": "https://eth-mainnet.g.alchemy.com/v2",
            "coingecko": "https://api.coingecko.com/api/v3",
            "etherscan": "https://api.etherscan.io/api"
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration.
        
        Returns:
            Dictionary of database configuration
        """
        return {
            "url": self.database_url,
            "echo": self.debug,
            "pool_pre_ping": True,
            "pool_recycle": 300
        }
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            "logs",
            "data/raw",
            "data/processed",
            Path(self.log_file).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).
    
    Returns:
        Settings instance
    """
    settings = Settings()
    settings.create_directories()
    return settings
