"""
Settings and configuration management for SafeFi DeFi Risk Assessment Agent.

This module handles all application settings, environment variables,
and configuration validation using Pydantic.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field
from loguru import logger


class Settings(BaseSettings):
    """
    Application settings and configuration.
    
    This class manages all configuration settings for the SafeFi application,
    including API keys, database connections, and logging configuration.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = Field(default="SafeFi DeFi Risk Assessment Agent", description="Application name")
    debug: bool = Field(default=False, description="Debug mode flag")
    environment: str = Field(default="development", description="Application environment")
    
    # API Keys - Updated for wallet-free APIs
    coingecko_api_key: Optional[str] = Field(default=None, description="CoinGecko API key")
    etherscan_api_key: Optional[str] = Field(default=None, description="Etherscan API key")
    defi_llama_api_key: Optional[str] = Field(default="not_required", description="DeFi Llama API key")
    alchemy_api_key: Optional[str] = Field(default=None, description="Alchemy API key")
    moralis_api_key: Optional[str] = Field(default=None, description="Moralis API key")
    quicknode_api_key: Optional[str] = Field(default=None, description="QuickNode API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="Hugging Face API key")
    
    # POSTGRESQL DATABASE CONFIGURATION
    database_url: str = Field(
        default="postgresql+asyncpg://safefi_user:password@localhost:5432/safefi_db",
        description="PostgreSQL database connection URL"
    )
    db_host: str = Field(default="127.0.0.1", description="Database host")  # Changed to IP
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="safefi_db", description="Database name")
    db_user: str = Field(default="safefi_user", description="Database user")
    db_password: str = Field(default="password", description="Database password")
    
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
    
    @field_validator("environment")
    @classmethod
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
    
    @field_validator("log_level")
    @classmethod
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
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """
        Validate PostgreSQL database URL format.
        
        Args:
            v: Database URL value
            
        Returns:
            Validated database URL
            
        Raises:
            ValueError: If database URL is not PostgreSQL format
        """
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("Database URL must be PostgreSQL format (postgresql:// or postgresql+asyncpg://)")
        return v
    
    def get_api_endpoints(self) -> Dict[str, str]:
        """
        Get API endpoints configuration for wallet-free APIs.
        
        Returns:
            Dictionary of API endpoints
        """
        return {
            "coingecko": "https://api.coingecko.com/api/v3",
            "etherscan": "https://api.etherscan.io/api",
            "defi_llama": "https://api.llama.fi",
            "alchemy": "https://eth-mainnet.g.alchemy.com/v2",
            "moralis": "https://deep-index.moralis.io/api/v2",
            "quicknode": "https://your-quicknode-endpoint.com",
            "huggingface": "https://api-inference.huggingface.co"
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get PostgreSQL database configuration.
        
        Returns:
            Dictionary of database configuration
        """
        return {
            "url": self.database_url,
            "echo": self.debug,
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": 20,
            "max_overflow": 10
        }
    
    def get_postgres_dsn(self) -> str:
        """
        Get PostgreSQL DSN for asyncpg.
        
        Returns:
            PostgreSQL connection string for asyncpg
        """
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")
    
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
