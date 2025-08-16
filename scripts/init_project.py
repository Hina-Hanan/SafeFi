"""
Project initialization script for SafeFi DeFi Risk Assessment Agent.

This script sets up the project environment, creates necessary directories,
and validates the installation.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directories() -> None:
    """Create necessary project directories."""
    directories = [
        "logs",
        "data/raw",
        "data/processed",
        "notebooks",
        "docs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_env_file() -> None:
    """Create .env file from .env.example if it doesn't exist."""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            # Copy .env.example to .env
            with open(".env.example", "r") as example:
                with open(".env", "w") as env_file:
                    env_file.write(example.read())
            logger.info("Created .env file from .env.example")
        else:
            logger.warning(".env.example not found, please create .env manually")
    else:
        logger.info(".env file already exists")


def install_requirements(requirement_file: str = "requirements/base.txt") -> bool:
    """Install Python requirements."""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirement_file
        ])
        logger.info(f"Successfully installed requirements from {requirement_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False


def validate_installation() -> bool:
    """Validate that key packages are installed."""
    required_packages = [
        "pandas",
        "numpy",
        "requests",
        "web3",
        "loguru",
        "pydantic",
        "aiohttp",
        "pytest"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is NOT installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    logger.info("All required packages are installed")
    return True


def main():
    """Main initialization function."""
    logger.info("Starting SafeFi project initialization...")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements. Please install manually.")
        sys.exit(1)
    
    # Validate installation
    if not validate_installation():
        logger.error("Installation validation failed.")
        sys.exit(1)
    
    logger.info("SafeFi project initialization completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Update .env file with your API keys")
    logger.info("2. Run tests: pytest tests/")
    logger.info("3. Start developing your agents!")


if __name__ == "__main__":
    main()
