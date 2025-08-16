"""
Main entry point for SafeFi DeFi Risk Assessment Agent.

This module provides the main application interface and CLI commands.
"""

import asyncio
import sys
from typing import Optional
import click
from loguru import logger

from .config.settings import get_settings
from .utils.logger import setup_logger
from .agents.data_collection_agent import DataCollectionAgent


async def run_data_collection():
    """Run data collection workflow."""
    logger.info("Starting data collection workflow...")
    
    agent = DataCollectionAgent()
    
    try:
        await agent.start()
        result = await agent.safe_execute()
        
        if result["success"]:
            logger.info("Data collection completed successfully")
            logger.info(f"Collected {len(result['result']['protocols'])} protocols")
            logger.info(f"Latest block: {result['result']['latest_block']}")
        else:
            logger.error(f"Data collection failed: {result['error']}")
    
    finally:
        await agent.stop()


async def health_check():
    """Perform system health check."""
    logger.info("Performing system health check...")
    
    agent = DataCollectionAgent()
    health = await agent.health_check()
    
    logger.info(f"Agent Status: {health['status']}")
    logger.info(f"Execution Count: {health['execution_count']}")
    
    # Check API health
    api_health = await agent.api_client.health_check()
    for api, status in api_health.items():
        logger.info(f"API {api}: {status}")


@click.group()
def cli():
    """SafeFi DeFi Risk Assessment Agent CLI."""
    # Setup logging
    setup_logger()
    logger.info("SafeFi DeFi Risk Assessment Agent")


@cli.command()
def collect():
    """Run data collection workflow."""
    asyncio.run(run_data_collection())


@cli.command()
def health():
    """Perform system health check."""
    asyncio.run(health_check())


@cli.command()
def version():
    """Show version information."""
    click.echo("SafeFi DeFi Risk Assessment Agent v0.1.0")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
