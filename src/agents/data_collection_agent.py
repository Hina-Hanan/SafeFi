
"""
Data Collection Agent for SafeFi DeFi Risk Assessment Agent.

This agent manages multi-source data gathering from DeFi APIs and blockchain.
"""

import asyncio
from typing import Any, Dict, Optional
from loguru import logger

from ..agents.base_agent import BaseAgent
from ..data_collection.api_client import APIClient
from ..data_collection.blockchain_client import BlockchainClient


class DataCollectionAgent(BaseAgent):
    """
    Data collection agent.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_name="DataCollectionAgent", config=config)
        self.api_client = APIClient()
        self.blockchain_client = BlockchainClient()

    async def validate_input(self, **kwargs) -> bool:
        # No input parameters needed for this example
        return True

    async def execute(self, **kwargs) -> Dict[str, Any]:
        self.logger.info("Executing data collection")
        results = {}
        # Collect protocol list
        try:
            protocols = await self.api_client.get_defi_protocols()
            results["protocols"] = protocols
            self.logger.info(f"Collected {len(protocols)} protocols")
        except Exception as e:
            self.logger.error(f"Failed to collect protocols: {str(e)}")
            results["protocols"] = []

        # Get latest block number
        try:
            block_number = self.blockchain_client.get_latest_block_number()
            results["latest_block"] = block_number
            self.logger.info(f"Latest blockchain block number {block_number}")
        except Exception as e:
            self.logger.error(f"Failed to get latest blockchain block: {str(e)}")
            results["latest_block"] = None

        return results

    async def on_start(self) -> None:
        self.logger.info("DataCollectionAgent starting up.")

    async def on_stop(self) -> None:
        self.logger.info("DataCollectionAgent shutting down.")
