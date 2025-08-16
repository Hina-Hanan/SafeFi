"""
Blockchain client for SafeFi DeFi Risk Assessment Agent.

This module provides Web3 interaction capabilities for reading blockchain data.
"""

import asyncio
from typing import Optional, Dict, Any
from web3 import Web3, HTTPProvider
from loguru import logger

from ..config.settings import get_settings
from ..utils.logger import get_logger, log_error_with_context


class BlockchainClient:
    """
    Blockchain client interacting with Ethereum and compatible chains.
    """

    def __init__(self, provider_url: Optional[str] = None):
        """Initialize blockchain client."""
        self.settings = get_settings()
        self.logger = get_logger("BlockchainClient")
        self.provider_url = provider_url or f"https://eth-mainnet.g.alchemy.com/v2/{self.settings.alchemy_api_key}"
        
        try:
            self.web3 = Web3(HTTPProvider(self.provider_url))
            if not self.web3.is_connected():
                self.logger.error(f"Unable to connect to blockchain at {self.provider_url}")
            else:
                self.logger.info(f"Connected to blockchain at {self.provider_url}")
        except Exception as e:
            log_error_with_context(e, {"provider_url": self.provider_url})
            raise

    def get_latest_block_number(self) -> int:
        """Return the latest block number."""
        try:
            block_number = self.web3.eth.block_number
            self.logger.debug(f"Latest block number: {block_number}")
            return block_number
        except Exception as e:
            log_error_with_context(e, {})
            return -1

    def get_block(self, block_number: int) -> Optional[Dict[str, Any]]:
        """Return a block's data given its number."""
        try:
            block = self.web3.eth.get_block(block_number)
            self.logger.debug(f"Retrieved block {block_number}")
            return dict(block)
        except Exception as e:
            log_error_with_context(e, {"block_number": block_number})
            return None

    def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Return transaction data by hash."""
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            self.logger.debug(f"Retrieved transaction {tx_hash}")
            return dict(tx)
        except Exception as e:
            log_error_with_context(e, {"tx_hash": tx_hash})
            return None

    def is_connected(self) -> bool:
        """Return connection status."""
        return self.web3.is_connected()
