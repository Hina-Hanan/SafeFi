"""
API client for SafeFi DeFi Risk Assessment Agent.

This module provides a unified API client for interacting with various
DeFi data sources and external APIs without wallet connections.
"""

import asyncio
import time
from typing import Any, Dict, Optional, List
from datetime import datetime
import aiohttp
from loguru import logger

from ..config.settings import get_settings
from ..utils.logger import get_logger, log_api_request, log_error_with_context
from ..utils.validators import validate_api_response


class APIClient:
    """
    Unified API client for external data sources.
    
    This class provides a centralized interface for making API requests
    to various DeFi data providers with rate limiting, error handling,
    and retry logic.
    """
    
    def __init__(self):
        """Initialize API client."""
        self.settings = get_settings()
        self.logger = get_logger("APIClient")
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = {}
        self.endpoints = self.settings.get_api_endpoints()
        
        # Initialize rate limiters for each API
        for api_name in self.endpoints.keys():
            self.rate_limiter[api_name] = {
                "requests": [],
                "limit": self.settings.api_rate_limit
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self) -> None:
        """Start aiohttp session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.settings.api_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info("API client session started")
    
    async def close_session(self) -> None:
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("API client session closed")
    
    async def _check_rate_limit(self, api_name: str) -> None:
        """Check and enforce rate limiting."""
        if api_name not in self.rate_limiter:
            raise ValueError(f"Unknown API: {api_name}")

        current_time = time.time()
        rate_info = self.rate_limiter[api_name]
    
        # Initialize keys if missing to avoid KeyError
        if "last_request" not in rate_info:
            rate_info["last_request"] = 0
        if "min_interval" not in rate_info:
            rate_info["min_interval"] = 1

        if current_time - rate_info["last_request"] < rate_info["min_interval"]:
            sleep_time = rate_info["min_interval"] - (current_time - rate_info["last_request"])
            await asyncio.sleep(sleep_time)

        rate_info["last_request"] = current_time

    
    async def make_request(
        self,
        api_name: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting and error handling.
        
        Args:
            api_name: Name of the API (coingecko, etherscan, etc.)
            endpoint: API endpoint path
            method: HTTP method
            params: Query parameters
            headers: Request headers
            retries: Number of retry attempts
            
        Returns:
            API response data
            
        Raises:
            aiohttp.ClientError: If request fails after retries
        """
        if not self.session:
            await self.start_session()
        
        # Check rate limiting
        await self._check_rate_limit(api_name)
        
        # Build full URL
        base_url = self.endpoints.get(api_name)
        if not base_url:
            raise ValueError(f"Unknown API: {api_name}")
        
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Add API key to headers/params based on API type
        request_headers = headers or {}
        request_params = params or {}
        
        if api_name == "coingecko" and self.settings.coingecko_api_key:
            request_headers["x-cg-demo-api-key"] = self.settings.coingecko_api_key
        
        elif api_name == "etherscan" and self.settings.etherscan_api_key:
            request_params["apikey"] = self.settings.etherscan_api_key
        
        elif api_name == "alchemy" and self.settings.alchemy_api_key:
            request_headers["Authorization"] = f"Bearer {self.settings.alchemy_api_key}"
        
        elif api_name == "moralis" and self.settings.moralis_api_key:
            request_headers["X-API-Key"] = self.settings.moralis_api_key
        
        elif api_name == "quicknode" and self.settings.quicknode_api_key:
            request_headers["Authorization"] = f"Bearer {self.settings.quicknode_api_key}"
        
        elif api_name == "huggingface" and self.settings.huggingface_api_key:
            request_headers["Authorization"] = f"Bearer {self.settings.huggingface_api_key}"
        
        start_time = time.time()
        
        for attempt in range(retries + 1):
            try:
                async with self.session.request(
                    method,
                    url,
                    params=request_params,
                    headers=request_headers
                ) as response:
                    response_time = time.time() - start_time
                    
                    # Log API request
                    log_api_request(url, method, response.status, response_time)
                    
                    # Handle different response codes
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"Successful {api_name} API request to {endpoint}")
                        return data
                    
                    elif response.status == 429:  # Rate limited
                        if attempt < retries:
                            wait_time = 2 ** attempt
                            self.logger.warning(f"Rate limited by {api_name}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API error {response.status}: {error_text}")
                        response.raise_for_status()
            
            except aiohttp.ClientError as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_context = {
                        "api_name": api_name,
                        "endpoint": endpoint,
                        "method": method,
                        "attempt": attempt + 1
                    }
                    log_error_with_context(e, error_context)
                    raise
        
        raise aiohttp.ClientError(f"Failed to make request after {retries + 1} attempts")
    
    async def get_defi_protocols(self) -> List[Dict[str, Any]]:
        """
        Get list of DeFi protocols from DeFi Llama.
        
        Returns:
            List of protocol data
        """
        try:
            data = await self.make_request("defi_llama", "protocols")
            self.logger.info(f"Retrieved {len(data)} DeFi protocols")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to get DeFi protocols: {str(e)}")
            return []
    
    async def get_protocol_tvl(self, protocol_name: str) -> Dict[str, Any]:
        """
        Get TVL data for a specific protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            Protocol TVL data
        """
        try:
            endpoint = f"protocol/{protocol_name}"
            data = await self.make_request("defi_llama", endpoint)
            self.logger.debug(f"Retrieved TVL data for {protocol_name}")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to get TVL for {protocol_name}: {str(e)}")
            return {}
    
    async def get_token_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get token price from CoinGecko.
        
        Args:
            token_id: CoinGecko token ID
            
        Returns:
            Token price data
        """
        try:
            endpoint = "simple/price"
            params = {
                "ids": token_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true"
            }
            data = await self.make_request("coingecko", endpoint, params=params)
            self.logger.debug(f"Retrieved price data for {token_id}")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to get price for {token_id}: {str(e)}")
            return {}
    
    async def get_ethereum_price(self) -> Dict[str, Any]:
        """
        Get Ethereum price from Etherscan.
        
        Returns:
            Ethereum price data
        """
        try:
            params = {
                "module": "stats",
                "action": "ethprice"
            }
            data = await self.make_request("etherscan", "", params=params)
            self.logger.debug("Retrieved ETH price from Etherscan")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to get ETH price: {str(e)}")
            return {}
    
    async def get_alchemy_data(self, query: str) -> Dict[str, Any]:
        """
        Get data from Alchemy subgraphs.
        
        Args:
            query: GraphQL query string
            
        Returns:
            Subgraph data
        """
        try:
            endpoint = "subgraphs/name/uniswap/uniswap-v3"
            headers = {"Content-Type": "application/json"}
            data = await self.make_request(
                "alchemy", 
                endpoint,
                method="POST",
                headers=headers,
                params={"query": query}
            )
            self.logger.debug("Retrieved data from Alchemy subgraphs")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to get Alchemy data: {str(e)}")
            return {}
    
    async def get_moralis_defi_data(self, protocol: str) -> Dict[str, Any]:
        """
        Get DeFi protocol data from Moralis.
        
        Args:
            protocol: Protocol name
            
        Returns:
            Protocol data from Moralis
        """
        try:
            endpoint = f"defi/{protocol}"
            data = await self.make_request("moralis", endpoint)
            self.logger.debug(f"Retrieved {protocol} data from Moralis")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to get Moralis data: {str(e)}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all API endpoints.
        
        Returns:
            Health check results
        """
        results = {}
        
        # Test CoinGecko
        try:
            await self.get_token_price("bitcoin")
            results["coingecko"] = "healthy"
        except Exception as e:
            results["coingecko"] = f"unhealthy: {str(e)}"
        
        # Test DeFiLlama
        try:
            protocols = await self.get_defi_protocols()
            if len(protocols) > 0:
                results["defi_llama"] = "healthy"
            else:
                results["defi_llama"] = "unhealthy: no data returned"
        except Exception as e:
            results["defi_llama"] = f"unhealthy: {str(e)}"
        
        # Test Etherscan
        try:
            await self.get_ethereum_price()
            results["etherscan"] = "healthy"
        except Exception as e:
            results["etherscan"] = f"unhealthy: {str(e)}"
        
        return results
