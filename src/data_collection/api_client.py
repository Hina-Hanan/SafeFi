"""
API client for SafeFi DeFi Risk Assessment Agent.

This module provides a unified API client for interacting with various
DeFi data sources and external APIs.
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
        """
        Check and enforce rate limiting.
        
        Args:
            api_name: Name of the API to check
        """
        current_time = time.time()
        rate_info = self.rate_limiter[api_name]
        
        # Remove requests older than 1 minute
        rate_info["requests"] = [
            req_time for req_time in rate_info["requests"]
            if current_time - req_time < 60
        ]
        
        # Check if we've exceeded the rate limit
        if len(rate_info["requests"]) >= rate_info["limit"]:
            sleep_time = 60 - (current_time - rate_info["requests"][0])
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached for {api_name}, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Add current request time
        rate_info["requests"].append(current_time)
    
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
            api_name: Name of the API (defi_llama, moralis, etc.)
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
        
        # Add API key to headers if available
        request_headers = headers or {}
        api_key = getattr(self.settings, f"{api_name}_api_key", None)
        if api_key:
            if api_name == "moralis":
                request_headers["X-API-Key"] = api_key
            elif api_name == "etherscan":
                params = params or {}
                params["apikey"] = api_key
            elif api_name == "coingecko":
                request_headers["x-cg-demo-api-key"] = api_key
        
        start_time = time.time()
        
        for attempt in range(retries + 1):
            try:
                async with self.session.request(
                    method,
                    url,
                    params=params,
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
            endpoint = f"simple/price"
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
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all API endpoints.
        
        Returns:
            Health check results
        """
        results = {}
        
        for api_name in self.endpoints.keys():
            try:
                # Simple test request to each API
                if api_name == "defi_llama":
                    await self.make_request(api_name, "protocols", params={"limit": 1})
                elif api_name == "coingecko":
                    await self.make_request(api_name, "ping")
                else:
                    # For other APIs, just check if the endpoint is reachable
                    await self.make_request(api_name, "")
                
                results[api_name] = "healthy"
                
            except Exception as e:
                self.logger.error(f"Health check failed for {api_name}: {str(e)}")
                results[api_name] = f"unhealthy: {str(e)}"
        
        return results
