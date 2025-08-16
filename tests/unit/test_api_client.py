"""
Unit tests for APIClient class.

This module contains comprehensive unit tests for the API client functionality
including rate limiting, error handling, and mock responses.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp

from src.data_collection.api_client import APIClient
from src.config.settings import get_settings


class TestAPIClient:
    """Test cases for APIClient class."""
    
    @pytest.fixture
    def api_client(self):
        """Create APIClient instance for testing."""
        return APIClient()
    
    @pytest.fixture
    async def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock()
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={"test": "data"})
        response.text = AsyncMock(return_value="OK")
        session.request.return_value.__aenter__ = AsyncMock(return_value=response)
        session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        return session
    
    @pytest.mark.asyncio
    async def test_start_session(self, api_client):
        """Test session initialization."""
        await api_client.start_session()
        assert api_client.session is not None
        await api_client.close_session()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, api_client):
        """Test rate limiting functionality."""
        api_client.rate_limiter["test_api"] = {
            "requests": [],
            "limit": 2
        }
        
        # Should not block for first request
        await api_client._check_rate_limit("test_api")
        assert len(api_client.rate_limiter["test_api"]["requests"]) == 1
        
        # Should not block for second request
        await api_client._check_rate_limit("test_api")
        assert len(api_client.rate_limiter["test_api"]["requests"]) == 2
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, api_client, mock_session):
        """Test successful API request."""
        api_client.session = mock_session
        
        result = await api_client.make_request(
            "defi_llama",
            "test-endpoint",
            method="GET"
        )
        
        assert result == {"test": "data"}
        mock_session.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_invalid_api(self, api_client):
        """Test request with invalid API name."""
        with pytest.raises(ValueError, match="Unknown API"):
            await api_client.make_request("invalid_api", "test")
    
    @pytest.mark.asyncio
    async def test_health_check(self, api_client):
        """Test health check functionality."""
        with patch.object(api_client, 'get_token_price', return_value={"bitcoin": {"usd": 50000}}):
            with patch.object(api_client, 'get_defi_protocols', return_value=[{"name": "test"}]):
                with patch.object(api_client, 'get_ethereum_price', return_value={"result": {"ethusd": "3000"}}):
                    results = await api_client.health_check()
                    
                    assert isinstance(results, dict)
                    assert "coingecko" in results
                    assert "defi_llama" in results
                    assert "etherscan" in results
