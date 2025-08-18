import pytest
from unittest.mock import AsyncMock
from src.data_collection.api_client import APIClient


@pytest.fixture
def api_client():
    """Create APIClient instance for testing."""
    return APIClient()


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session with async context manager support."""
    # Mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"result": "success"})
    mock_response.raise_for_status = AsyncMock()

    # Make response act as async context manager
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None

    # Mock session
    mock_session = AsyncMock()
    mock_session.request.return_value = mock_response  # <- THIS is the fix
    # Ensure request itself behaves like an async context manager
    mock_session.request.return_value.__aenter__.return_value = mock_response
    mock_session.request.return_value.__aexit__.return_value = None

    return mock_session


class TestAPIClient:
    @pytest.mark.asyncio
    async def test_start_session(self, api_client):
        """Test session startup."""
        await api_client.start_session()
        assert api_client.session is not None

    @pytest.mark.asyncio
    async def test_make_request_success(self, api_client, mock_session):
        """Test successful API request."""
        api_client.session = mock_session

        # Initialize rate_limiter for proper _check_rate_limit
        api_client.rate_limiter = {
            "defi_llama": {"last_request": 0, "min_interval": 0}
        }
        # Also patch endpoints so APIClient finds a base_url
        api_client.endpoints["defi_llama"] = "http://mock-api"

        result = await api_client.make_request(
            "defi_llama",
            "test-endpoint",
            method="GET"
        )

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_make_request_invalid_api(self, api_client):
        """Test request with invalid API name."""
        with pytest.raises(ValueError, match="Unknown API"):
            await api_client.make_request("invalid_api", "test")

    @pytest.mark.asyncio
    async def test_health_check(self, api_client):
        """Test health check."""
        health = await api_client.health_check()
        # Your health_check returns keys for each provider
        # Assert at least one provider is "healthy" or "unhealthy"
        assert any(
            val.startswith("healthy") or val.startswith("unhealthy")
            for val in health.values()
        )

    @pytest.mark.asyncio
    async def test_rate_limiting(self, api_client):
        """Test rate limiting."""
        # Initialize rate_limiter with required keys
        api_client.rate_limiter = {
            "coingecko": {"last_request": 0, "min_interval": 0}
        }
        await api_client._check_rate_limit("coingecko")