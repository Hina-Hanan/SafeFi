"""
Integration tests for the complete data pipeline.

This module tests the end-to-end data collection and processing workflow.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from src.agents.data_collection_agent import DataCollectionAgent
from src.config.settings import get_settings


class TestDataPipeline:
    """Integration tests for data pipeline."""
    
    @pytest.fixture
    def data_agent(self):
        """Create data collection agent for testing."""
        return DataCollectionAgent()
    
    @pytest.mark.asyncio
    async def test_full_data_collection_workflow(self, data_agent):
        """Test complete data collection workflow."""
        # Mock external dependencies
        mock_protocols = [
            {"name": "Uniswap", "tvl": 5000000000, "category": "dex", "chain": "ethereum"},
            {"name": "Aave", "tvl": 8000000000, "category": "lending", "chain": "ethereum"}
        ]
        
        with patch.object(data_agent.api_client, 'get_defi_protocols', return_value=mock_protocols):
            with patch.object(data_agent.blockchain_client, 'get_latest_block_number', return_value=18500000):
                
                # Execute data collection
                result = await data_agent.safe_execute()
                
                # Verify results
                assert result["success"] is True
                assert "protocols" in result["result"]
                assert "latest_block" in result["result"]
                assert len(result["result"]["protocols"]) == 2
                assert result["result"]["latest_block"] == 18500000
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, data_agent):
        """Test complete agent lifecycle."""
        # Test startup
        await data_agent.start()
        assert data_agent.is_running is True
        
        # Test health check
        health = await data_agent.health_check()
        assert health["status"] in ["healthy", "unhealthy"]
        assert health["agent_name"] == "DataCollectionAgent"
        
        # Test shutdown
        await data_agent.stop()
        assert data_agent.is_running is False
    
    @pytest.mark.asyncio
    async def test_api_client_integration(self, data_agent):
        """Test API client integration."""
        # Test session management
        await data_agent.api_client.start_session()
        assert data_agent.api_client.session is not None
        
        # Test health check
        with patch.object(data_agent.api_client, 'make_request', return_value={"status": "ok"}):
            health_results = await data_agent.api_client.health_check()
            assert isinstance(health_results, dict)
        
        # Clean up
        await data_agent.api_client.close_session()
        assert data_agent.api_client.session is None
