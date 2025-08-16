"""
Unit tests for BaseAgent class.

This module tests the base agent functionality including lifecycle management,
health checks, and error handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime

from src.agents.base_agent import BaseAgent


class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    async def execute(self, **kwargs):
        """Test implementation of execute method."""
        if kwargs.get("should_fail"):
            raise ValueError("Test error")
        return {"status": "success", "data": kwargs}
    
    async def validate_input(self, **kwargs):
        """Test implementation of validate_input method."""
        return not kwargs.get("invalid_input", False)


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent instance."""
        return TestAgent("TestAgent", {"test_config": "value"})
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, test_agent):
        """Test agent initialization."""
        assert test_agent.agent_name == "TestAgent"
        assert test_agent.config == {"test_config": "value"}
        assert test_agent.is_running is False
        assert test_agent.execution_count == 0
    
    @pytest.mark.asyncio
    async def test_start_agent(self, test_agent):
        """Test agent startup."""
        await test_agent.start()
        assert test_agent.is_running is True
        await test_agent.stop()
    
    @pytest.mark.asyncio
    async def test_stop_agent(self, test_agent):
        """Test agent shutdown."""
        await test_agent.start()
        await test_agent.stop()
        assert test_agent.is_running is False
    
    @pytest.mark.asyncio
    async def test_safe_execute_success(self, test_agent):
        """Test successful safe execution."""
        result = await test_agent.safe_execute(test_param="value")
        
        assert result["success"] is True
        assert result["result"]["status"] == "success"
        assert test_agent.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_safe_execute_failure(self, test_agent):
        """Test safe execution with failure."""
        result = await test_agent.safe_execute(should_fail=True)
        
        assert result["success"] is False
        assert "Test error" in result["error"]
        assert result["error_type"] == "ValueError"
    
    @pytest.mark.asyncio
    async def test_safe_execute_invalid_input(self, test_agent):
        """Test safe execution with invalid input."""
        result = await test_agent.safe_execute(invalid_input=True)
        
        assert result["success"] is False
        assert "Input validation failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_agent):
        """Test health check functionality."""
        health_status = await test_agent.health_check()
        
        assert health_status["agent_name"] == "TestAgent"
        assert health_status["is_running"] is False
        assert health_status["execution_count"] == 0
        assert "timestamp" in health_status
    
    def test_get_status(self, test_agent):
        """Test get status functionality."""
        status = test_agent.get_status()
        
        assert status["agent_name"] == "TestAgent"
        assert status["is_running"] is False
        assert status["execution_count"] == 0
        assert status["config"] == {"test_config": "value"}
