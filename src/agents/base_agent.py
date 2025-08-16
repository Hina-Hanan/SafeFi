"""
Base agent class for SafeFi DeFi Risk Assessment Agent.

This module provides the base agent class that all other agents inherit from,
ensuring consistent behavior and interfaces across the multi-agent system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
from loguru import logger

from ..config.settings import get_settings
from ..utils.logger import get_logger, log_function_call, log_error_with_context


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the SafeFi system.
    
    This class provides common functionality and enforces a consistent
    interface for all agents in the multi-agent system.
    """
    
    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent
            config: Optional agent-specific configuration
        """
        self.agent_name = agent_name
        self.config = config or {}
        self.settings = get_settings()
        self.logger = get_logger(agent_name)
        self.is_running = False
        self.last_execution = None
        self.execution_count = 0
        
        self.logger.info(f"Initialized {agent_name} agent")
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute agent's main functionality.
        
        Args:
            **kwargs: Agent-specific parameters
            
        Returns:
            Dictionary containing execution results
        """
        pass
    
    @abstractmethod
    async def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    async def start(self) -> None:
        """Start the agent."""
        try:
            self.is_running = True
            self.logger.info(f"Starting {self.agent_name} agent")
            
            # Perform any startup tasks
            await self.on_start()
            
        except Exception as e:
            self.logger.error(f"Failed to start {self.agent_name}: {str(e)}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop the agent."""
        try:
            self.is_running = False
            self.logger.info(f"Stopping {self.agent_name} agent")
            
            # Perform any cleanup tasks
            await self.on_stop()
            
        except Exception as e:
            self.logger.error(f"Failed to stop {self.agent_name}: {str(e)}")
            raise
    
    async def on_start(self) -> None:
        """Hook for agent startup tasks. Override in subclasses if needed."""
        pass
    
    async def on_stop(self) -> None:
        """Hook for agent cleanup tasks. Override in subclasses if needed."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform agent health check.
        
        Returns:
            Dictionary containing health status
        """
        try:
            status = {
                "agent_name": self.agent_name,
                "is_running": self.is_running,
                "last_execution": self.last_execution,
                "execution_count": self.execution_count,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy" if self.is_running else "stopped"
            }
            
            # Perform agent-specific health checks
            additional_checks = await self.perform_health_checks()
            status.update(additional_checks)
            
            return status
            
        except Exception as e:
            error_context = {"agent_name": self.agent_name}
            log_error_with_context(e, error_context)
            
            return {
                "agent_name": self.agent_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def perform_health_checks(self) -> Dict[str, Any]:
        """
        Perform agent-specific health checks.
        Override in subclasses for custom checks.
        
        Returns:
            Dictionary containing additional health check results
        """
        return {}
    
    async def safe_execute(self, **kwargs) -> Dict[str, Any]:
        """
        Safely execute agent with error handling and logging.
        
        Args:
            **kwargs: Execution parameters
            
        Returns:
            Dictionary containing execution results
        """
        start_time = datetime.utcnow()
        
        try:
            # Log function call
            log_function_call(f"{self.agent_name}.execute", kwargs)
            
            # Validate input
            if not await self.validate_input(**kwargs):
                raise ValueError("Input validation failed")
            
            # Execute agent
            result = await self.execute(**kwargs)
            
            # Update execution tracking
            self.last_execution = start_time
            self.execution_count += 1
            
            # Log successful execution
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(
                f"{self.agent_name} executed successfully",
                extra={"execution_time": execution_time}
            )
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_context = {
                "agent_name": self.agent_name,
                "execution_time": execution_time,
                "kwargs": kwargs
            }
            log_error_with_context(e, error_context)
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Dictionary containing agent status
        """
        return {
            "agent_name": self.agent_name,
            "is_running": self.is_running,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "execution_count": self.execution_count,
            "config": self.config
        }
