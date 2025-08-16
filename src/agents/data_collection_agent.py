"""
Data Collection Agent for SafeFi DeFi Risk Assessment Agent.

This agent manages multi-source data gathering from DeFi APIs and blockchain
with PostgreSQL storage.
"""

import asyncio
from typing import Any, Dict, Optional
from loguru import logger

from ..agents.base_agent import BaseAgent
from ..data_collection.api_client import APIClient
from ..data_collection.blockchain_client import BlockchainClient
from ..database.postgres_manager import PostgreSQLManager


class DataCollectionAgent(BaseAgent):
    """
    Data collection agent with PostgreSQL storage.
    
    This agent collects data from multiple DeFi APIs and blockchain sources,
    then stores the data in PostgreSQL for analysis and risk assessment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCollectionAgent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(agent_name="DataCollectionAgent", config=config)
        self.api_client = APIClient()
        self.blockchain_client = BlockchainClient()
        self.db_manager = PostgreSQLManager()

    async def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters for data collection.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            True if input is valid
        """
        # For basic data collection, no specific input validation needed
        return True

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute data collection from APIs and store in PostgreSQL.
        
        Args:
            **kwargs: Execution parameters
            
        Returns:
            Dictionary containing execution results
        """
        self.logger.info("Executing data collection with PostgreSQL storage")
        results = {}
        
        try:
            # Collect protocols from DeFiLlama
            protocols = await self.api_client.get_defi_protocols()
            results["protocols_collected"] = len(protocols)
            self.logger.info(f"Collected {len(protocols)} protocols from DeFiLlama")
            
            # Store top 10 protocols in PostgreSQL for testing
            stored_count = 0
            for protocol in protocols[:10]:
                try:
                    # Prepare protocol data
                    protocol_data = {
                        "name": protocol.get("name", "Unknown"),
                        "category": protocol.get("category", "Unknown"),
                        "chain": protocol.get("chain", "ethereum"),
                        "tvl": protocol.get("tvl", 0)
                    }
                    
                    # Insert protocol
                    protocol_id = await self.db_manager.insert_protocol(protocol_data)
                    
                    # Store TVL metric if available
                    if protocol.get("tvl") and protocol["tvl"] > 0:
                        await self.db_manager.insert_metric(
                            protocol_id=protocol_id,
                            metric_type="tvl",
                            value=float(protocol["tvl"]),
                            source="defillama"
                        )
                    
                    stored_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to store protocol {protocol.get('name', 'Unknown')}: {e}")
                    
            results["protocols_stored"] = stored_count
            self.logger.info(f"Stored {stored_count} protocols in PostgreSQL")
            
            # Test additional API endpoints
            try:
                # Test CoinGecko Bitcoin price
                btc_price = await self.api_client.get_token_price("bitcoin")
                if btc_price and "bitcoin" in btc_price:
                    results["bitcoin_price"] = btc_price["bitcoin"]["usd"]
                    self.logger.info(f"Bitcoin price: ${btc_price['bitcoin']['usd']}")
                
                # Test Etherscan ETH price
                eth_price = await self.api_client.get_ethereum_price()
                if eth_price and "result" in eth_price:
                    results["eth_price"] = eth_price["result"]["ethusd"]
                    self.logger.info(f"ETH price: ${eth_price['result']['ethusd']}")
                    
            except Exception as e:
                self.logger.error(f"Failed to collect price data: {e}")
                results["price_data_error"] = str(e)
            
            # Test blockchain connection
            try:
                block_number = self.blockchain_client.get_latest_block_number()
                results["latest_block"] = block_number
                self.logger.info(f"Latest blockchain block: {block_number}")
            except Exception as e:
                self.logger.error(f"Failed to get blockchain data: {e}")
                results["blockchain_error"] = str(e)
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            results["collection_error"] = str(e)
            
        # Test database health
        try:
            health = await self.db_manager.health_check()
            results["database_health"] = health["status"]
            results["database_info"] = {
                "protocols": health.get("protocols", 0),
                "metrics": health.get("metrics", 0),
                "alerts": health.get("alerts", 0)
            }
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            results["database_health"] = f"Failed: {e}"

        return results

    async def collect_protocol_details(self, protocol_name: str) -> Dict[str, Any]:
        """
        Collect detailed information for a specific protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            Detailed protocol information
        """
        try:
            # Get protocol TVL history from DeFiLlama
            tvl_data = await self.api_client.get_protocol_tvl(protocol_name)
            
            # Get protocol from database
            protocol = await self.db_manager.get_protocol_by_name(protocol_name)
            
            if protocol:
                # Get historical metrics
                metrics = await self.db_manager.get_metrics(protocol["id"], limit=50)
                
                return {
                    "protocol": protocol,
                    "tvl_history": tvl_data,
                    "metrics": metrics,
                    "status": "success"
                }
            else:
                return {
                    "error": f"Protocol {protocol_name} not found in database",
                    "status": "not_found"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to collect protocol details for {protocol_name}: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def generate_risk_alert(self, protocol_id: int, alert_type: str, 
                                 severity: str, message: str) -> bool:
        """
        Generate and store a risk alert.
        
        Args:
            protocol_id: Protocol ID
            alert_type: Type of alert
            severity: Alert severity level
            message: Alert message
            
        Returns:
            True if alert was stored successfully
        """
        try:
            alert_id = await self.db_manager.insert_alert(
                protocol_id=protocol_id,
                alert_type=alert_type,
                severity=severity,
                message=message
            )
            
            self.logger.info(f"Generated {severity} alert {alert_id} for protocol {protocol_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate alert: {e}")
            return False

    async def on_start(self) -> None:
        """Start database connection and API client on agent startup."""
        try:
            await self.db_manager.connect()
            await self.api_client.start_session()
            self.logger.info("DataCollectionAgent started with PostgreSQL connection")
        except Exception as e:
            self.logger.error(f"Failed to start DataCollectionAgent: {e}")
            raise

    async def on_stop(self) -> None:
        """Close database connection and API client on agent shutdown."""
        try:
            await self.db_manager.disconnect()
            await self.api_client.close_session()
            self.logger.info("DataCollectionAgent stopped, connections closed")
        except Exception as e:
            self.logger.error(f"Error stopping DataCollectionAgent: {e}")

    async def perform_health_checks(self) -> Dict[str, Any]:
        """
        Perform agent-specific health checks.
        
        Returns:
            Dictionary containing health check results
        """
        health_results = {}
        
        try:
            # Check database health
            db_health = await self.db_manager.health_check()
            health_results["database"] = db_health["status"]
            
            # Check API health
            api_health = await self.api_client.health_check()
            health_results["apis"] = api_health
            
            # Check blockchain connection
            try:
                block_number = self.blockchain_client.get_latest_block_number()
                health_results["blockchain"] = "healthy" if block_number > 0 else "unhealthy"
            except Exception:
                health_results["blockchain"] = "unhealthy"
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_results["error"] = str(e)
        
        return health_results
