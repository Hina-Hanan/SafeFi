"""
PostgreSQL database manager for SafeFi DeFi Risk Assessment Agent.

This module provides async PostgreSQL connection management and CRUD operations.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncpg
from loguru import logger

from ..config.settings import get_settings
from ..utils.logger import get_logger, log_error_with_context


class PostgreSQLManager:
    """
    Async PostgreSQL database manager.
    
    Handles connection pooling, CRUD operations, and database schema management
    for the SafeFi DeFi Risk Assessment system.
    """
    
    def __init__(self):
        """Initialize PostgreSQL manager."""
        self.settings = get_settings()
        self.logger = get_logger("PostgreSQLManager")
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Create PostgreSQL connection pool."""
        try:
            # Get PostgreSQL DSN for asyncpg
            db_url = self.settings.get_postgres_dsn()
            
            self.pool = await asyncpg.create_pool(
                db_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'jit': 'off'  # Disable JIT for better performance on small queries
                }
            )
            
            self.logger.info("PostgreSQL connection pool created successfully")
            await self._create_tables()
            
        except Exception as e:
            log_error_with_context(e, {"database_url": self.settings.database_url})
            raise
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("PostgreSQL connection pool closed")
    
    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            async with self.pool.acquire() as conn:
                # Create protocols table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS protocols (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL UNIQUE,
                        category VARCHAR(50),
                        chain VARCHAR(50),
                        address VARCHAR(42),
                        tvl DECIMAL(20,2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create metrics table for time-series data
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id SERIAL PRIMARY KEY,
                        protocol_id INTEGER REFERENCES protocols(id) ON DELETE CASCADE,
                        metric_type VARCHAR(50) NOT NULL,
                        value DECIMAL(20,8) NOT NULL,
                        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        source VARCHAR(50) NOT NULL
                    );
                """)
                
                # Create alerts table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id SERIAL PRIMARY KEY,
                        protocol_id INTEGER REFERENCES protocols(id) ON DELETE CASCADE,
                        alert_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
                        message TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved BOOLEAN DEFAULT FALSE
                    );
                """)
                
                # Create indexes for better performance
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_protocol_timestamp ON metrics(protocol_id, timestamp);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_protocols_name ON protocols(name);")
                
                self.logger.info("Database tables and indexes created successfully")
                
        except Exception as e:
            log_error_with_context(e, {"operation": "create_tables"})
            raise
    
    async def insert_protocol(self, protocol_data: Dict[str, Any]) -> int:
        """
        Insert protocol data and return ID.
        
        Args:
            protocol_data: Protocol information dictionary
            
        Returns:
            Protocol ID
        """
        try:
            async with self.pool.acquire() as conn:
                protocol_id = await conn.fetchval("""
                    INSERT INTO protocols (name, category, chain, address, tvl)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (name) 
                    DO UPDATE SET 
                        category = EXCLUDED.category,
                        chain = EXCLUDED.chain,
                        address = EXCLUDED.address,
                        tvl = EXCLUDED.tvl,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """,
                    protocol_data['name'],
                    protocol_data.get('category'),
                    protocol_data.get('chain'),
                    protocol_data.get('address'),
                    protocol_data.get('tvl')
                )
                
                self.logger.debug(f"Inserted/updated protocol {protocol_data['name']} with ID {protocol_id}")
                return protocol_id
                
        except Exception as e:
            log_error_with_context(e, {"protocol_data": protocol_data})
            raise
    
    async def insert_metric(self, protocol_id: int, metric_type: str, 
                           value: float, source: str) -> None:
        """
        Insert time-series metric data.
        
        Args:
            protocol_id: Protocol ID
            metric_type: Type of metric (tvl, volume, price)
            value: Metric value
            source: Data source name
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO metrics (protocol_id, metric_type, value, source)
                    VALUES ($1, $2, $3, $4);
                """, protocol_id, metric_type, value, source)
                
                self.logger.debug(f"Inserted metric {metric_type}={value} for protocol {protocol_id}")
                
        except Exception as e:
            log_error_with_context(e, {
                "protocol_id": protocol_id,
                "metric_type": metric_type,
                "value": value,
                "source": source
            })
            raise
    
    async def insert_alert(self, protocol_id: int, alert_type: str, 
                          severity: str, message: str) -> int:
        """
        Insert alert data.
        
        Args:
            protocol_id: Protocol ID
            alert_type: Type of alert
            severity: Alert severity (low, medium, high, critical)
            message: Alert message
            
        Returns:
            Alert ID
        """
        try:
            async with self.pool.acquire() as conn:
                alert_id = await conn.fetchval("""
                    INSERT INTO alerts (protocol_id, alert_type, severity, message)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id;
                """, protocol_id, alert_type, severity, message)
                
                self.logger.debug(f"Inserted alert {alert_type} for protocol {protocol_id}")
                return alert_id
                
        except Exception as e:
            log_error_with_context(e, {
                "protocol_id": protocol_id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message
            })
            raise
    
    async def get_protocols(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get protocols from database.
        
        Args:
            limit: Maximum number of protocols to return
            
        Returns:
            List of protocol dictionaries
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, name, category, chain, address, tvl, created_at, updated_at
                    FROM protocols
                    ORDER BY tvl DESC NULLS LAST
                    LIMIT $1;
                """, limit)
                
                protocols = [dict(row) for row in rows]
                self.logger.debug(f"Retrieved {len(protocols)} protocols")
                return protocols
                
        except Exception as e:
            log_error_with_context(e, {"limit": limit})
            return []
    
    async def get_protocol_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get protocol by name.
        
        Args:
            name: Protocol name
            
        Returns:
            Protocol dictionary or None
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, name, category, chain, address, tvl, created_at, updated_at
                    FROM protocols
                    WHERE name = $1;
                """, name)
                
                if row:
                    protocol = dict(row)
                    self.logger.debug(f"Retrieved protocol {name}")
                    return protocol
                return None
                
        except Exception as e:
            log_error_with_context(e, {"name": name})
            return None
    
    async def get_metrics(self, protocol_id: int, metric_type: str = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get metrics for a protocol.
        
        Args:
            protocol_id: Protocol ID
            metric_type: Optional metric type filter
            limit: Maximum number of metrics to return
            
        Returns:
            List of metric dictionaries
        """
        try:
            async with self.pool.acquire() as conn:
                if metric_type:
                    rows = await conn.fetch("""
                        SELECT id, protocol_id, metric_type, value, timestamp, source
                        FROM metrics
                        WHERE protocol_id = $1 AND metric_type = $2
                        ORDER BY timestamp DESC
                        LIMIT $3;
                    """, protocol_id, metric_type, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT id, protocol_id, metric_type, value, timestamp, source
                        FROM metrics
                        WHERE protocol_id = $1
                        ORDER BY timestamp DESC
                        LIMIT $2;
                    """, protocol_id, limit)
                
                metrics = [dict(row) for row in rows]
                self.logger.debug(f"Retrieved {len(metrics)} metrics for protocol {protocol_id}")
                return metrics
                
        except Exception as e:
            log_error_with_context(e, {
                "protocol_id": protocol_id,
                "metric_type": metric_type,
                "limit": limit
            })
            return []
    
    async def get_alerts(self, protocol_id: int = None, resolved: bool = None, 
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alerts from database.
        
        Args:
            protocol_id: Optional protocol ID filter
            resolved: Optional resolved status filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, protocol_id, alert_type, severity, message, created_at, resolved
                    FROM alerts
                """
                params = []
                conditions = []
                
                if protocol_id is not None:
                    conditions.append(f"protocol_id = ${len(params) + 1}")
                    params.append(protocol_id)
                
                if resolved is not None:
                    conditions.append(f"resolved = ${len(params) + 1}")
                    params.append(resolved)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1};"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                alerts = [dict(row) for row in rows]
                self.logger.debug(f"Retrieved {len(alerts)} alerts")
                return alerts
                
        except Exception as e:
            log_error_with_context(e, {
                "protocol_id": protocol_id,
                "resolved": resolved,
                "limit": limit
            })
            return []
    
    async def update_alert_status(self, alert_id: int, resolved: bool) -> bool:
        """
        Update alert resolved status.
        
        Args:
            alert_id: Alert ID
            resolved: New resolved status
            
        Returns:
            True if updated successfully
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE alerts 
                    SET resolved = $1 
                    WHERE id = $2;
                """, resolved, alert_id)
                
                success = result == "UPDATE 1"
                if success:
                    self.logger.debug(f"Updated alert {alert_id} resolved status to {resolved}")
                return success
                
        except Exception as e:
            log_error_with_context(e, {"alert_id": alert_id, "resolved": resolved})
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Health check results
        """
        try:
            async with self.pool.acquire() as conn:
                # Test basic connectivity
                version = await conn.fetchval("SELECT version();")
                
                # Test table access
                protocol_count = await conn.fetchval("SELECT COUNT(*) FROM protocols;")
                metric_count = await conn.fetchval("SELECT COUNT(*) FROM metrics;")
                alert_count = await conn.fetchval("SELECT COUNT(*) FROM alerts;")
                
                return {
                    "status": "healthy",
                    "database": "postgresql",
                    "version": version.split()[0:2],  # Just PostgreSQL version
                    "protocols": protocol_count,
                    "metrics": metric_count,
                    "alerts": alert_count,
                    "pool_size": self.pool.get_size(),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            log_error_with_context(e, {"operation": "health_check"})
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
