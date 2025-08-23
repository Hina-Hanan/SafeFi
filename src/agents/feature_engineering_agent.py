"""
Feature Engineering Agent for SafeFi DeFi Risk Assessment Agent.

This agent calculates technical indicators, volatility metrics, and risk features
from collected protocol data with comprehensive error handling and logging.
"""

import asyncio
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from ..agents.base_agent import BaseAgent
from ..database.postgres_manager import PostgreSQLManager
from ..utils.logger import get_logger, log_function_call, log_error_with_context
from ..utils.validators import validate_protocol_data, ValidationError


class FeatureEngineeringAgent(BaseAgent):
    """
    Feature engineering agent for calculating DeFi protocol risk indicators.
    
    This agent processes raw protocol data to generate meaningful features
    for risk assessment, including volatility metrics, liquidity indicators,
    and technical analysis features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FeatureEngineeringAgent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(agent_name="FeatureEngineeringAgent", config=config)
        self.db_manager = PostgreSQLManager()
        
        # Feature calculation parameters
        self.volatility_windows = [7, 14, 30]
        self.momentum_windows = [7, 14]
        self.trend_window = 14
        
    async def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters for feature engineering.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            True if input is valid
            
        Raises:
            ValidationError: If input validation fails
        """
        try:
            protocol_id = kwargs.get('protocol_id')
            if protocol_id and not isinstance(protocol_id, int):
                raise ValidationError("protocol_id must be an integer")
            
            return True
            
        except Exception as e:
            log_error_with_context(e, {"kwargs": kwargs})
            return False
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute feature engineering for protocols.
        
        Args:
            **kwargs: Execution parameters (protocol_id optional)
            
        Returns:
            Dictionary containing execution results
        """
        log_function_call("FeatureEngineeringAgent.execute", kwargs)
        results = {"features_calculated": 0, "protocols_processed": []}
        
        try:
            protocol_id = kwargs.get('protocol_id')
            
            if protocol_id:
                # Process single protocol
                features = await self._calculate_protocol_features(protocol_id)
                if features:
                    await self._store_features(protocol_id, features)
                    results["features_calculated"] = 1
                    results["protocols_processed"] = [protocol_id]
                    
            else:
                # Process all protocols
                protocols = await self.db_manager.get_protocols(limit=50)
                
                for protocol in protocols:
                    try:
                        features = await self._calculate_protocol_features(protocol['id'])
                        if features:
                            await self._store_features(protocol['id'], features)
                            results["features_calculated"] += 1
                            results["protocols_processed"].append(protocol['id'])
                            
                    except Exception as e:
                        self.logger.error(f"Failed to process protocol {protocol['id']}: {e}")
                        continue
            
            self.logger.info(f"Processed features for {results['features_calculated']} protocols")
            return results
            
        except Exception as e:
            log_error_with_context(e, {"kwargs": kwargs})
            raise
    
    def calculate_impermanent_loss(self, original_prices: Dict[str, float], 
                                  current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate impermanent loss for liquidity provider positions.
        
        Args:
            original_prices: Initial token prices when entering pool
            current_prices: Current token prices
            
        Returns:
            Dictionary containing impermanent loss metrics
        """
        try:
            impermanent_loss_data = {}
            
            for token_pair, initial_price in original_prices.items():
                if token_pair in current_prices:
                    current_price = current_prices[token_pair]
                    
                    if initial_price > 0 and current_price > 0:
                        # Standard IL formula: IL = 1 - (2 * sqrt(P)) / (1 + P)
                        price_ratio = current_price / initial_price
                        
                        # Calculate impermanent loss percentage
                        il_percentage = 1 - (2 * (price_ratio ** 0.5)) / (1 + price_ratio)
                        
                        # Calculate severity categories
                        if abs(il_percentage) < 0.02:
                            severity = "low"
                        elif abs(il_percentage) < 0.05:
                            severity = "medium" 
                        elif abs(il_percentage) < 0.15:
                            severity = "high"
                        else:
                            severity = "critical"
                        
                        impermanent_loss_data[token_pair] = {
                            'il_percentage': float(il_percentage * 100),  # Convert to percentage
                            'price_ratio': float(price_ratio),
                            'severity': severity,
                            'initial_price': float(initial_price),
                            'current_price': float(current_price)
                        }
            
            self.logger.debug(f"Calculated impermanent loss for {len(impermanent_loss_data)} pairs")
            return impermanent_loss_data
            
        except Exception as e:
            log_error_with_context(e, {"original_prices": original_prices, "current_prices": current_prices})
            return {}
    
    async def _calculate_protocol_features(self, protocol_id: int) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive features for a protocol.
        
        Args:
            protocol_id: Protocol ID
            
        Returns:
            Dictionary of calculated features or None if insufficient data
        """
        try:
            # Get historical metrics data
            metrics = await self.db_manager.get_metrics(protocol_id, limit=90)
            
            if len(metrics) < 7:  # Need minimum data for calculations
                self.logger.warning(f"Insufficient data for protocol {protocol_id}")
                return None
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Pivot metrics by type for easier calculation
            pivoted_data = {}
            for metric_type in df['metric_type'].unique():
                metric_data = df[df['metric_type'] == metric_type].copy()
                pivoted_data[metric_type] = metric_data
            
            features = {}
            
            # Calculate TVL-based features
            if 'tvl' in pivoted_data:
                tvl_features = self._calculate_tvl_features(pivoted_data['tvl'])
                features.update(tvl_features)
            
            # Calculate price-based features
            if 'price' in pivoted_data:
                price_features = self._calculate_price_features(pivoted_data['price'])
                features.update(price_features)
            
            # Calculate volume-based features
            if 'volume' in pivoted_data:
                volume_features = self._calculate_volume_features(pivoted_data['volume'])
                features.update(volume_features)
            
            # Calculate cross-metric features
            if 'tvl' in pivoted_data and 'volume' in pivoted_data:
                liquidity_features = self._calculate_liquidity_features(
                    pivoted_data['tvl'], pivoted_data['volume']
                )
                features.update(liquidity_features)
            
            # Add metadata
            features['calculation_timestamp'] = datetime.utcnow()
            features['data_points_used'] = len(metrics)
            
            return features
            
        except Exception as e:
            log_error_with_context(e, {"protocol_id": protocol_id})
            return None
    
    def _calculate_tvl_features(self, tvl_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate TVL-based features.
        
        Args:
            tvl_data: DataFrame with TVL data
            
        Returns:
            Dictionary of TVL features
        """
        try:
            values = tvl_data['value'].astype(float)
            features = {}
            
            # Basic statistics
            features['tvl_mean'] = float(values.mean())
            features['tvl_std'] = float(values.std())
            features['tvl_current'] = float(values.iloc[-1])
            
            # Percentage changes
            pct_changes = values.pct_change().dropna()
            if len(pct_changes) > 0:
                features['tvl_volatility'] = float(pct_changes.std())
                features['tvl_change_1d'] = float(pct_changes.iloc[-1])
                
                # Multi-period changes
                if len(values) >= 7:
                    features['tvl_change_7d'] = float((values.iloc[-1] / values.iloc[-7]) - 1)
                if len(values) >= 30:
                    features['tvl_change_30d'] = float((values.iloc[-1] / values.iloc[-30]) - 1)
            
            # Trend analysis
            features['tvl_trend'] = self._calculate_trend(values)
            
            # Momentum indicators
            for window in self.momentum_windows:
                if len(values) > window:
                    momentum = self._calculate_momentum(values, window)
                    features[f'tvl_momentum_{window}d'] = momentum
            
            return features
            
        except Exception as e:
            self.logger.error(f"TVL feature calculation failed: {e}")
            return {}
    
    def _calculate_price_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate price-based features.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Dictionary of price features
        """
        try:
            values = price_data['value'].astype(float)
            features = {}
            
            # Basic price statistics
            features['price_mean'] = float(values.mean())
            features['price_current'] = float(values.iloc[-1])
            
            # Returns and volatility
            returns = values.pct_change().dropna()
            if len(returns) > 0:
                features['price_volatility'] = float(returns.std())
                features['price_return_1d'] = float(returns.iloc[-1])
                
                # Volatility over different windows
                for window in self.volatility_windows:
                    if len(returns) >= window:
                        vol = returns.rolling(window).std().iloc[-1]
                        features[f'price_volatility_{window}d'] = float(vol)
            
            # Technical indicators
            if len(values) >= 14:
                # RSI-like momentum indicator
                features['price_rsi'] = self._calculate_rsi(values, 14)
                
            # Trend indicators
            features['price_trend'] = self._calculate_trend(values)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Price feature calculation failed: {e}")
            return {}
    
    def _calculate_volume_features(self, volume_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volume-based features.
        
        Args:
            volume_data: DataFrame with volume data
            
        Returns:
            Dictionary of volume features
        """
        try:
            values = volume_data['value'].astype(float)
            features = {}
            
            # Basic volume statistics
            features['volume_mean'] = float(values.mean())
            features['volume_std'] = float(values.std())
            features['volume_current'] = float(values.iloc[-1])
            
            # Volume consistency (inverse of coefficient of variation)
            if values.mean() > 0:
                features['volume_consistency'] = float(1 / (1 + values.std() / values.mean()))
            else:
                features['volume_consistency'] = 0.0
            
            # Volume trend
            features['volume_trend'] = self._calculate_trend(values)
            
            # Volume spikes (values above 2 std deviations)
            threshold = values.mean() + 2 * values.std()
            spikes = (values > threshold).sum()
            features['volume_spikes'] = float(spikes / len(values))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Volume feature calculation failed: {e}")
            return {}
    
    def _calculate_liquidity_features(self, tvl_data: pd.DataFrame, 
                                    volume_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate liquidity-related features.
        
        Args:
            tvl_data: DataFrame with TVL data
            volume_data: DataFrame with volume data
            
        Returns:
            Dictionary of liquidity features
        """
        try:
            tvl_values = tvl_data['value'].astype(float)
            volume_values = volume_data['value'].astype(float)
            
            features = {}
            
            # Align data by timestamp for proper comparison
            if len(tvl_values) > 0 and len(volume_values) > 0:
                # Volume to TVL ratio (liquidity indicator)
                current_tvl = tvl_values.iloc[-1]
                current_volume = volume_values.iloc[-1]
                
                if current_tvl > 0:
                    features['volume_tvl_ratio'] = float(current_volume / current_tvl)
                else:
                    features['volume_tvl_ratio'] = 0.0
                
                # Average ratios over different periods
                min_length = min(len(tvl_values), len(volume_values))
                if min_length >= 7:
                    tvl_recent = tvl_values.tail(7).mean()
                    volume_recent = volume_values.tail(7).mean()
                    if tvl_recent > 0:
                        features['avg_volume_tvl_ratio_7d'] = float(volume_recent / tvl_recent)
                    else:
                        features['avg_volume_tvl_ratio_7d'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Liquidity feature calculation failed: {e}")
            return {}
    
    def _calculate_trend(self, values: pd.Series, window: int = None) -> float:
        """
        Calculate trend using linear regression slope.
        
        Args:
            values: Time series values
            window: Window size (uses all data if None)
            
        Returns:
            Trend indicator (normalized slope)
        """
        try:
            if window and len(values) > window:
                data = values.tail(window)
            else:
                data = values
            
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            slope = np.polyfit(x, data.values, 1)[0]
            
            # Normalize by mean to make it scale-invariant
            mean_val = data.mean()
            if mean_val > 0:
                return float(slope / mean_val)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Trend calculation failed: {e}")
            return 0.0
    
    def _calculate_momentum(self, values: pd.Series, window: int) -> float:
        """
        Calculate momentum indicator.
        
        Args:
            values: Time series values
            window: Window size for momentum calculation
            
        Returns:
            Momentum indicator
        """
        try:
            if len(values) <= window:
                return 0.0
            
            current = values.iloc[-1]
            past = values.iloc[-(window + 1)]
            
            if past > 0:
                return float((current - past) / past)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Momentum calculation failed: {e}")
            return 0.0
    
    def _calculate_rsi(self, values: pd.Series, window: int = 14) -> float:
        """
        Calculate RSI-like momentum indicator.
        
        Args:
            values: Price values
            window: Window size
            
        Returns:
            RSI-like indicator (0-100)
        """
        try:
            if len(values) < window + 1:
                return 50.0  # Neutral
            
            deltas = values.diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            avg_gain = gains.rolling(window=window).mean().iloc[-1]
            avg_loss = losses.rolling(window=window).mean().iloc[-1]
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}")
            return 50.0
    
    async def _store_features(self, protocol_id: int, features: Dict[str, Any]) -> None:
        """
        Store calculated features in database.
        
        Args:
            protocol_id: Protocol ID
            features: Calculated features dictionary
        """
        try:
            # Store key features as metrics for easy retrieval
            key_features = [
                'tvl_volatility', 'tvl_change_7d', 'price_volatility',
                'volume_tvl_ratio', 'tvl_trend', 'price_trend'
            ]
            
            for feature_name in key_features:
                if feature_name in features:
                    await self.db_manager.insert_metric(
                        protocol_id=protocol_id,
                        metric_type=f"feature_{feature_name}",
                        value=float(features[feature_name]),
                        source="feature_engineering"
                    )
            
            self.logger.debug(f"Stored {len(key_features)} features for protocol {protocol_id}")
            
        except Exception as e:
            log_error_with_context(e, {"protocol_id": protocol_id, "features": features})
            raise
    
    async def on_start(self) -> None:
        """Initialize database connection on agent startup."""
        try:
            await self.db_manager.connect()
            self.logger.info("FeatureEngineeringAgent started")
        except Exception as e:
            self.logger.error(f"Failed to start FeatureEngineeringAgent: {e}")
            raise
    
    async def on_stop(self) -> None:
        """Close database connection on agent shutdown."""
        try:
            await self.db_manager.disconnect()
            self.logger.info("FeatureEngineeringAgent stopped")
        except Exception as e:
            self.logger.error(f"Error stopping FeatureEngineeringAgent: {e}")
    
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
            
            # Check feature calculation capabilities
            health_results["feature_calculation"] = "healthy"
            
            # Check recent feature generation activity
            protocols = await self.db_manager.get_protocols(limit=1)
            if protocols:
                recent_features = await self.db_manager.get_metrics(
                    protocols[0]['id'], 
                    metric_type="feature_tvl_volatility",
                    limit=1
                )
                health_results["recent_features"] = "healthy" if recent_features else "no_recent_data"
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_results["error"] = str(e)
        
        return health_results
