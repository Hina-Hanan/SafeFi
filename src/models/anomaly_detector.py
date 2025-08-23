"""
Anomaly Detector for SafeFi DeFi Risk Assessment Agent.

This module implements anomaly detection for identifying unusual behavior
in DeFi protocol metrics that may indicate potential risks or exploits.
"""

import asyncio
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from loguru import logger

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

from ..utils.logger import get_logger, log_function_call, log_error_with_context
from ..database.postgres_manager import PostgreSQLManager


class AnomalyDetector:
    """
    Anomaly detection system for DeFi protocol monitoring.
    
    This class implements multiple anomaly detection algorithms to identify
    unusual patterns in protocol metrics that may indicate security risks,
    market manipulation, or technical issues.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize AnomalyDetector.
        
        Args:
            contamination: Expected proportion of anomalies in the data
        """
        self.logger = get_logger("AnomalyDetector")
        self.db_manager = PostgreSQLManager()
        self.contamination = contamination
        
        # Models
        self.isolation_forest = None
        self.lof_detector = None
        self.scaler = None
        
        # Configuration
        self.feature_columns = [
            'tvl', 'volume_24h', 'price_change_24h', 'volume_change_24h',
            'tvl_volatility', 'price_volatility', 'volume_tvl_ratio'
        ]
        
        # Thresholds
        self.anomaly_threshold = -0.5  # Isolation Forest threshold
        self.lof_threshold = 2.0       # LOF threshold
        
        # Models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    async def prepare_training_data(self, 
                                  protocol_ids: Optional[List[int]] = None,
                                  days_back: int = 30) -> pd.DataFrame:
        """
        Prepare training data from historical protocol metrics.
        
        Args:
            protocol_ids: List of protocol IDs to include (None for all)
            days_back: Number of days of historical data to use
            
        Returns:
            DataFrame with prepared training data
        """
        log_function_call("AnomalyDetector.prepare_training_data", {
            "protocol_ids": protocol_ids,
            "days_back": days_back
        })
        
        try:
            # Get protocols
            if protocol_ids:
                protocols = []
                for pid in protocol_ids:
                    protocol = await self.db_manager.get_protocol_by_name(str(pid))
                    if protocol:
                        protocols.append(protocol)
            else:
                protocols = await self.db_manager.get_protocols(limit=50)
            
            all_data = []
            
            for protocol in protocols:
                # Get metrics for each protocol
                metrics = await self.db_manager.get_metrics(
                    protocol['id'], 
                    limit=days_back * 96  # Assuming 15-minute intervals
                )
                
                if len(metrics) < 10:  # Need minimum data
                    continue
                
                # Convert to DataFrame and process
                df = pd.DataFrame(metrics)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Pivot metrics by type
                pivot_data = df.pivot_table(
                    index='timestamp',
                    columns='metric_type',
                    values='value',
                    aggfunc='first'
                ).reset_index()
                
                # Add protocol info
                pivot_data['protocol_id'] = protocol['id']
                pivot_data['protocol_name'] = protocol['name']
                
                # Calculate derived features
                pivot_data = self._calculate_anomaly_features(pivot_data)
                
                all_data.append(pivot_data)
            
            if not all_data:
                self.logger.warning("No training data available, creating synthetic data")
                return self._create_synthetic_anomaly_data()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Clean and prepare
            combined_data = combined_data.dropna()
            self.logger.info(f"Prepared training data with {len(combined_data)} samples")
            
            return combined_data
            
        except Exception as e:
            log_error_with_context(e, {
                "protocol_ids": protocol_ids,
                "days_back": days_back
            })
            # Fall back to synthetic data
            return self._create_synthetic_anomaly_data()
    
    def _create_synthetic_anomaly_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic data for anomaly detection training.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic training data
        """
        try:
            np.random.seed(42)
            
            # Generate normal protocol behavior
            normal_data = {
                'tvl': np.random.lognormal(15, 1, int(n_samples * 0.9)),
                'volume_24h': np.random.lognormal(13, 1.5, int(n_samples * 0.9)),
                'price_change_24h': np.random.normal(0.01, 0.05, int(n_samples * 0.9)),
                'volume_change_24h': np.random.normal(0.02, 0.15, int(n_samples * 0.9)),
                'tvl_volatility': np.random.exponential(0.03, int(n_samples * 0.9)),
                'price_volatility': np.random.exponential(0.05, int(n_samples * 0.9)),
                'volume_tvl_ratio': np.random.exponential(0.1, int(n_samples * 0.9))
            }
            
            # Generate anomalous behavior (10% of data)
            n_anomalies = int(n_samples * 0.1)
            anomaly_data = {
                'tvl': np.concatenate([
                    np.random.lognormal(15, 1, n_anomalies // 2),  # Normal-ish
                    np.random.uniform(1e6, 1e10, n_anomalies // 2)  # Extreme values
                ]),
                'volume_24h': np.concatenate([
                    np.random.lognormal(16, 2, n_anomalies // 2),  # High volume spikes
                    np.random.uniform(1e3, 1e6, n_anomalies // 2)  # Low volume
                ]),
                'price_change_24h': np.concatenate([
                    np.random.normal(0.5, 0.2, n_anomalies // 2),  # Large price moves
                    np.random.normal(-0.4, 0.15, n_anomalies // 2)
                ]),
                'volume_change_24h': np.random.normal(0, 0.8, n_anomalies),  # High volatility
                'tvl_volatility': np.random.exponential(0.2, n_anomalies),  # High TVL volatility
                'price_volatility': np.random.exponential(0.3, n_anomalies),  # High price volatility
                'volume_tvl_ratio': np.random.exponential(1.0, n_anomalies)  # Unusual ratios
            }
            
            # Combine normal and anomalous data
            combined_data = {}
            for feature in normal_data.keys():
                combined_data[feature] = np.concatenate([
                    normal_data[feature],
                    anomaly_data[feature]
                ])
            
            # Add metadata
            combined_data['protocol_id'] = np.random.randint(1, 20, n_samples)
            combined_data['timestamp'] = pd.date_range(
                start='2023-01-01', 
                periods=n_samples, 
                freq='15min'
            )
            
            # Create labels (0 = normal, 1 = anomaly)
            labels = np.concatenate([
                np.zeros(int(n_samples * 0.9)),
                np.ones(n_anomalies)
            ])
            combined_data['is_anomaly'] = labels
            
            df = pd.DataFrame(combined_data)
            
            # Shuffle the data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            self.logger.info(f"Created synthetic anomaly detection data with {len(df)} samples")
            return df
            
        except Exception as e:
            log_error_with_context(e, {"n_samples": n_samples})
            raise
    
    def _calculate_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features relevant for anomaly detection.
        
        Args:
            df: DataFrame with raw metrics
            
        Returns:
            DataFrame with calculated features
        """
        try:
            # Calculate rolling statistics for anomaly detection
            for col in ['tvl', 'volume_24h', 'price_usd']:
                if col in df.columns:
                    # Rolling mean and std
                    df[f'{col}_rolling_mean'] = df[col].rolling(window=24, min_periods=1).mean()
                    df[f'{col}_rolling_std'] = df[col].rolling(window=24, min_periods=1).std()
                    
                    # Z-score (standardized deviation from rolling mean)
                    df[f'{col}_zscore'] = (df[col] - df[f'{col}_rolling_mean']) / (df[f'{col}_rolling_std'] + 1e-8)
                    
                    # Percentage change
                    df[f'{col}_pct_change'] = df[col].pct_change()
            
            # Cross-metric features
            if 'tvl' in df.columns and 'volume_24h' in df.columns:
                df['volume_tvl_ratio'] = df['volume_24h'] / (df['tvl'] + 1e-8)
            
            return df
            
        except Exception as e:
            log_error_with_context(e, {"df_shape": df.shape})
            return df
    
    async def train_models(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train anomaly detection models.
        
        Args:
            training_data: Optional training data (will fetch if None)
            
        Returns:
            Dictionary containing training results
        """
        log_function_call("AnomalyDetector.train_models", {
            "training_data_provided": training_data is not None
        })
        
        try:
            # Prepare training data
            if training_data is None:
                training_data = await self.prepare_training_data()
            
            # Select features for training
            feature_cols = [col for col in self.feature_columns if col in training_data.columns]
            
            if len(feature_cols) < 3:
                self.logger.warning(f"Insufficient features for training: {feature_cols}")
                return {"success": False, "error": "Insufficient features"}
            
            X = training_data[feature_cols].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            
            isolation_scores = self.isolation_forest.fit(X_scaled)
            
            # Train Local Outlier Factor
            self.lof_detector = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True  # For prediction on new data
            )
            
            self.lof_detector.fit(X_scaled)
            
            # Evaluate training performance
            if_scores = self.isolation_forest.score_samples(X_scaled)
            lof_scores = self.lof_detector.score_samples(X_scaled)
            
            results = {
                "success": True,
                "training_samples": len(X),
                "features_used": feature_cols,
                "isolation_forest_scores": {
                    "mean": float(if_scores.mean()),
                    "std": float(if_scores.std()),
                    "min": float(if_scores.min()),
                    "max": float(if_scores.max())
                },
                "lof_scores": {
                    "mean": float(lof_scores.mean()),
                    "std": float(lof_scores.std()),
                    "min": float(lof_scores.min()),
                    "max": float(lof_scores.max())
                },
                "training_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Anomaly detection models trained successfully on {len(X)} samples")
            return results
            
        except Exception as e:
            log_error_with_context(e, {
                "training_data_shape": training_data.shape if training_data is not None else None
            })
            return {"success": False, "error": str(e)}
    
    async def detect_anomalies(self, 
                             protocol_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Detect anomalies in protocol data.
        
        Args:
            protocol_data: Protocol metrics data
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            if self.isolation_forest is None or self.lof_detector is None:
                raise ValueError("Models not trained. Call train_models() first.")
            
            # Convert to DataFrame if needed
            if isinstance(protocol_data, dict):
                df = pd.DataFrame([protocol_data])
            else:
                df = protocol_data.copy()
            
            # Select features
            feature_cols = [col for col in self.feature_columns if col in df.columns]
            
            if len(feature_cols) < 3:
                return {"error": "Insufficient features for anomaly detection"}
            
            X = df[feature_cols].copy()
            X = X.fillna(X.mean())
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get anomaly scores
            if_scores = self.isolation_forest.score_samples(X_scaled)
            if_predictions = self.isolation_forest.predict(X_scaled)
            
            lof_scores = self.lof_detector.score_samples(X_scaled)
            lof_predictions = self.lof_detector.predict(X_scaled)
            
            # Combine results
            results = []
            for i in range(len(df)):
                is_anomaly_if = if_predictions[i] == -1
                is_anomaly_lof = lof_predictions[i] == -1
                
                # Combined anomaly decision
                is_anomaly = is_anomaly_if or is_anomaly_lof
                
                # Anomaly severity (lower scores = more anomalous)
                severity = "low"
                if if_scores[i] < -0.3 or lof_scores[i] < -2.0:
                    severity = "medium"
                if if_scores[i] < -0.5 or lof_scores[i] < -3.0:
                    severity = "high"
                if if_scores[i] < -0.7 or lof_scores[i] < -4.0:
                    severity = "critical"
                
                result = {
                    "is_anomaly": bool(is_anomaly),
                    "isolation_forest_score": float(if_scores[i]),
                    "lof_score": float(lof_scores[i]),
                    "severity": severity,
                    "confidence": float(max(abs(if_scores[i]), abs(lof_scores[i] / 4.0))),
                    "features_analyzed": feature_cols,
                    "detection_timestamp": datetime.utcnow().isoformat()
                }
                
                # Add explanation
                if is_anomaly:
                    result["explanation"] = self._generate_anomaly_explanation(
                        X.iloc[i], if_scores[i], lof_scores[i]
                    )
                
                results.append(result)
            
            return {
                "success": True,
                "anomalies_detected": sum(1 for r in results if r["is_anomaly"]),
                "total_samples": len(results),
                "results": results[0] if len(results) == 1 else results
            }
            
        except Exception as e:
            log_error_with_context(e, {
                "protocol_data_type": type(protocol_data).__name__
            })
            return {"success": False, "error": str(e)}
    
    def _generate_anomaly_explanation(self, features: pd.Series, 
                                    if_score: float, lof_score: float) -> Dict[str, Any]:
        """
        Generate explanation for detected anomaly.
        
        Args:
            features: Feature values for the anomalous sample
            if_score: Isolation Forest score
            lof_score: LOF score
            
        Returns:
            Dictionary containing anomaly explanation
        """
        try:
            explanation = {
                "primary_factors": [],
                "recommendations": []
            }
            
            # Analyze feature values for explanation
            for feature, value in features.items():
                if pd.isna(value):
                    continue
                
                if feature == 'tvl' and value > 1e10:
                    explanation["primary_factors"].append("Unusually high Total Value Locked")
                elif feature == 'volume_24h' and value > 1e9:
                    explanation["primary_factors"].append("Extremely high 24h trading volume")
                elif feature == 'price_change_24h' and abs(value) > 0.5:
                    explanation["primary_factors"].append("Large price movement in 24h")
                elif feature == 'tvl_volatility' and value > 0.2:
                    explanation["primary_factors"].append("High TVL volatility detected")
            
            # Add recommendations based on severity
            if if_score < -0.5:
                explanation["recommendations"].append("Investigate protocol for potential issues")
                explanation["recommendations"].append("Monitor closely for exploit indicators")
            
            if lof_score < -3.0:
                explanation["recommendations"].append("Check for market manipulation signs")
                explanation["recommendations"].append("Verify data source accuracy")
            
            return explanation
            
        except Exception as e:
            log_error_with_context(e, {"if_score": if_score, "lof_score": lof_score})
            return {"error": "Failed to generate explanation"}
    
    def save_models(self, filepath: Optional[str] = None) -> str:
        """
        Save trained anomaly detection models.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where models were saved
        """
        try:
            if not filepath:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filepath = self.models_dir / f"anomaly_detector_{timestamp}.pkl"
            
            save_data = {
                'isolation_forest': self.isolation_forest,
                'lof_detector': self.lof_detector,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'contamination': self.contamination,
                'anomaly_threshold': self.anomaly_threshold,
                'lof_threshold': self.lof_threshold,
                'version': '1.0.0',
                'save_timestamp': datetime.utcnow().isoformat()
            }
            
            joblib.dump(save_data, filepath)
            self.logger.info(f"Anomaly detection models saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            log_error_with_context(e, {"filepath": filepath})
            raise
    
    def load_models(self, filepath: str) -> bool:
        """
        Load trained anomaly detection models.
        
        Args:
            filepath: Path to saved models
            
        Returns:
            True if loaded successfully
        """
        try:
            save_data = joblib.load(filepath)
            
            self.isolation_forest = save_data['isolation_forest']
            self.lof_detector = save_data['lof_detector']
            self.scaler = save_data['scaler']
            self.feature_columns = save_data['feature_columns']
            self.contamination = save_data.get('contamination', self.contamination)
            self.anomaly_threshold = save_data.get('anomaly_threshold', self.anomaly_threshold)
            self.lof_threshold = save_data.get('lof_threshold', self.lof_threshold)
            
            self.logger.info(f"Anomaly detection models loaded from: {filepath}")
            return True
            
        except Exception as e:
            log_error_with_context(e, {"filepath": filepath})
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current models.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'models_trained': self.isolation_forest is not None and self.lof_detector is not None,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'anomaly_threshold': self.anomaly_threshold,
            'lof_threshold': self.lof_threshold,
            'has_scaler': self.scaler is not None
        }
