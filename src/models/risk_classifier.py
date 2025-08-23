"""
Risk Classifier for SafeFi DeFi Risk Assessment Agent.

This module implements the main risk classification system that uses
the best performing algorithm selected through model comparison.
"""

import asyncio
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path
from loguru import logger

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ..utils.logger import get_logger, log_function_call, log_error_with_context
from ..database.postgres_manager import PostgreSQLManager
from .model_evaluator import ModelEvaluator


class RiskClassifier:
    """
    Main risk classification system for DeFi protocols.
    
    This class implements risk classification using the best performing
    algorithm selected through comprehensive model evaluation.
    """
    
    def __init__(self):
        """Initialize RiskClassifier."""
        self.logger = get_logger("RiskClassifier")
        self.db_manager = PostgreSQLManager()
        self.model_evaluator = ModelEvaluator()
        
        # Model components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.model_info = None
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.25,
            'medium': 0.50,
            'high': 0.75
        }
        
        # Models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    async def prepare_training_data(self, use_synthetic: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for model training.
        
        Args:
            use_synthetic: Whether to use synthetic data for initial training
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        log_function_call("RiskClassifier.prepare_training_data", {"use_synthetic": use_synthetic})
        
        try:
            if use_synthetic:
                return self._create_synthetic_training_data()
            else:
                # Try to get real data from database
                return await self._get_real_training_data()
                
        except Exception as e:
            log_error_with_context(e, {"use_synthetic": use_synthetic})
            self.logger.warning("Falling back to synthetic training data")
            return self._create_synthetic_training_data()
    
    def _create_synthetic_training_data(self, n_samples: int = 1500) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create realistic synthetic training data for DeFi risk assessment.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        try:
            np.random.seed(42)  # For reproducibility
            
            # Generate realistic DeFi protocol features
            data = {
                # TVL features
                'tvl_volatility': np.random.exponential(0.05, n_samples),  # TVL volatility
                'tvl_change_7d': np.random.normal(0.02, 0.20, n_samples),  # Weekly TVL change
                'tvl_change_30d': np.random.normal(0.08, 0.35, n_samples), # Monthly TVL change
                'tvl_trend': np.random.normal(0.01, 0.15, n_samples),      # TVL trend
                
                # Price features  
                'price_volatility': np.random.exponential(0.08, n_samples), # Price volatility
                'price_return_1d': np.random.normal(0.001, 0.05, n_samples), # Daily returns
                'price_trend': np.random.normal(0.005, 0.10, n_samples),    # Price trend
                'price_rsi': np.random.uniform(20, 80, n_samples),          # RSI indicator
                
                # Volume features
                'volume_consistency': np.random.beta(3, 2, n_samples),      # Volume consistency
                'volume_trend': np.random.normal(0.02, 0.12, n_samples),   # Volume trend
                'volume_spikes': np.random.exponential(0.05, n_samples),   # Volume spikes
                
                # Liquidity features
                'volume_tvl_ratio': np.random.exponential(0.15, n_samples), # Volume/TVL ratio
                'avg_volume_tvl_ratio_7d': np.random.exponential(0.12, n_samples),
                
                # Smart contract features
                'technical_risk': np.random.beta(2, 5, n_samples),         # Technical risk (lower is better)
                'maturity_risk': np.random.beta(3, 4, n_samples),          # Maturity risk
                'governance_risk': np.random.beta(2, 4, n_samples),        # Governance risk
                'historical_risk': np.random.beta(1, 8, n_samples),       # Historical exploit risk
            }
            
            df = pd.DataFrame(data)
            
            # Clip extreme values to realistic ranges
            df['tvl_volatility'] = np.clip(df['tvl_volatility'], 0, 0.5)
            df['price_volatility'] = np.clip(df['price_volatility'], 0, 1.0)
            df['volume_tvl_ratio'] = np.clip(df['volume_tvl_ratio'], 0, 2.0)
            df['volume_spikes'] = np.clip(df['volume_spikes'], 0, 0.3)
            
            # Create realistic risk labels based on multiple factors
            risk_scores = (
                df['tvl_volatility'] * 0.15 +
                np.abs(df['tvl_change_7d']) * 0.10 +
                df['price_volatility'] * 0.20 +
                df['technical_risk'] * 0.25 +
                df['maturity_risk'] * 0.10 +
                df['governance_risk'] * 0.10 +
                df['historical_risk'] * 0.10
            )
            
            # Add some noise to make it more realistic
            risk_scores += np.random.normal(0, 0.05, n_samples)
            risk_scores = np.clip(risk_scores, 0, 1)
            
            # Convert to risk categories with realistic distribution
            conditions = [
                risk_scores <= 0.25,           # Low risk: 25%
                (risk_scores > 0.25) & (risk_scores <= 0.45),  # Medium risk: 35%
                (risk_scores > 0.45) & (risk_scores <= 0.70),  # High risk: 30%
                risk_scores > 0.70              # Critical risk: 10%
            ]
            
            choices = ['Low', 'Medium', 'High', 'Critical']
            labels = pd.Series(np.select(conditions, choices, default='Medium'))
            
            # Store feature columns for later use
            self.feature_columns = df.columns.tolist()
            
            self.logger.info(f"Created synthetic dataset with {len(df)} samples")
            self.logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
            
            return df, labels
            
        except Exception as e:
            log_error_with_context(e, {"n_samples": n_samples})
            raise
    
    async def _get_real_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get real training data from database.
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        try:
            # This would be implemented to fetch real data from your database
            # For now, fall back to synthetic data
            self.logger.info("Real training data not available, using synthetic data")
            return self._create_synthetic_training_data()
            
        except Exception as e:
            log_error_with_context(e, {})
            raise
    
    async def train_and_evaluate_models(self, use_synthetic: bool = True) -> str:
        """
        Train and evaluate multiple models, selecting the best performer.
        
        Args:
            use_synthetic: Whether to use synthetic training data
            
        Returns:
            Name of the best performing model
        """
        log_function_call("RiskClassifier.train_and_evaluate_models", {"use_synthetic": use_synthetic})
        
        try:
            # Prepare training data
            X, y = await self.prepare_training_data(use_synthetic)
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Perform comprehensive model comparison
            self.logger.info("Starting comprehensive algorithm evaluation...")
            comparison_results = await self.model_evaluator.compare_algorithms(X, y_encoded)
            
            # Extract best model information
            best_model_name = comparison_results['best_model']
            best_model_data = comparison_results['comparison_results'][best_model_name]
            
            # Store model components
            self.model = best_model_data['trained_model']
            self.scaler = comparison_results['scaler']
            self.model_info = {
                'model_name': best_model_name,
                'evaluation_results': comparison_results['comparison_results'],
                'reasoning': comparison_results['reasoning'],
                'training_date': datetime.utcnow().isoformat()
            }
            
            # Save the best model
            model_path = await self.model_evaluator.save_best_model(
                comparison_results, best_model_name
            )
            
            self.logger.info(f"Model training completed. Best model: {best_model_name}")
            self.logger.info(f"Model saved to: {model_path}")
            
            return best_model_name
            
        except Exception as e:
            log_error_with_context(e, {"use_synthetic": use_synthetic})
            raise
    
    async def predict_risk(self, features: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Predict risk level for given features.
        
        Args:
            features: Feature dictionary or DataFrame
            
        Returns:
            Dictionary containing risk prediction and confidence
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train_and_evaluate_models() first.")
            
            # Convert to DataFrame if needed
            if isinstance(features, dict):
                feature_df = pd.DataFrame([features])
            else:
                feature_df = features.copy()
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0  # Default value for missing features
            
            # Reorder columns to match training data
            feature_df = feature_df[self.feature_columns]
            
            # Handle missing values
            feature_df = feature_df.fillna(0)
            
            # Apply scaling if needed
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_df)
            else:
                feature_array = feature_df.values
            
            # Make predictions
            risk_proba = self.model.predict_proba(feature_array)[0]
            risk_class_encoded = self.model.predict(feature_array)
            
            # Convert back to risk level
            risk_level = self.label_encoder.inverse_transform([risk_class_encoded])
            
            # Get class probabilities
            risk_classes = self.label_encoder.classes_
            prob_dict = {
                self.label_encoder.inverse_transform([i]): float(risk_proba[i])
                for i in range(len(risk_classes))
            }
            
            # Calculate overall risk score (0-100)
            risk_weights = {'Low': 0, 'Medium': 35, 'High': 70, 'Critical': 95}
            overall_risk_score = sum(prob_dict[level] * risk_weights[level] for level in prob_dict)
            
            # Generate explanation
            explanation = self._generate_risk_explanation(features, risk_level, prob_dict)
            
            prediction_result = {
                'risk_level': risk_level,
                'overall_risk_score': float(overall_risk_score),
                'confidence': float(max(risk_proba)),
                'class_probabilities': prob_dict,
                'explanation': explanation,
                'model_used': self.model_info['model_name'] if self.model_info else 'Unknown',
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
            return prediction_result
            
        except Exception as e:
            log_error_with_context(e, {
                "features_type": type(features).__name__,
                "model_available": self.model is not None
            })
            raise
    
    def _generate_risk_explanation(self, features: Union[Dict, pd.DataFrame], 
                                 risk_level: str, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate explanation for risk prediction.
        
        Args:
            features: Input features
            risk_level: Predicted risk level
            probabilities: Class probabilities
            
        Returns:
            Dictionary containing explanation
        """
        try:
            if isinstance(features, pd.DataFrame):
                features = features.iloc[0].to_dict()
            
            explanation = {
                'risk_level': risk_level,
                'key_factors': [],
                'recommendations': []
            }
            
            # Analyze key risk factors
            if features.get('tvl_volatility', 0) > 0.1:
                explanation['key_factors'].append("High TVL volatility indicates instability")
            
            if features.get('price_volatility', 0) > 0.15:
                explanation['key_factors'].append("High price volatility suggests market uncertainty")
            
            if features.get('technical_risk', 0) > 0.5:
                explanation['key_factors'].append("Elevated technical risk from smart contract factors")
            
            if features.get('volume_tvl_ratio', 0) < 0.05:
                explanation['key_factors'].append("Low trading volume relative to TVL")
            
            # Generate recommendations based on risk level
            if risk_level == 'Low':
                explanation['recommendations'] = [
                    "Protocol shows stable metrics",
                    "Continue monitoring for any changes",
                    "Suitable for conservative investment strategies"
                ]
            elif risk_level == 'Medium':
                explanation['recommendations'] = [
                    "Monitor protocol metrics more closely",
                    "Consider position sizing carefully",
                    "Review recent protocol updates and news"
                ]
            elif risk_level == 'High':
                explanation['recommendations'] = [
                    "Exercise caution with this protocol",
                    "Consider reducing position size",
                    "Set up alerts for key metrics"
                ]
            else:  # Critical
                explanation['recommendations'] = [
                    "Strong recommendation to avoid or exit positions",
                    "High probability of significant losses",
                    "Wait for risk factors to improve before considering"
                ]
            
            return explanation
            
        except Exception as e:
            log_error_with_context(e, {"risk_level": risk_level})
            return {"error": "Failed to generate explanation"}
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save trained model to file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where model was saved
        """
        try:
            if not filepath:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filepath = self.models_dir / f"risk_classifier_{timestamp}.pkl"
            
            save_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'model_info': self.model_info,
                'risk_thresholds': self.risk_thresholds,
                'version': '1.0.0',
                'save_timestamp': datetime.utcnow().isoformat()
            }
            
            joblib.dump(save_data, filepath)
            self.logger.info(f"Model saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            log_error_with_context(e, {"filepath": filepath})
            raise
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            True if loaded successfully
        """
        try:
            save_data = joblib.load(filepath)
            
            self.model = save_data['model']
            self.scaler = save_data.get('scaler')
            self.label_encoder = save_data['label_encoder']
            self.feature_columns = save_data['feature_columns']
            self.model_info = save_data.get('model_info', {})
            self.risk_thresholds = save_data.get('risk_thresholds', self.risk_thresholds)
            
            self.logger.info(f"Model loaded from: {filepath}")
            return True
            
        except Exception as e:
            log_error_with_context(e, {"filepath": filepath})
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_loaded': self.model is not None,
            'model_info': self.model_info,
            'feature_columns': self.feature_columns,
            'risk_thresholds': self.risk_thresholds,
            'has_scaler': self.scaler is not None
        }
            