"""Tests for AnomalyDetector."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from src.models.anomaly_detector import AnomalyDetector


@pytest.fixture
def anomaly_detector():
    """Create AnomalyDetector for testing."""
    return AnomalyDetector(contamination=0.1)


@pytest.fixture
def sample_protocol_data():
    """Sample protocol data for testing."""
    return {
        'tvl': 1000000,
        'volume_24h': 500000,
        'price_change_24h': 0.05,
        'volume_change_24h': 0.1,
        'tvl_volatility': 0.03,
        'price_volatility': 0.06,
        'volume_tvl_ratio': 0.5
    }


class TestAnomalyDetector:
    def test_initialization(self, anomaly_detector):
        """Test AnomalyDetector initialization."""
        assert anomaly_detector.contamination == 0.1
        assert anomaly_detector.isolation_forest is None
        assert anomaly_detector.lof_detector is None
    
    def test_create_synthetic_anomaly_data(self, anomaly_detector):
        """Test synthetic data creation."""
        df = anomaly_detector._create_synthetic_anomaly_data(n_samples=100)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'is_anomaly' in df.columns
        assert df['is_anomaly'].sum() > 0  # Should have some anomalies
        
        # Check that we have expected features
        expected_features = ['tvl', 'volume_24h', 'price_change_24h']
        for feature in expected_features:
            assert feature in df.columns
    
    def test_calculate_anomaly_features(self, anomaly_detector):
        """Test anomaly feature calculation."""
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='H'),
            'tvl': np.random.uniform(1e6, 1e7, 50),
            'volume_24h': np.random.uniform(1e5, 1e6, 50),
            'price_usd': np.random.uniform(100, 200, 50)
        })
        
        result = anomaly_detector._calculate_anomaly_features(df)
        
        assert 'tvl_rolling_mean' in result.columns
        assert 'tvl_rolling_std' in result.columns
        assert 'tvl_zscore' in result.columns
        assert 'volume_tvl_ratio' in result.columns
    
    @pytest.mark.asyncio
    async def test_train_models_synthetic(self, anomaly_detector):
        """Test model training with synthetic data."""
        # Create synthetic training data
        training_data = anomaly_detector._create_synthetic_anomaly_data(n_samples=200)
        
        # Train models
        result = await anomaly_detector.train_models(training_data)
        
        assert result['success'] is True
        assert result['training_samples'] == 200
        assert anomaly_detector.isolation_forest is not None
        assert anomaly_detector.lof_detector is not None
        assert anomaly_detector.scaler is not None
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_untrained(self, anomaly_detector, sample_protocol_data):
        """Test anomaly detection with untrained models."""
        result = await anomaly_detector.detect_anomalies(sample_protocol_data)
        
        assert result['success'] is False
        assert 'not trained' in result['error']
    
    def test_generate_anomaly_explanation(self, anomaly_detector):
        """Test anomaly explanation generation."""
        features = pd.Series({
            'tvl': 1e11,  # Very high TVL
            'volume_24h': 1e10,  # Very high volume
            'price_change_24h': 0.6,  # Large price change
            'tvl_volatility': 0.25  # High volatility
        })
        
        explanation = anomaly_detector._generate_anomaly_explanation(
            features, if_score=-0.6, lof_score=-3.5
        )
        
        assert 'primary_factors' in explanation
        assert 'recommendations' in explanation
        assert len(explanation['primary_factors']) > 0
        assert len(explanation['recommendations']) > 0
    
    def test_model_info(self, anomaly_detector):
        """Test model info retrieval."""
        info = anomaly_detector.get_model_info()
        
        assert 'models_trained' in info
        assert 'feature_columns' in info
        assert 'contamination' in info
        assert info['models_trained'] is False  # Not trained yet
