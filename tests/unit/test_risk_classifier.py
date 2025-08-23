"""Tests for RiskClassifier - Fixed Version."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from src.models.risk_classifier import RiskClassifier


@pytest.fixture
def risk_classifier():
    """Create RiskClassifier instance for testing."""
    return RiskClassifier()


@pytest.fixture
def sample_features():
    """Sample features for risk prediction testing."""
    return {
        'tvl_volatility': 0.05,
        'tvl_change_7d': 0.02,
        'price_volatility': 0.08,
        'volume_tvl_ratio': 0.12,
        'technical_risk': 0.3,
        'maturity_risk': 0.2,
        'governance_risk': 0.25,
        'historical_risk': 0.1
    }


class TestRiskClassifier:
    def test_initialization(self, risk_classifier):
        """Test RiskClassifier initializes correctly."""
        assert risk_classifier.model is None
        assert risk_classifier.scaler is None
        assert risk_classifier.label_encoder is None
        assert 'low' in risk_classifier.risk_thresholds
        assert 'medium' in risk_classifier.risk_thresholds
        assert 'high' in risk_classifier.risk_thresholds
    
    def test_create_synthetic_data(self, risk_classifier):
        """Test synthetic training data creation."""
        X, y = risk_classifier._create_synthetic_training_data(n_samples=100)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 100
        assert len(y) == 100
        assert all(label in ['Low', 'Medium', 'High', 'Critical'] for label in y.unique())
        
        # Check feature columns are stored
        assert risk_classifier.feature_columns == X.columns.tolist()
    
    @pytest.mark.asyncio
    async def test_prepare_training_data(self, risk_classifier):
        """Test training data preparation."""
        X, y = await risk_classifier.prepare_training_data(use_synthetic=True)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) > 0
        assert len(y) > 0
        # FIXED: Compare lengths, not shape
        assert len(X) == len(y)
    
    @pytest.mark.asyncio
    async def test_predict_without_training_raises_error(self, risk_classifier, sample_features):
        """Test that prediction without training raises ValueError."""
        with pytest.raises(ValueError, match="Model not trained"):
            await risk_classifier.predict_risk(sample_features)
    
    def test_generate_risk_explanation(self, risk_classifier):
        """Test risk explanation generation."""
        features = {
            'tvl_volatility': 0.05,
            'price_volatility': 0.08,
            'technical_risk': 0.3
        }
        probabilities = {'Low': 0.1, 'Medium': 0.6, 'High': 0.25, 'Critical': 0.05}
        
        explanation = risk_classifier._generate_risk_explanation(
            features, 'Medium', probabilities
        )
        
        assert isinstance(explanation, dict)
        assert 'risk_level' in explanation
        assert 'key_factors' in explanation
        assert 'recommendations' in explanation
        assert explanation['risk_level'] == 'Medium'
        assert isinstance(explanation['recommendations'], list)
    
    def test_model_info_no_model(self, risk_classifier):
        """Test model info when no model is loaded."""
        info = risk_classifier.get_model_info()
        
        assert 'model_loaded' in info
        assert info['model_loaded'] is False
        assert 'feature_columns' in info
        assert 'risk_thresholds' in info
    
    @pytest.mark.asyncio
    async def test_full_training_pipeline(self, risk_classifier):
        """Test complete training pipeline."""
        # Mock model evaluator to avoid long training time
        risk_classifier.model_evaluator = AsyncMock()
        risk_classifier.model_evaluator.compare_algorithms.return_value = {
            'best_model': 'Random Forest',
            'comparison_results': {
                'Random Forest': {
                    'trained_model': MagicMock(),
                    'cv_mean': 0.85,
                    'test_accuracy': 0.83,
                    'auc_score': 0.87,
                    'f1_score': 0.84
                }
            },
            'reasoning': ['High accuracy'],
            'scaler': None
        }
        risk_classifier.model_evaluator.save_best_model = AsyncMock(return_value='/test/model.pkl')
        
        result = await risk_classifier.train_and_evaluate_models()
        
        assert result == 'Random Forest'
        assert risk_classifier.model is not None
        assert risk_classifier.model_info is not None
    
    @pytest.mark.asyncio
    async def test_predict_with_trained_model(self, risk_classifier, sample_features):
        """Test prediction with a trained model."""
    # Setup mock trained model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.6, 0.3, 0.0]])
        mock_model.predict.return_value = np.array([1])
    
        risk_classifier.model = mock_model
        risk_classifier.feature_columns = list(sample_features.keys())
    
    # FIXED: Robust mock inverse transform function
        def safe_mock_inverse_transform(labels):
            mapping = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
            if isinstance(labels, (list, tuple, np.ndarray)):
            # Handle nested arrays/lists by flattening them
                flat_labels = []
                for lbl in labels:
                    if isinstance(lbl, (list, tuple, np.ndarray)):
                        flat_labels.extend([int(x) for x in lbl])
                    else:
                        flat_labels.append(int(lbl))
                return [mapping[label] for label in flat_labels]
            else:
              return mapping[int(labels)]
    
    # Mock label encoder
        risk_classifier.label_encoder = MagicMock()
        risk_classifier.label_encoder.inverse_transform = safe_mock_inverse_transform
        risk_classifier.label_encoder.classes_ = np.array([0, 1, 2, 3])
    
        prediction = await risk_classifier.predict_risk(sample_features)
    
        assert 'risk_level' in prediction
        assert 'overall_risk_score' in prediction
        assert 'confidence' in prediction
        assert 'class_probabilities' in prediction
        assert isinstance(prediction['overall_risk_score'], float)
        assert 0 <= prediction['overall_risk_score'] <= 100
