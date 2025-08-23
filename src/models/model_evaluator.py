"""
ML Model Evaluator for SafeFi DeFi Risk Assessment Agent.

This module provides comprehensive algorithm comparison and evaluation
capabilities for selecting the best performing risk classification model.
"""

import asyncio
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path
from loguru import logger

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

# Safe imports with better error handling
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info("LightGBM loaded successfully")
except (ImportError, ValueError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    logger.warning(f"LightGBM not available: {e}")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    logger.info("CatBoost loaded successfully")
except (ImportError, ValueError, OSError) as e:
    CATBOOST_AVAILABLE = False
    logger.warning(f"CatBoost not available (this is OK): {e}")

from ..utils.logger import get_logger, log_function_call, log_error_with_context
from ..config.settings import get_settings


class ModelEvaluator:
    """
    Comprehensive ML model evaluation and comparison system.
    
    This class implements algorithm comparison functionality to select
    the best performing model for DeFi risk classification with detailed
    performance analysis and reasoning.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.settings = get_settings()
        self.logger = get_logger("ModelEvaluator")
        self.models_config = {}
        self.evaluation_results = {}
        self.best_model_info = {}
        
        # Create models directory if it doesn't exist
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    def prepare_algorithms(self) -> Dict[str, Dict]:
        """
        Prepare different ML algorithms for comprehensive comparison.
        
        Returns:
            Dictionary of algorithm configurations with hyperparameters
        """
        algorithms = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'needs_scaling': False,
                'strengths': [
                    "Handles mixed data types excellently",
                    "Provides feature importance for explainability", 
                    "Robust to outliers in DeFi metrics",
                    "Good performance with limited training data",
                    "Natural handling of missing values"
                ]
            },
            
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    verbosity=0
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                },
                'needs_scaling': False,
                'strengths': [
                    "Excellent performance on structured data",
                    "Built-in regularization prevents overfitting",
                    "Handles missing values automatically",
                    "Fast training and prediction",
                    "Strong performance on financial data"
                ]
            },
            
            'Support Vector Machine': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'needs_scaling': True,
                'strengths': [
                    "Strong theoretical foundation",
                    "Effective in high-dimensional spaces",
                    "Memory efficient",
                    "Versatile with different kernel functions",
                    "Good performance on small datasets"
                ]
            },
            
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    solver='liblinear'
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2']
                },
                'needs_scaling': True,
                'strengths': [
                    "Fast and simple implementation",
                    "Provides probability estimates",
                    "No hyperparameter tuning needed",
                    "Good baseline performance",
                    "Interpretable coefficients"
                ]
            },
            
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                'needs_scaling': False,
                'strengths': [
                    "Highly interpretable decisions",
                    "No assumptions about data distribution",
                    "Handles both numerical and categorical data",
                    "Automatic feature selection",
                    "Fast prediction"
                ]
            }
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            algorithms['LightGBM'] = {
                'model': lgb.LGBMClassifier(
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                },
                'needs_scaling': False,
                'strengths': [
                    "Memory efficient for real-time applications",
                    "Fast training suitable for frequent retraining",
                    "Good performance on small datasets",
                    "Native categorical feature support",
                    "Lower memory usage than XGBoost"
                ]
            }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            algorithms['CatBoost'] = {
                'model': CatBoostClassifier(
                    random_state=42,
                    verbose=False,
                    allow_writing_files=False
                ),
                'params': {
                    'iterations': [100, 200, 300],
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'needs_scaling': False,
                'strengths': [
                    "Excellent handling of categorical features",
                    "Robust to overfitting",
                    "No need for extensive hyperparameter tuning",
                    "Good performance out-of-the-box",
                    "Built-in cross-validation"
                ]
            }
        
        self.models_config = algorithms
        return algorithms
    
    async def compare_algorithms(self, X: pd.DataFrame, y: pd.Series, 
                                test_size: float = 0.2) -> Dict[str, Any]:
        """
        Perform comprehensive algorithm comparison and evaluation.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Test set proportion
            
        Returns:
            Dictionary containing comparison results and best model selection
        """
        log_function_call("ModelEvaluator.compare_algorithms", {
            "X_shape": X.shape,
            "y_unique": y.unique().tolist(),
            "test_size": test_size
        })
        
        try:
            # Prepare algorithms
            if not self.models_config:
                self.prepare_algorithms()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Prepare scaled versions for algorithms that need scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            comparison_results = {}
            
            self.logger.info("Starting comprehensive ML algorithm comparison...")
            self.logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
            self.logger.info(f"Classes: {y.value_counts().to_dict()}")
            self.logger.info("=" * 80)
            
            # Evaluate each algorithm
            for name, config in self.models_config.items():
                try:
                    self.logger.info(f"Evaluating {name}...")
                    
                    # Choose appropriate data (scaled or original)
                    if config['needs_scaling']:
                        X_tr, X_te = X_train_scaled, X_test_scaled
                    else:
                        X_tr, X_te = X_train.values, X_test.values
                    
                    # Perform grid search with cross-validation
                    grid_search = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    # Fit and evaluate
                    grid_search.fit(X_tr, y_train)
                    best_model = grid_search.best_estimator_
                    
                    # Detailed evaluation
                    results = await self._evaluate_model(
                        best_model, X_tr, X_te, y_train, y_test, name
                    )
                    
                    # Add configuration info
                    results.update({
                        'best_params': grid_search.best_params_,
                        'best_cv_score': grid_search.best_score_,
                        'needs_scaling': config['needs_scaling'],
                        'strengths': config['strengths']
                    })
                    
                    comparison_results[name] = results
                    
                    # Log results
                    self.logger.info(f"✓ {name} completed:")
                    self.logger.info(f"  - CV Accuracy: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
                    self.logger.info(f"  - Test Accuracy: {results['test_accuracy']:.3f}")
                    self.logger.info(f"  - AUC Score: {results['auc_score']:.3f}")
                    self.logger.info(f"  - Best Params: {grid_search.best_params_}")
                    
                except Exception as e:
                    self.logger.error(f"✗ {name} failed: {e}")
                    continue
            
            # Select best model and provide reasoning
            best_model_name, reasoning = self._select_best_model(comparison_results)
            
            # Store results
            self.evaluation_results = comparison_results
            self.best_model_info = {
                'name': best_model_name,
                'results': comparison_results[best_model_name],
                'reasoning': reasoning,
                'evaluation_timestamp': datetime.utcnow().isoformat()
            }
            
            # Print final summary
            self._print_comparison_summary(comparison_results, best_model_name, reasoning)
            
            return {
                'comparison_results': comparison_results,
                'best_model': best_model_name,
                'reasoning': reasoning,
                'scaler': scaler if any(c['needs_scaling'] for c in self.models_config.values()) else None
            }
            
        except Exception as e:
            log_error_with_context(e, {
                "X_shape": X.shape if hasattr(X, 'shape') else 'unknown',
                "y_shape": y.shape if hasattr(y, 'shape') else 'unknown'
            })
            raise
    
    async def _evaluate_model(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray, 
                            model_name: str) -> Dict[str, Any]:
        """
        Perform detailed model evaluation with multiple metrics.
        
        Args:
            model: Trained model instance
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            model_name: Name of the model
            
        Returns:
            Dictionary containing detailed evaluation metrics
        """
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # Classification metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # AUC score (multi-class)
            try:
                auc_score = roc_auc_score(
                    y_test, y_pred_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
            except Exception:
                auc_score = 0.0
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            return {
                'model_name': model_name,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'test_accuracy': float(test_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_score': float(auc_score),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'trained_model': model
            }
            
        except Exception as e:
            log_error_with_context(e, {"model_name": model_name})
            raise
    
    def _select_best_model(self, results: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Select best model and provide detailed reasoning.
        
        Args:
            results: Comparison results dictionary
            
        Returns:
            Tuple of (best_model_name, reasoning_points)
        """
        try:
            # Calculate composite scores for each model
            model_scores = {}
            
            for name, result in results.items():
                # Weighted composite score
                composite_score = (
                    result['cv_mean'] * 0.35 +          # Cross-validation accuracy
                    result['test_accuracy'] * 0.25 +    # Test accuracy
                    result['auc_score'] * 0.20 +        # AUC score
                    result['f1_score'] * 0.20           # F1 score
                )
                
                # Stability penalty (high variance reduces score)
                stability_penalty = result['cv_std'] * 0.5
                final_score = composite_score - stability_penalty
                
                model_scores[name] = final_score
            
            # Select best model
            best_model = max(model_scores.keys(), key=lambda x: model_scores[x])
            best_result = results[best_model]
            
            # Generate reasoning
            reasoning = []
            
            # Performance-based reasons
            if best_result['cv_mean'] > 0.75:
                reasoning.append(f"High cross-validation accuracy ({best_result['cv_mean']:.3f})")
            
            if best_result['cv_std'] < 0.05:
                reasoning.append(f"Low variance across folds ({best_result['cv_std']:.3f}) indicating stability")
            
            if best_result['auc_score'] > 0.8:
                reasoning.append(f"Excellent AUC score ({best_result['auc_score']:.3f})")
            
            if best_result['f1_score'] > 0.75:
                reasoning.append(f"Strong F1 score ({best_result['f1_score']:.3f}) showing balanced precision/recall")
            
            # Algorithm-specific reasons
            if best_model in self.models_config:
                algorithm_strengths = self.models_config[best_model]['strengths']
                reasoning.extend([f"Algorithm strength: {strength}" for strength in algorithm_strengths[:3]])
            
            # Comparative reasons
            second_best = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)[1]
            performance_gap = model_scores[best_model] - model_scores[second_best]
            if performance_gap > 0.02:
                reasoning.append(f"Significantly outperforms second-best model ({second_best}) by {performance_gap:.3f}")
            
            return best_model, reasoning
            
        except Exception as e:
            log_error_with_context(e, {"results_keys": list(results.keys())})
            # Fallback to highest CV accuracy
            return max(results.keys(), key=lambda x: results[x]['cv_mean']), ["Fallback selection based on CV accuracy"]
    
    def _print_comparison_summary(self, results: Dict[str, Any], best_model: str, 
                                reasoning: List[str]) -> None:
        """
        Print comprehensive comparison summary.
        
        Args:
            results: Comparison results
            best_model: Best model name
            reasoning: Selection reasoning
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎯 ALGORITHM COMPARISON RESULTS")
        self.logger.info("=" * 80)
        
        # Results table
        self.logger.info(f"{'Algorithm':<20} {'CV Acc':<8} {'Test Acc':<9} {'AUC':<6} {'F1':<6} {'Status'}")
        self.logger.info("-" * 80)
        
        # Sort by performance
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['cv_mean'],
            reverse=True
        )
        
        for name, result in sorted_results:
            status = "🏆 BEST" if name == best_model else "   "
            self.logger.info(
                f"{name:<20} {result['cv_mean']:.3f}    {result['test_accuracy']:.3f}     "
                f"{result['auc_score']:.3f}  {result['f1_score']:.3f}  {status}"
            )
        
        # Best model reasoning
        self.logger.info(f"\n🎯 SELECTED ALGORITHM: {best_model}")
        self.logger.info("=" * 80)
        self.logger.info("SELECTION REASONING:")
        for i, reason in enumerate(reasoning, 1):
            self.logger.info(f"  {i}. {reason}")
        
        self.logger.info(f"\n✅ {best_model} will be used for production DeFi risk assessment.")
        self.logger.info("=" * 80)
    
    async def save_best_model(self, model_data: Dict[str, Any], 
                            model_name: str) -> str:
        """
        Save the best performing model and associated data.
        
        Args:
            model_data: Model data dictionary
            model_name: Name of the best model
            
        Returns:
            Path to saved model file
        """
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_filename = f"best_model_{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
            model_path = self.models_dir / model_filename
            
            # Prepare save data
            save_data = {
                'model': model_data['comparison_results'][model_name]['trained_model'],
                'model_name': model_name,
                'scaler': model_data.get('scaler'),
                'evaluation_results': model_data['comparison_results'],
                'reasoning': model_data['reasoning'],
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            }
            
            # Save model
            joblib.dump(save_data, model_path)
            
            self.logger.info(f"Best model saved to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            log_error_with_context(e, {"model_name": model_name})
            raise
    
    def get_model_comparison_summary(self) -> Dict[str, Any]:
        """
        Get summary of model comparison results.
        
        Returns:
            Summary dictionary
        """
        return {
            'evaluation_results': self.evaluation_results,
            'best_model_info': self.best_model_info,
            'algorithms_compared': list(self.models_config.keys()),
            'evaluation_timestamp': datetime.utcnow().isoformat()
        }
