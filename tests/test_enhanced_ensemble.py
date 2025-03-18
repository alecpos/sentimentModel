"""
Tests for the enhanced ensemble implementation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from app.models.ml.prediction.enhanced_ensemble import (
    EnhancedBaggingEnsemble,
    EnhancedStackingEnsemble,
    optimize_ensemble_weights,
    visualize_ensemble_performance
)

@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    return X, y

@pytest.fixture
def train_test_data(sample_data):
    """Split sample data into train and test sets."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def test_enhanced_bagging_ensemble_initialization():
    """Test initialization of EnhancedBaggingEnsemble."""
    base_estimator = DecisionTreeClassifier(random_state=42)
    ensemble = EnhancedBaggingEnsemble(
        base_estimator=base_estimator,
        n_estimators=5,
        random_state=42
    )
    
    assert ensemble.base_estimator == base_estimator
    assert ensemble.n_estimators == 5
    assert ensemble.random_state == 42
    assert len(ensemble.estimators) == 0
    assert ensemble.performance_metrics.inference_time == []
    assert ensemble.performance_metrics.prediction_distribution == []

def test_enhanced_bagging_ensemble_fit(train_test_data):
    """Test fitting of EnhancedBaggingEnsemble."""
    X_train, _, y_train, _ = train_test_data
    base_estimator = DecisionTreeClassifier(random_state=42)
    ensemble = EnhancedBaggingEnsemble(
        base_estimator=base_estimator,
        n_estimators=5,
        random_state=42
    )
    
    ensemble.fit(X_train, y_train)
    
    assert len(ensemble.estimators) == 5
    assert all(isinstance(est, DecisionTreeClassifier) for est in ensemble.estimators)
    assert ensemble.performance_metrics.last_updated is not None

def test_enhanced_bagging_ensemble_predict(train_test_data):
    """Test prediction of EnhancedBaggingEnsemble."""
    X_train, X_test, y_train, y_test = train_test_data
    base_estimator = DecisionTreeClassifier(random_state=42)
    ensemble = EnhancedBaggingEnsemble(
        base_estimator=base_estimator,
        n_estimators=5,
        random_state=42
    )
    
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    
    assert len(predictions) == len(y_test)
    assert all(pred in [0, 1] for pred in predictions)
    assert len(ensemble.performance_metrics.inference_time) > 0
    assert len(ensemble.performance_metrics.prediction_distribution) > 0

def test_enhanced_bagging_ensemble_predict_proba(train_test_data):
    """Test probability prediction of EnhancedBaggingEnsemble."""
    X_train, X_test, y_train, y_test = train_test_data
    base_estimator = DecisionTreeClassifier(random_state=42)
    ensemble = EnhancedBaggingEnsemble(
        base_estimator=base_estimator,
        n_estimators=5,
        random_state=42
    )
    
    ensemble.fit(X_train, y_train)
    probas = ensemble.predict_proba(X_test)
    
    assert probas.shape == (len(y_test), 2)
    assert all(0 <= prob <= 1 for prob in probas.flatten())
    assert np.allclose(probas.sum(axis=1), 1.0)

def test_enhanced_stacking_ensemble_initialization():
    """Test initialization of EnhancedStackingEnsemble."""
    base_estimators = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42)
    ]
    meta_learner = LogisticRegression(random_state=42)
    ensemble = EnhancedStackingEnsemble(
        base_estimators=base_estimators,
        meta_learner=meta_learner,
        use_proba=True,
        n_splits=5,
        random_state=42
    )
    
    assert ensemble.base_estimators == base_estimators
    assert ensemble.meta_learner == meta_learner
    assert ensemble.use_proba is True
    assert ensemble.n_splits == 5
    assert ensemble.random_state == 42
    assert ensemble.performance_metrics.inference_time == []
    assert ensemble.performance_metrics.prediction_distribution == []

def test_enhanced_stacking_ensemble_fit(train_test_data):
    """Test fitting of EnhancedStackingEnsemble."""
    X_train, X_test, y_train, y_test = train_test_data
    base_estimators = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42)
    ]
    meta_learner = LogisticRegression(random_state=42)
    ensemble = EnhancedStackingEnsemble(
        base_estimators=base_estimators,
        meta_learner=meta_learner,
        use_proba=True,
        n_splits=5,
        random_state=42
    )
    
    ensemble.fit(X_train, y_train, X_test, y_test)
    
    assert ensemble.performance_metrics.last_updated is not None
    assert len(ensemble.performance_metrics.cv_scores) > 0

def test_enhanced_stacking_ensemble_predict(train_test_data):
    """Test prediction of EnhancedStackingEnsemble."""
    X_train, X_test, y_train, y_test = train_test_data
    base_estimators = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42)
    ]
    meta_learner = LogisticRegression(random_state=42)
    ensemble = EnhancedStackingEnsemble(
        base_estimators=base_estimators,
        meta_learner=meta_learner,
        use_proba=True,
        n_splits=5,
        random_state=42
    )
    
    ensemble.fit(X_train, y_train, X_test, y_test)
    predictions = ensemble.predict(X_test)
    
    assert len(predictions) == len(y_test)
    assert all(pred in [0, 1] for pred in predictions)
    assert len(ensemble.performance_metrics.inference_time) > 0
    assert len(ensemble.performance_metrics.prediction_distribution) > 0

def test_enhanced_stacking_ensemble_predict_proba(train_test_data):
    """Test probability prediction of EnhancedStackingEnsemble."""
    X_train, X_test, y_train, y_test = train_test_data
    base_estimators = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42)
    ]
    meta_learner = LogisticRegression(random_state=42)
    ensemble = EnhancedStackingEnsemble(
        base_estimators=base_estimators,
        meta_learner=meta_learner,
        use_proba=True,
        n_splits=5,
        random_state=42
    )
    
    ensemble.fit(X_train, y_train, X_test, y_test)
    probas = ensemble.predict_proba(X_test)
    
    assert probas.shape == (len(y_test), 2)
    assert all(0 <= prob <= 1 for prob in probas.flatten())
    assert np.allclose(probas.sum(axis=1), 1.0)

def test_optimize_ensemble_weights(train_test_data):
    """Test ensemble weight optimization."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Create and train base models
    base_models = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        LogisticRegression(random_state=42)
    ]
    
    for model in base_models:
        model.fit(X_train, y_train)
    
    # Optimize weights
    weights = optimize_ensemble_weights(base_models, X_test, y_test)
    
    assert len(weights) == len(base_models)
    assert all(0 <= w <= 1 for w in weights)
    assert np.isclose(np.sum(weights), 1.0)

def test_visualize_ensemble_performance(train_test_data, tmp_path):
    """Test ensemble performance visualization."""
    X_train, X_test, y_train, y_test = train_test_data
    base_estimator = DecisionTreeClassifier(random_state=42)
    ensemble = EnhancedBaggingEnsemble(
        base_estimator=base_estimator,
        n_estimators=5,
        random_state=42
    )
    
    # Train and make predictions to generate metrics
    ensemble.fit(X_train, y_train)
    ensemble.predict(X_test)
    ensemble.predict_proba(X_test)
    
    # Test visualization
    fig = visualize_ensemble_performance(ensemble)
    assert fig is not None
    
    # Test saving to file
    output_dir = str(tmp_path)
    fig = visualize_ensemble_performance(ensemble, output_dir)
    assert (tmp_path / "ensemble_performance.png").exists()

def test_ensemble_performance_metrics(train_test_data):
    """Test performance metrics tracking."""
    X_train, X_test, y_train, y_test = train_test_data
    base_estimator = DecisionTreeClassifier(random_state=42)
    ensemble = EnhancedBaggingEnsemble(
        base_estimator=base_estimator,
        n_estimators=5,
        random_state=42
    )
    
    # Train and make predictions
    ensemble.fit(X_train, y_train)
    ensemble.predict(X_test)
    ensemble.predict_proba(X_test)
    
    metrics = ensemble.performance_metrics
    
    assert len(metrics.inference_time) > 0
    assert len(metrics.prediction_distribution) > 0
    assert metrics.last_updated is not None
    assert all(t > 0 for t in metrics.inference_time)
    assert all(0 <= p <= 1 for p in metrics.prediction_distribution)

def test_ensemble_accuracy(train_test_data):
    """Test ensemble accuracy."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Train bagging ensemble
    bagging = EnhancedBaggingEnsemble(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=5,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    bagging_preds = bagging.predict(X_test)
    bagging_acc = accuracy_score(y_test, bagging_preds)
    
    # Train stacking ensemble
    base_estimators = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42)
    ]
    stacking = EnhancedStackingEnsemble(
        base_estimators=base_estimators,
        meta_learner=LogisticRegression(random_state=42),
        use_proba=True,
        n_splits=5,
        random_state=42
    )
    stacking.fit(X_train, y_train, X_test, y_test)
    stacking_preds = stacking.predict(X_test)
    stacking_acc = accuracy_score(y_test, stacking_preds)
    
    # Both ensembles should perform better than random
    assert bagging_acc > 0.5
    assert stacking_acc > 0.5 