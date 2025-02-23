"""Integration tests for the ML pipeline."""

import pytest
import pandas as pd
import numpy as np
from app.models.ml.prediction.account_health_predictor import AdvancedHealthPredictor

@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic features
    data = {
        'ctr': np.random.uniform(0.01, 0.1, n_samples),
        'conversion_rate': np.random.uniform(0.001, 0.05, n_samples),
        'cost_per_conversion': np.random.uniform(10, 200, n_samples),
        'impressions': np.random.randint(1000, 10000, n_samples),
        'clicks': np.random.randint(10, 1000, n_samples),
        'spend': np.random.uniform(100, 5000, n_samples),
        'revenue': np.random.uniform(0, 10000, n_samples),
        'conversions': np.random.randint(1, 100, n_samples)
    }
    
    # Add some correlations
    df = pd.DataFrame(data)
    df['health_score'] = (
        0.3 * df['ctr'] / df['ctr'].mean() +
        0.3 * df['conversion_rate'] / df['conversion_rate'].mean() +
        0.4 * (df['revenue'] - df['spend']) / df['spend'].clip(lower=1)
    ).clip(0, 1)
    
    return df

def test_full_health_pipeline(sample_data):
    """Test the complete health prediction pipeline."""
    # Initialize predictor
    predictor = AdvancedHealthPredictor()
    
    # Train model
    results = predictor.train(sample_data)
    
    # Verify training results
    assert results['metrics']['r2'] > 0.7, "Model R2 score should be above 0.7"
    assert 'feature_importance' in results, "Feature importance should be calculated"
    assert len(results['feature_importance']) > 0, "Feature importance should not be empty"
    
    # Test prediction
    sample_metrics = {
        'ctr': 0.05,
        'conversion_rate': 0.02,
        'cost_per_conversion': 50.0,
        'impressions': 5000,
        'clicks': 250,
        'spend': 1000.0,
        'revenue': 2000.0,
        'conversions': 40
    }
    
    prediction = predictor.predict_health_score(sample_metrics)
    
    # Verify prediction structure
    assert 0 <= prediction['health_score'] <= 1, "Health score should be between 0 and 1"
    assert 'confidence_interval' in prediction, "Should include confidence interval"
    assert 'risk_factors' in prediction, "Should include risk factors"
    assert 'optimization_suggestions' in prediction, "Should include optimization suggestions"
    
    # Test preprocessing validation
    with pytest.raises(ValueError):
        bad_metrics = sample_metrics.copy()
        del bad_metrics['ctr']
        predictor.predict_health_score(bad_metrics) 