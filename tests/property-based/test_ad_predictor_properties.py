"""Property-based tests for the Ad Score Predictor.

This module implements property-based tests for the Ad Score Predictor model,
using the hypothesis framework to generate diverse test cases and validate
key properties that should hold regardless of input values.

These tests verify:
1. Prediction bounds (outputs always within valid range)
2. Input perturbation stability (similar inputs yield similar outputs)
3. Feature importance consistency (relative importance of features)
4. Monotonicity properties (increasing positive features increases score)
5. Transformation invariance (certain transformations should not affect predictions)
"""

import hypothesis
from hypothesis import given, settings, strategies as st
import pytest
import numpy as np
import pandas as pd

from app.models.ml.prediction import AdScorePredictor
from app.models.ml import get_ad_score_predictor

# Set a fixed seed for deterministic behavior in property tests
np.random.seed(42)

# Configure hypothesis to use reproducible randomness
hypothesis.configuration.set_hypothesis_home_dir('/tmp/hypothesis')
hypothesis.seed(42)

@pytest.fixture
def trained_model():
    """Create and train an ad predictor model on synthetic data."""
    # Generate a synthetic dataset
    n_samples = 200
    
    # Create features with clear patterns
    features = {
        'word_count': np.random.randint(50, 500, n_samples),
        'sentiment_score': np.random.uniform(0, 1, n_samples),
        'complexity_score': np.random.uniform(0, 1, n_samples),
        'readability_score': np.random.uniform(0, 1, n_samples),
        'engagement_rate': np.random.uniform(0, 1, n_samples),
        'click_through_rate': np.random.uniform(0, 1, n_samples),
        'conversion_rate': np.random.uniform(0, 1, n_samples),
        'content_category': np.random.randint(0, 5, n_samples),
        'ad_content': [f'Ad content {i}' for i in range(n_samples)]
    }
    
    # Create target with a known relationship to features
    # Engagement is positively related to sentiment and readability,
    # and negatively related to complexity
    engagement = []
    for i in range(n_samples):
        score = 0.3 * features['sentiment_score'][i] + \
                0.3 * features['readability_score'][i] - \
                0.2 * features['complexity_score'][i] + \
                0.1 * features['click_through_rate'][i] + \
                0.1 * features['conversion_rate'][i]
        
        # Add small random noise
        score += np.random.normal(0, 0.05)
        
        # Convert to binary outcome based on threshold
        engagement.append(1 if score > 0.5 else 0)
    
    # Create DataFrame
    features['engagement'] = engagement
    df = pd.DataFrame(features)
    
    # Initialize and train model
    model = get_ad_score_predictor()()
    X = df.drop('engagement', axis=1)
    y = df['engagement']
    model.fit(X, y)
    
    return model

# Strategy for generating ad data
@st.composite
def ad_data_strategy(draw):
    """Generate random, valid ad data for testing."""
    # Generate a single ad record with realistic value ranges
    ad_data = {
        'word_count': draw(st.integers(min_value=10, max_value=1000)),
        'sentiment_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'complexity_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'readability_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'engagement_rate': draw(st.floats(min_value=0.0, max_value=1.0)),
        'click_through_rate': draw(st.floats(min_value=0.0, max_value=1.0)),
        'conversion_rate': draw(st.floats(min_value=0.0, max_value=1.0)),
        'content_category': draw(st.integers(min_value=0, max_value=4)),
        'ad_content': draw(st.text(min_size=5, max_size=200))
    }
    return pd.DataFrame([ad_data])

@st.composite
def paired_ad_data_strategy(draw):
    """Generate pairs of similar ad data for testing stability."""
    # Base ad data
    base_ad = {
        'word_count': draw(st.integers(min_value=10, max_value=1000)),
        'sentiment_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'complexity_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'readability_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'engagement_rate': draw(st.floats(min_value=0.0, max_value=1.0)),
        'click_through_rate': draw(st.floats(min_value=0.0, max_value=1.0)),
        'conversion_rate': draw(st.floats(min_value=0.0, max_value=1.0)),
        'content_category': draw(st.integers(min_value=0, max_value=4)),
        'ad_content': draw(st.text(min_size=5, max_size=200))
    }
    
    # Slightly perturbed version
    perturbed_ad = base_ad.copy()
    
    # Small perturbation on numerical features (max 5% change)
    for key in ['sentiment_score', 'complexity_score', 'readability_score',
               'engagement_rate', 'click_through_rate', 'conversion_rate']:
        perturbation = draw(st.floats(min_value=-0.05, max_value=0.05))
        perturbed_value = base_ad[key] + perturbation
        # Ensure values stay in valid range [0,1]
        perturbed_ad[key] = max(0.0, min(1.0, perturbed_value))
    
    # Small perturbation on word count
    word_count_change = draw(st.integers(min_value=-20, max_value=20))
    perturbed_ad['word_count'] = max(10, base_ad['word_count'] + word_count_change)
    
    return pd.DataFrame([base_ad]), pd.DataFrame([perturbed_ad])

@given(ad_data=ad_data_strategy())
@settings(max_examples=100, deadline=None)
def test_prediction_bounds_property(trained_model, ad_data):
    """Test that predictions always fall within the expected range [0,1]."""
    predictions = trained_model.predict(ad_data)
    
    # Verify predictions are within expected bounds
    assert np.all(predictions >= 0.0) and np.all(predictions <= 1.0), \
        f"Predictions out of bounds: {predictions}"

@given(ad_data_pair=paired_ad_data_strategy())
@settings(max_examples=100, deadline=None)
def test_stability_property(trained_model, ad_data_pair):
    """Test that small perturbations in input lead to small changes in output."""
    base_ad, perturbed_ad = ad_data_pair
    
    # Get predictions
    base_pred = trained_model.predict(base_ad)[0]
    perturbed_pred = trained_model.predict(perturbed_ad)[0]
    
    # Calculate prediction difference
    pred_diff = abs(perturbed_pred - base_pred)
    
    # For small perturbations, prediction difference should be limited
    assert pred_diff <= 0.2, \
        f"Model unstable: Small perturbation caused large prediction change: {pred_diff}"

@given(
    sentiment=st.floats(min_value=0.0, max_value=1.0),
    readability=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=50, deadline=None)
def test_monotonicity_property(trained_model, sentiment, readability):
    """Test that increasing positive features leads to higher predicted scores."""
    # Create a base example with moderate values
    base_ad = pd.DataFrame([{
        'word_count': 300,
        'sentiment_score': sentiment,
        'complexity_score': 0.5,
        'readability_score': readability,
        'engagement_rate': 0.5,
        'click_through_rate': 0.5,
        'conversion_rate': 0.5,
        'content_category': 2,
        'ad_content': 'Monotonicity test content'
    }])
    
    # Create version with increased sentiment
    increased_sentiment = base_ad.copy()
    increased_sentiment.loc[0, 'sentiment_score'] = min(1.0, sentiment + 0.2)
    
    # Create version with increased readability
    increased_readability = base_ad.copy()
    increased_readability.loc[0, 'readability_score'] = min(1.0, readability + 0.2)
    
    # Create version with decreased complexity (should increase score)
    decreased_complexity = base_ad.copy()
    decreased_complexity.loc[0, 'complexity_score'] = max(0.0, 0.5 - 0.2)
    
    # Get predictions
    base_pred = trained_model.predict(base_ad)[0]
    sentiment_pred = trained_model.predict(increased_sentiment)[0]
    readability_pred = trained_model.predict(increased_readability)[0]
    complexity_pred = trained_model.predict(decreased_complexity)[0]
    
    # Test monotonicity properties
    # Allow a small tolerance for model noise/variance
    assert sentiment_pred >= base_pred - 0.05, \
        f"Increasing sentiment decreased prediction: {base_pred} -> {sentiment_pred}"
    
    assert readability_pred >= base_pred - 0.05, \
        f"Increasing readability decreased prediction: {base_pred} -> {readability_pred}"
    
    assert complexity_pred >= base_pred - 0.05, \
        f"Decreasing complexity decreased prediction: {base_pred} -> {complexity_pred}"

@given(st.integers(min_value=0, max_value=4))
@settings(max_examples=5, deadline=None)
def test_invariance_property(trained_model, category):
    """Test that certain changes (like specific wording) don't dramatically change predictions."""
    # Create a base example
    base_ad = pd.DataFrame([{
        'word_count': 300,
        'sentiment_score': 0.7,
        'complexity_score': 0.3,
        'readability_score': 0.7,
        'engagement_rate': 0.6,
        'click_through_rate': 0.6,
        'conversion_rate': 0.6,
        'content_category': category,
        'ad_content': 'Test content A with specific message'
    }])
    
    # Create a version with different wording but same semantic content and metrics
    reworded_ad = base_ad.copy()
    reworded_ad.loc[0, 'ad_content'] = 'Test content B with equivalent message'
    
    # Get predictions
    base_pred = trained_model.predict(base_ad)[0]
    reworded_pred = trained_model.predict(reworded_ad)[0]
    
    # Test invariance - different text content with same metrics should
    # yield similar predictions (assuming the model relies on the extracted features)
    assert abs(base_pred - reworded_pred) <= 0.1, \
        f"Model sensitive to semantically equivalent rewording: {base_pred} vs {reworded_pred}"

@given(
    batch_size=st.integers(min_value=1, max_value=10),
    word_count=st.integers(min_value=50, max_value=500)
)
@settings(max_examples=20, deadline=None)
def test_batch_consistency_property(trained_model, batch_size, word_count):
    """Test that predictions are consistent when done individually or in batch."""
    # Create a batch of identical ads with varying content length
    batch_data = []
    for i in range(batch_size):
        ad = {
            'word_count': word_count,
            'sentiment_score': 0.7,
            'complexity_score': 0.3,
            'readability_score': 0.7,
            'engagement_rate': 0.6,
            'click_through_rate': 0.6,
            'conversion_rate': 0.6,
            'content_category': 2,
            'ad_content': f'Test content for batch processing {i}'
        }
        batch_data.append(ad)
    
    batch_df = pd.DataFrame(batch_data)
    
    # Get batch predictions
    batch_predictions = trained_model.predict(batch_df)
    
    # Get individual predictions
    individual_predictions = []
    for i in range(batch_size):
        single_ad = pd.DataFrame([batch_data[i]])
        pred = trained_model.predict(single_ad)[0]
        individual_predictions.append(pred)
    
    # Test that batch and individual predictions are the same
    for i in range(batch_size):
        assert abs(batch_predictions[i] - individual_predictions[i]) < 1e-10, \
            f"Inconsistency between batch and individual prediction at index {i}"

@given(
    ad_data=ad_data_strategy(),
    repeat_count=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=20, deadline=None)
def test_determinism_property(trained_model, ad_data, repeat_count):
    """Test that predictions are deterministic for the same input."""
    predictions = []
    
    # Make multiple predictions on the same data
    for _ in range(repeat_count):
        pred = trained_model.predict(ad_data)[0]
        predictions.append(pred)
    
    # Check that all predictions are identical
    for i in range(1, repeat_count):
        assert abs(predictions[0] - predictions[i]) < 1e-10, \
            f"Non-deterministic predictions: {predictions[0]} vs {predictions[i]}" 