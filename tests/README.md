# WITHIN ML Testing Framework

This document provides an overview of the testing framework for the WITHIN ML platform, covering testing standards, organization, and how to run tests.

## Testing Approach

The WITHIN ML platform follows a comprehensive testing approach that combines:

1. **Unit Testing**: Testing individual components in isolation
2. **Integration Testing**: Testing interactions between components
3. **System Testing**: End-to-end validation of ML pipelines
4. **Performance Testing**: Evaluate model and API performance
5. **Fairness Testing**: Ensure models are fair across demographics
6. **Robustness Testing**: Verify model stability against adversarial inputs
7. **Drift Detection Testing**: Ensure models can detect data/concept drift
8. **Production Validation Testing**: Validate deployment strategies and monitoring
9. **Property-Based Testing**: Testing invariants across wide input ranges
10. **Neuro-Symbolic Testing**: Verify model adherence to logical constraints

## Directory Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Shared test fixtures 
├── test_adversarial.py             # Adversarial robustness tests
├── test_anomaly_detector.py        # Anomaly detection tests
├── test_fairness.py                # Fairness testing across demographics
├── test_ad_predictor.py            # Ad predictor core tests
├── test_enhanced_ad_predictor.py   # Enhanced predictor tests
├── error_test.py                   # Error handling tests
├── test_drift_detection.py         # Drift detection tests
├── test_production_validation.py   # Production validation tests
├── test_data_catalog.py            # Data catalog tests
├── test_search_service.py          # Search service tests
├── test_lineage_service.py         # Data lineage tests
├── test_data_models.py             # Data model tests
├── test_data_lake.py               # Data lake tests
├── test_security_search.py         # Security tests
├── test_feedback_handler.py        # Feedback handling tests
├── test_validation_feedback.py     # Validation feedback tests
├── test_data_pipeline_service.py   # Data pipeline tests
├── integration_test.py             # Integration tests
├── property-based/                 # Property-based tests
│   ├── __init__.py
│   └── test_ad_predictor_properties.py
├── benchmark/                      # Performance benchmarks
├── visual-regression/              # UI regression tests
└── .benchmarks/                    # Benchmark result data
```

## Testing Framework

The project uses the following testing tools:

- **pytest**: Primary testing framework
- **pytest-cov**: Test coverage reporting
- **pytest-xdist**: Parallel test execution
- **hypothesis**: Property-based testing
- **locust**: Load and performance testing
- **pytest-benchmark**: Performance benchmarking
- **mock**: Mocking library for unit tests
- **docker-compose**: For integration testing with dependencies

## Modern Testing Features (2025 Modernization)

The test suite has been modernized with these enhanced capabilities:

### 1. Fairness Testing

- **Intersectional Fairness**: Testing fairness across intersecting demographic attributes
- **Counterfactual Fairness**: Testing fairness using causal modeling approaches
- **Subgroup Robustness**: Testing performance across population subgroups

### 2. Robustness Testing

- **Certified Robustness**: Using randomized smoothing for provable robustness guarantees
- **Gradient Masking Detection**: Identifying problematic defensive techniques
- **Adversarial Testing**: Testing model behavior against adversarial inputs

### 3. Drift Detection

- **Data Drift**: Testing ability to detect distribution shifts in input data
- **Concept Drift**: Testing ability to detect changes in patterns/relationships
- **Gradual vs. Sudden Drift**: Testing detection of different drift patterns

### 4. Production Validation

- **Shadow Deployment**: Testing models in production without affecting outcomes
- **A/B Testing**: Rigorous comparative testing methods 
- **Monitoring-Based Validation**: Testing automated feedback loops
- **Canary Testing**: Incremental deployment testing

### 5. Neuro-Symbolic Testing

- **Logical Constraints**: Testing adherence to domain-specific rules
- **Feature Interaction Logic**: Testing feature interaction consistency
- **Invariance Properties**: Testing transformation invariance

### 6. Property-Based Testing

- **Monotonicity**: Testing that positive features increase scores appropriately
- **Prediction Bounds**: Testing valid output ranges across random inputs
- **Input Perturbation Stability**: Testing robustness to small changes
- **Batch Consistency**: Testing batch vs. individual predictions match
- **Determinism**: Testing for predictable, repeatable outputs

### 7. Model Calibration Testing

- **Uncertainty Estimation**: Testing appropriate confidence/uncertainty estimates
- **Calibration Quality**: Testing reliability of probability estimates
- **Environment-Specific Calibration**: Testing adaptation to different contexts

## Test Types with Examples

### Unit Tests

Unit tests verify the functionality of individual components in isolation:

```python
# Example unit test for AdScoreModel
def test_ad_score_model_prediction():
    """Test that model produces expected predictions for known input."""
    # Arrange
    model = AdScoreModel(model_type="gradient_boosting", random_state=42)
    X = pd.DataFrame({
        "feature1": [0.5, 0.7, 0.3],
        "feature2": [0.1, 0.2, 0.3],
        "feature3": [0.8, 0.6, 0.4]
    })
    model.feature_columns = list(X.columns)
    model.model = MockModel(return_value=np.array([85.0, 72.0, 63.0]))
    
    # Act
    predictions = model.predict(X)
    
    # Assert
    assert len(predictions) == 3
    assert isinstance(predictions, np.ndarray)
    assert predictions[0] == 85.0
    assert predictions[1] == 72.0
    assert predictions[2] == 63.0
```

### Integration Tests

Integration tests verify that components work together correctly:

```python
# Example integration test for the ad scoring pipeline
def test_ad_scoring_pipeline_integration(db_session, test_ad_data):
    """Test the full ad scoring pipeline from data input to prediction storage."""
    # Arrange
    pipeline = AdScoringPipeline(
        preprocessor=RealPreprocessor(),
        model=RealAdScoreModel(),
        storage=RealScoreStorage(session=db_session)
    )
    
    # Act
    results = pipeline.run(test_ad_data)
    
    # Assert
    assert results["success"] is True
    assert len(results["predictions"]) == len(test_ad_data)
    assert all(0 <= score <= 100 for score in results["predictions"])
    
    # Verify storage
    stored_scores = db_session.query(AdScore).all()
    assert len(stored_scores) == len(test_ad_data)
```

### Performance Tests

Performance tests ensure the system meets performance requirements:

```python
# Example performance test for the ad score prediction API
@pytest.mark.benchmark
def test_ad_score_api_performance(benchmark, api_client, test_ad_payload):
    """Benchmark the ad score prediction API endpoint."""
    # Execute the API call and measure performance
    result = benchmark(
        lambda: api_client.post("/api/v1/ad-score/predict", json=test_ad_payload)
    )
    
    # Assert on the benchmark
    assert result.stats.mean < 0.3  # Mean response time under 300ms
    assert result.stats.rounds >= 100  # Ensure enough samples
    assert result.stats.stddev < 0.05  # Stable performance
```

### Model Validation Tests

Model validation tests verify ML model quality:

```python
# Example model validation test for fairness
def test_ad_score_model_fairness(fairness_test_data):
    """Test model fairness across demographic groups."""
    # Arrange
    model = AdScoreModel()
    model.fit(fairness_test_data["X_train"], fairness_test_data["y_train"])
    predictions = model.predict(fairness_test_data["X_test"])
    
    # Act
    assessor = FairnessAssessor(protected_attributes=["gender", "age_group"])
    assessment = assessor.assess_fairness(
        predictions=predictions,
        actual_values=fairness_test_data["y_test"],
        metadata=fairness_test_data["metadata"]
    )
    
    # Assert
    assert assessment["overall_fairness"] is True, "Model shows unfair bias"
    
    # Check each protected attribute
    for attr, data in assessment["protected_attributes"].items():
        assert data["status"] == "fair", f"Unfairness detected for {attr}"
        assert len(data["violations"]) == 0, f"Fairness violations for {attr}"
```

### Property-Based Testing Example

The project uses property-based testing for robust validation:

```python
from hypothesis import given, strategies as st

@given(
    impressions=st.integers(min_value=0, max_value=1000000),
    clicks=st.integers(min_value=0, max_value=100000),
    ctr=st.floats(min_value=0, max_value=1)
)
def test_ctr_calculation_properties(impressions, clicks, ctr):
    """Test properties of CTR calculation with various inputs."""
    # If impressions is 0, CTR should be 0
    if impressions == 0:
        assert calculate_ctr(impressions, clicks) == 0
        
    # CTR should never be greater than 1
    assert calculate_ctr(impressions, clicks) <= 1
    
    # CTR should never be negative
    assert calculate_ctr(impressions, clicks) >= 0
    
    # If clicks > impressions, function should handle it gracefully
    if clicks > impressions and impressions > 0:
        assert calculate_ctr(impressions, clicks) == 1
```

### Neuro-Symbolic Testing Example

Testing adherence to logical constraints and rules:

```python
def test_logical_constraints():
    """Test that model predictions adhere to logical domain constraints."""
    # Arrange
    model = get_ad_score_predictor()()
    
    # Create a sample with high readability and positive sentiment
    positive_sample = pd.DataFrame([{
        'word_count': 300,
        'sentiment_score': 0.9,  # Very positive
        'complexity_score': 0.2,  # Low complexity
        'readability_score': 0.9  # High readability
    }])
    
    # Create a sample with low readability and negative sentiment
    negative_sample = pd.DataFrame([{
        'word_count': 300,
        'sentiment_score': 0.1,  # Very negative
        'complexity_score': 0.9,  # High complexity
        'readability_score': 0.1  # Low readability
    }])
    
    # Act
    positive_prediction = model.predict(positive_sample)[0]
    negative_prediction = model.predict(negative_sample)[0]
    
    # Assert
    # Positive samples should have higher scores than negative ones
    assert positive_prediction > negative_prediction
    
    # Scores should be in logical ranges given inputs
    assert positive_prediction > 0.6  # Should be high given very positive inputs
    assert negative_prediction < 0.4  # Should be low given very negative inputs
```

## Test Fixtures

Reusable test fixtures provide test data and dependencies:

```python
# Example fixtures in conftest.py
@pytest.fixture
def test_ad_data():
    """Provide a standard set of test ad data."""
    return pd.DataFrame({
        "ad_id": ["ad1", "ad2", "ad3"],
        "headline": [
            "Limited Time Offer: 20% Off All Products",
            "New Collection Now Available",
            "Free Shipping on Orders Over $50"
        ],
        "description": [
            "Shop our entire collection and save with this exclusive discount.",
            "Check out our latest styles for the new season.",
            "Limited time offer for all customers."
        ],
        "platform": ["facebook", "google", "facebook"],
        "impressions": [1000, 1500, 800],
        "clicks": [50, 60, 30],
        "conversions": [5, 8, 3]
    })

@pytest.fixture
def trained_ad_score_model(test_ad_data):
    """Provide a trained ad score model."""
    # Create features and target
    X = test_ad_data[["impressions", "clicks", "conversions"]]
    y = np.array([75, 82, 68])  # Mock scores
    
    # Train model
    model = AdScoreModel()
    model.fit(X, y)
    
    return model
```

The test suite uses a variety of modern fixtures defined in `conftest.py` to support testing:

- **Model fixtures** that provide pre-trained models
- **Data fixtures** that provide synthetic training and test data
- **Demographic data fixtures** for fairness testing
- **Drift test data fixtures** for drift detection testing
- **Mock service fixtures** for external dependencies
- **Production monitoring fixtures** for validation testing

## Test Categories

### API Tests

Tests for API endpoints ensuring:
- Correct response formats
- Proper status codes
- Input validation
- Authentication and authorization
- Error handling
- Rate limiting

### ETL Tests

Tests for data pipelines ensuring:
- Correct data extraction
- Proper transformations
- Accurate data loading
- Error handling and recovery
- Data validation
- Cross-platform compatibility

### Model Tests

Tests for ML models ensuring:
- Correct prediction functionality
- Performance metrics meet requirements
- Reproducibility with fixed seeds
- Proper feature importance calculation
- Correct serialization/deserialization
- Appropriate error handling

### Fairness Tests

Tests for model fairness ensuring:
- Equal performance across demographic groups
- No disparate impact
- Demographic parity where appropriate
- Equal opportunity
- Protection against bias amplification
- Intersectional fairness
- Counterfactual fairness

## Test Data

### Synthetic Test Data

The test suite uses synthetic data generators:

```python
from tests.fixtures.data_generators import generate_ad_data

# Generate synthetic test data
test_data = generate_ad_data(
    n_samples=1000,
    platforms=["facebook", "google", "tiktok"],
    date_range=("2023-01-01", "2023-03-31"),
    include_text=True,
    include_performance=True
)
```

Tests should use:

1. **Synthetic data from fixtures** for reproducible results
2. **Seeded random generation** for consistent test behavior
3. **Parameterized tests** to cover multiple scenarios
4. **Appropriate tolerances** for floating-point comparisons

## Running Tests

### Basic Test Execution

To run all tests:

```bash
pytest tests/
```

### Running Specific Test Categories

Run fairness tests:
```bash
pytest tests/test_fairness.py
```

Run property-based tests:
```bash
pytest tests/property-based/
```

Run robustness tests:
```bash
pytest tests/test_adversarial.py
```

### Running With Performance Profiling

```bash
pytest tests/ --benchmark-enable
```

### Selecting Test Categories

To run only drift detection tests:
```bash
pytest tests/ -m "drift"
```

To run tests with specific markers:
```bash
pytest tests/ -m "fairness and not slow"
```

## Code Coverage

The project requires minimum 90% test coverage:

```bash
# Run tests with coverage
pytest --cov=app tests/

# Generate coverage report
pytest --cov=app --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

Coverage requirements by module:

| Module | Min Coverage |
|--------|--------------|
| `app.api` | 95% |
| `app.models.ml.prediction` | 90% |
| `app.etl` | 90% |
| `app.nlp` | 85% |
| `app.monitoring` | 85% |

## CI/CD Integration

Tests are automatically run as part of the CI/CD pipeline:

1. **Unit tests**: Run on every commit
2. **Integration tests**: Run on pull requests
3. **Performance tests**: Run on release branches and nightly
4. **Model validation**: Run before model deployment
5. **Extended test suites**: Run before releases

## Test Monitoring

The test suite includes test result monitoring:

1. Test execution metrics are exported to the monitoring system
2. Failed test alerts are sent to the development team
3. Test coverage trends are tracked over time
4. Test performance is monitored for degradation

## Adding New Tests

When adding new tests:

1. **Follow the existing categorization** to place tests in appropriate files
2. **Use fixtures from `conftest.py`** where possible
3. **Add appropriate markers** to allow selective test execution
4. **Include both positive and negative test cases**
5. **For ML models, test edge cases and boundary conditions**
6. **Document test assumptions and test data characteristics**
7. **Include performance considerations** for slow tests
8. **Add property-based tests for invariants** where appropriate

## Development Guidelines

When adding or modifying tests:

1. **Use Fixtures**: Use fixtures for reusable test components
2. **Implement Mocks**: Mock external dependencies for unit tests
3. **Include Edge Cases**: Test with edge cases and error scenarios
4. **Use Representative Data**: Test with representative data distributions
5. **Test Reproducibility**: Test model reproducibility with fixed seeds
6. **Verify Metrics**: Compare model metrics against baselines
7. **Test Feature Importance**: Test feature importance stability
8. **Include Bias Tests**: Test for demographic parity and equal opportunity
9. **Performance Budgets**: Test against performance requirements
10. **Document Test Purposes**: Explain the purpose of each test

## Compliance Testing

The test suite includes specific tests for:

1. Regulatory compliance (demographic fairness, privacy)
2. Ethical AI guidelines
3. Model governance requirements
4. Data quality standards 