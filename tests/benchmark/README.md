# ML Model Benchmark Testing Framework

This directory contains a comprehensive benchmarking framework for testing Machine Learning models against documented performance targets and fairness standards.

## Overview

The benchmark framework provides a systematic way to:

1. **Validate model performance** against documented benchmarks
2. **Ensure model fairness** across different demographic groups
3. **Track performance over time** by comparing with previous benchmark runs
4. **Generate comprehensive reports** for stakeholders

## Framework Components

The benchmarking framework consists of several components:

1. **Performance benchmarks** (`test_model_benchmarks.py`) - Tests model performance metrics against targets
2. **Fairness tests** (`test_fairness_properties.py`) - Property-based tests for model fairness 
3. **Benchmark runner** (`run_benchmarks.py`) - Script to run tests and generate reports
4. **Reporting utilities** - Tools to create visualizations and reports

## Performance Benchmarks

The performance benchmarks evaluate ML models against key metrics defined in our documentation, including:

- **RMSE** (Root Mean Square Error) - Target: < 8.2
- **R²** (Coefficient of determination) - Target: > 0.76
- **Spearman correlation** - Target: > 0.72
- **Precision@K** - Target: > 0.81
- **Recall@K** - Target: > 0.77

Models are tested using both synthetic data and real-world test sets to ensure comprehensive evaluation.

## Fairness Testing

The fairness tests use property-based testing to verify that models satisfy key fairness properties:

- **Demographic Parity** - Predictions should be independent of protected attributes
- **Equal Opportunity** - True positive rates should be similar across demographic groups
- **Treatment Consistency** - Similar individuals should receive similar predictions
- **Intersectional Fairness** - Models should be fair across intersections of multiple attributes

The framework uses hypothesis testing to generate diverse test cases and validate these properties.

## Running Benchmarks

The benchmark framework can be run using the `run_benchmarks.py` script:

```bash
# Run all benchmarks
python run_benchmarks.py --all --report

# Run only fairness tests
python run_benchmarks.py --fairness

# Run only performance benchmarks
python run_benchmarks.py --performance

# Compare with previous benchmark runs
python run_benchmarks.py --all --compare

# Generate a comprehensive report
python run_benchmarks.py --all --report
```

## Output and Reports

The benchmark framework generates structured outputs:

1. **JSON files** with detailed benchmark results
2. **Markdown reports** with summary tables and comparisons
3. **Log files** with detailed information about test runs

Results are organized by timestamp, allowing for easy tracking of performance over time.

## Directory Structure

```
tests/
├── benchmark/
│   ├── README.md                     # This file
│   ├── test_model_benchmarks.py      # Model performance benchmarks
│   └── __init__.py                   # Package initialization
├── property-based/
│   ├── test_fairness_properties.py   # Fairness property tests
│   ├── test_ad_predictor_properties.py # Model-specific property tests
│   └── __init__.py                   # Package initialization
└── __init__.py                       # Package initialization
```

## Target Models

The benchmark framework primarily tests:

1. **Ad Score Predictor** - Predicts ad performance scores
2. **Account Health Predictor** - Evaluates account health status
3. **Fairness-aware models** - Models with specific fairness guarantees

## Adding New Benchmarks

To add a new benchmark:

1. Update the benchmark targets in `tests/benchmark/test_model_benchmarks.py`
2. Add model-specific test cases
3. Update the documentation to reflect new benchmark targets

## Contributing

When contributing to the benchmark framework, please ensure:

1. All tests follow the project's coding standards
2. Benchmark targets align with documented specifications
3. New fairness properties are properly documented
4. Property-based tests use appropriate strategies for generating test data

## License

This benchmarking framework is part of the WITHIN internal tools and is subject to the same license and usage restrictions as the main project. 