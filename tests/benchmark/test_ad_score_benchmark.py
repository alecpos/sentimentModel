# tests/benchmark/test_ad_score_benchmark.py
import pytest
import pandas as pd
import numpy as np
from app.models.ml import get_ad_score_predictor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import json
import os
from datetime import datetime

class TestAdScoreBenchmark:
    """Tests to evaluate Ad Score Predictor against documented benchmarks."""
    
    @pytest.fixture
    def model(self):
        """Initialize the Ad Score Predictor model."""
        return get_ad_score_predictor()()
    
    @pytest.fixture
    def benchmark_dataset(self):
        """Load or generate a benchmark dataset that matches the documented benchmarks."""
        # In a real implementation, you would load an actual dataset
        # For demonstration, generating synthetic data
        n_samples = 500
        np.random.seed(42)
        
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
        y = 0.3 * features['sentiment_score'] + \
            0.3 * features['readability_score'] - \
            0.2 * features['complexity_score'] + \
            0.1 * features['click_through_rate'] + \
            0.1 * features['conversion_rate']
        
        # Add noise
        y += np.random.normal(0, 0.05, n_samples)
        
        # Convert to bounded range
        y = (y - y.min()) / (y.max() - y.min())
        
        return pd.DataFrame(features), y
    
    @pytest.fixture
    def benchmark_targets(self):
        """Target metrics from the benchmark document."""
        return {
            "rmse": 8.2,  # Scaled to 0-100 range
            "r2": 0.76,
            "spearman_rho": 0.72,
            "precision_at_10": 0.81,
            "recall_at_10": 0.77
        }
    
    def test_performance_metrics(self, model, benchmark_dataset, benchmark_targets):
        """Test if the model meets the documented performance metrics."""
        X, y = benchmark_dataset
        
        # Train the model
        model.fit(X, y)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred)) * 100  # Scale to 0-100
        r2 = r2_score(y, y_pred)
        spearman_corr, _ = spearmanr(y, y_pred)
        
        # Calculate precision and recall at top 10%
        k = int(len(y) * 0.1)
        top_pred_indices = np.argsort(y_pred)[-k:]
        top_true_indices = np.argsort(y)[-k:]
        
        # Convert to binary labels for top k
        y_binary = np.zeros_like(y)
        y_binary[top_true_indices] = 1
        
        y_pred_binary = np.zeros_like(y_pred)
        y_pred_binary[top_pred_indices] = 1
        
        precision = precision_score(y_binary, y_pred_binary)
        recall = recall_score(y_binary, y_pred_binary)
        
        # Compare with benchmark targets
        metrics = {
            "rmse": rmse,
            "r2": r2,
            "spearman_rho": spearman_corr,
            "precision_at_10": precision,
            "recall_at_10": recall
        }
        
        # Save results to file for analysis
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "targets": benchmark_targets,
            "differences": {k: metrics[k] - benchmark_targets[k] for k in metrics}
        }
        
        # Create directory if it doesn't exist
        os.makedirs("benchmark_results", exist_ok=True)
        
        # Save results
        with open(f"benchmark_results/ad_score_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print results for easy comparison
        print("\nAd Score Benchmark Results:")
        print("--------------------------")
        print(f"Metric          | Actual    | Target    | Difference")
        print(f"--------------  | --------- | --------- | ----------")
        for metric, value in metrics.items():
            target = benchmark_targets[metric]
            diff = value - target
            print(f"{metric:15} | {value:.6f} | {target:.6f} | {diff:+.6f}")
        
        # Optional: Assert if metrics meet targets
        # Comment these out if you just want to measure without failing tests
        assert rmse <= benchmark_targets["rmse"] * 1.1, f"RMSE too high: {rmse:.2f} vs target {benchmark_targets['rmse']:.2f}"
        assert r2 >= benchmark_targets["r2"] * 0.9, f"R² too low: {r2:.2f} vs target {benchmark_targets['r2']:.2f}"
        assert spearman_corr >= benchmark_targets["spearman_rho"] * 0.9, f"Spearman correlation too low: {spearman_corr:.2f} vs target {benchmark_targets['spearman_rho']:.2f}"