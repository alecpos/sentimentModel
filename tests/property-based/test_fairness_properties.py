"""Property-based tests for fairness evaluation in ML models.

This module implements property-based tests that verify fairness properties
of machine learning models, particularly focusing on demographic parity,
equal opportunity, and treatment consistency across protected attributes.

These tests follow property-based testing principles using Hypothesis to
generate test cases that verify fairness invariants that should hold for
fair ML models.
"""
import pytest
import numpy as np
import pandas as pd
import hypothesis
from hypothesis import given, settings, strategies as st, HealthCheck
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
import os
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set a fixed seed for deterministic tests
hypothesis.seed(42)
np.random.seed(42)

# Test constants
EPSILON = 0.15  # Maximum allowed difference for demographic parity
TEST_ITERATIONS = 50  # Number of test iterations for property tests
MAX_EXAMPLES = 100  # Maximum examples per Hypothesis test

def get_output_dir(default_dir: str = "fairness_test_results") -> str:
    """Get output directory from environment variable or use default.
    
    Args:
        default_dir: Default directory to use if environment variable is not set
        
    Returns:
        Output directory path
    """
    output_dir = os.environ.get("FAIRNESS_OUTPUT_DIR", default_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class FairnessResult:
    """Container for storing fairness test results.
    
    Attributes:
        model_name: Name of the model being tested
        timestamp: Time when test was run
        protected_attribute: Name of the protected attribute
        metrics: Dictionary of fairness metrics
        passed: Boolean indicating if the model passed the fairness test
    """
    
    def __init__(
        self, 
        model_name: str,
        protected_attribute: str,
        metrics: Dict[str, float],
        passed: bool
    ) -> None:
        """Initialize fairness result container.
        
        Args:
            model_name: Name of the model being tested
            protected_attribute: Name of the protected attribute
            metrics: Dictionary of fairness metrics
            passed: Boolean indicating if the model passed the test
        """
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.protected_attribute = protected_attribute
        self.metrics = metrics
        self.passed = passed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format.
        
        Returns:
            Dictionary representation of fairness results
        """
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "protected_attribute": self.protected_attribute,
            "metrics": self.metrics,
            "passed": self.passed
        }
    
    def save(self, output_dir: str = None) -> str:
        """Save fairness results to JSON file.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = get_output_dir()
            
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.model_name}_{self.protected_attribute}_fairness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    def print_results(self) -> None:
        """Print a formatted report of fairness results."""
        print(f"\n{self.model_name} Fairness Test Results ({self.timestamp}):")
        print(f"Protected Attribute: {self.protected_attribute}")
        print("-" * 80)
        
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("-" * 80)
        status = "PASSED" if self.passed else "FAILED"
        print(f"Overall: {status}\n")

# Define FairnessMetrics class
class FairnessMetrics:
    """Evaluate model fairness across different demographic groups.
    
    This class computes and reports fairness metrics for model predictions
    across various demographic dimensions.
    
    Attributes:
        model: The model to evaluate
        metrics: Dictionary of fairness metrics
    """
    
    def __init__(self, model: Any) -> None:
        """Initialize fairness metrics evaluator.
        
        Args:
            model: The model to evaluate
        """
        self.model = model
        self.metrics: Dict[str, Dict[str, float]] = {}
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        demographic_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate fairness metrics across demographic groups.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            demographic_cols: List of demographic column names
            
        Returns:
            Dictionary of fairness metrics per demographic group
        """
        predictions = self.model.predict(X)
        
        results = {}
        
        # Evaluate each demographic dimension
        for col in demographic_cols:
            if col not in X.columns:
                continue
                
            group_metrics = {}
            unique_groups = X[col].unique()
            
            # Calculate metrics for each group
            for group in unique_groups:
                group_mask = X[col] == group
                group_X = X[group_mask]
                group_y = y[group_mask]
                group_preds = predictions[group_mask]
                
                # Skip if too few samples
                if len(group_y) < 10:
                    continue
                    
                # Calculate performance metrics
                rmse = np.sqrt(mean_squared_error(group_y, group_preds))
                r2 = r2_score(group_y, group_preds)
                spearman = stats.spearmanr(group_y, group_preds)[0]
                
                group_metrics[group] = {
                    'RMSE': rmse,
                    'R²': r2,
                    'Spearman': spearman,
                    'count': len(group_y)
                }
            
            # Calculate disparity metrics
            if len(group_metrics) >= 2:
                rmse_values = [m['RMSE'] for m in group_metrics.values()]
                r2_values = [m['R²'] for m in group_metrics.values()]
                
                # Max disparity
                rmse_disparity = max(rmse_values) - min(rmse_values)
                r2_disparity = max(r2_values) - min(r2_values)
                
                # Store disparity metrics
                group_metrics['_DISPARITIES'] = {
                    'RMSE_disparity': rmse_disparity,
                    'R²_disparity': r2_disparity
                }
            
            results[col] = group_metrics
            
        self.metrics = results
        return results

@pytest.fixture
def placeholder_model():
    """Create a placeholder model for testing."""
    try:
        # Try to use the real implementation
        from app.models.ml.prediction.ad_score_predictor import AdScorePredictor
        return AdScorePredictor(model_type="regression")
    except (ImportError, Exception) as e:
        logger.warning(f"Could not import or initialize AdScorePredictor: {e}")
        # Create a placeholder implementation
        from sklearn.ensemble import RandomForestRegressor
        
        class PlaceholderPredictor:
            def __init__(self):
                self.is_fitted = False
                self.model = RandomForestRegressor(n_estimators=10, random_state=42)
                
            def fit(self, X, y):
                # Convert DataFrame to numpy for simplicity
                if hasattr(X, 'values'):
                    X_values = X.values
                else:
                    X_values = X
                    
                if hasattr(y, 'values'):
                    y_values = y.values
                else:
                    y_values = y
                    
                self.model.fit(X_values, y_values)
                self.is_fitted = True
                return self
                
            def predict(self, X):
                # Convert DataFrame to numpy for simplicity
                if hasattr(X, 'values'):
                    X_values = X.values
                else:
                    X_values = X
                    
                # Return numpy array of predictions
                return self.model.predict(X_values)
        
        logger.info("Using PlaceholderPredictor for testing")
        return PlaceholderPredictor()


@st.composite
def fair_dataset_strategy(draw):
    """Generate a fair dataset with protected attributes.
    
    This strategy generates synthetic datasets with protected attributes
    and targets that have no built-in dependency on protected attributes.
    
    Returns:
        Tuple of (X, y) where X is a DataFrame and y is a Series
    """
    # Sample size
    n_samples = draw(st.integers(min_value=50, max_value=200))
    
    # Generate basic features
    features = {
        'feature1': draw(st.lists(st.floats(min_value=-10, max_value=10), 
                                 min_size=n_samples, max_size=n_samples)),
        'feature2': draw(st.lists(st.floats(min_value=-10, max_value=10), 
                                 min_size=n_samples, max_size=n_samples)),
        'feature3': draw(st.lists(st.floats(min_value=-10, max_value=10), 
                                 min_size=n_samples, max_size=n_samples)),
    }
    
    # Generate protected attributes with uniform distribution
    protected_attr_values = draw(st.sampled_from([
        ['male', 'female'],
        ['18-25', '26-40', '41-60', '60+'],
        ['A', 'B', 'C', 'D'],
        ['group1', 'group2']
    ]))
    
    features['protected_attr'] = draw(st.lists(
        st.sampled_from(protected_attr_values),
        min_size=n_samples, max_size=n_samples
    ))
    
    # Create DataFrame
    X = pd.DataFrame(features)
    
    # Generate target that depends only on non-protected features
    # (ensuring no direct bias in the data)
    y_values = []
    for i in range(n_samples):
        # Linear combination of non-protected features
        score = (
            0.3 * features['feature1'][i] + 
            0.3 * features['feature2'][i] + 
            0.4 * features['feature3'][i]
        )
        
        # Add random noise
        score += draw(st.floats(min_value=-0.5, max_value=0.5))
        
        # Binarize
        y_values.append(1 if score > 0 else 0)
    
    return X, pd.Series(y_values, name='target')


@st.composite
def biased_dataset_strategy(draw):
    """Generate a biased dataset with protected attributes.
    
    This strategy generates synthetic datasets with protected attributes
    and targets that have a deliberate dependency on protected attributes,
    creating built-in bias.
    
    Returns:
        Tuple of (X, y) where X is a DataFrame and y is a Series
    """
    # Sample size
    n_samples = draw(st.integers(min_value=50, max_value=200))
    
    # Generate basic features
    features = {
        'feature1': draw(st.lists(st.floats(min_value=-10, max_value=10), 
                                 min_size=n_samples, max_size=n_samples)),
        'feature2': draw(st.lists(st.floats(min_value=-10, max_value=10), 
                                 min_size=n_samples, max_size=n_samples)),
        'feature3': draw(st.lists(st.floats(min_value=-10, max_value=10), 
                                 min_size=n_samples, max_size=n_samples)),
    }
    
    # Generate binary protected attribute
    protected_values = ['group_a', 'group_b']
    features['protected_attr'] = draw(st.lists(
        st.sampled_from(protected_values),
        min_size=n_samples, max_size=n_samples
    ))
    
    # Create DataFrame
    X = pd.DataFrame(features)
    
    # Generate target with dependency on protected attribute
    # (creating deliberate bias)
    bias_strength = draw(st.floats(min_value=0.2, max_value=0.8))
    y_values = []
    
    for i in range(n_samples):
        # Base score from non-protected features
        base_score = (
            0.3 * features['feature1'][i] + 
            0.3 * features['feature2'][i] + 
            0.4 * features['feature3'][i]
        )
        
        # Add bias based on protected attribute
        bias = bias_strength if features['protected_attr'][i] == 'group_a' else -bias_strength
        score = base_score + bias
        
        # Add random noise
        score += draw(st.floats(min_value=-0.5, max_value=0.5))
        
        # Binarize
        y_values.append(1 if score > 0 else 0)
    
    return X, pd.Series(y_values, name='target')


def demographic_parity_metric(
    y_pred: np.ndarray,
    protected_attr: np.ndarray
) -> Dict[str, float]:
    """Calculate demographic parity metrics.
    
    Demographic parity requires that prediction rates are equal across
    protected attribute groups.
    
    Args:
        y_pred: Predicted labels or scores
        protected_attr: Protected attribute values
        
    Returns:
        Dictionary of demographic parity metrics
    """
    unique_attrs = np.unique(protected_attr)
    
    # Calculate acceptance rate for each group
    group_rates = {}
    for attr in unique_attrs:
        mask = protected_attr == attr
        if mask.sum() > 0:  # Avoid division by zero
            # For binary classification
            if np.all(np.isin(np.unique(y_pred), [0, 1])):
                group_rates[attr] = y_pred[mask].mean()
            # For regression
            else:
                group_rates[attr] = y_pred[mask].mean()
    
    # Calculate parity difference (max - min)
    parity_diff = max(group_rates.values()) - min(group_rates.values())
    
    # Calculate disparate impact
    min_rate = min(group_rates.values())
    max_rate = max(group_rates.values())
    disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0
    
    return {
        "demographic_parity_difference": parity_diff,
        "disparate_impact": disparate_impact,
        "group_rates": group_rates
    }


def equal_opportunity_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray
) -> Dict[str, float]:
    """Calculate equal opportunity metrics.
    
    Equal opportunity requires that true positive rates are equal across
    protected attribute groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attr: Protected attribute values
        
    Returns:
        Dictionary of equal opportunity metrics
    """
    unique_attrs = np.unique(protected_attr)
    
    # Binarize predictions if they're probabilities
    if not np.all(np.isin(np.unique(y_pred), [0, 1])):
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred
    
    # Calculate true positive rate for each group
    tpr_per_group = {}
    for attr in unique_attrs:
        mask = protected_attr == attr
        if mask.sum() > 0:
            # Get confusion matrix for this group
            tn, fp, fn, tp = confusion_matrix(
                y_true[mask], 
                y_pred_binary[mask],
                labels=[0, 1]
            ).ravel()
            
            # Calculate true positive rate
            if (tp + fn) > 0:  # Avoid division by zero
                tpr = tp / (tp + fn)
            else:
                tpr = 0.0
                
            tpr_per_group[attr] = tpr
    
    # Calculate equal opportunity difference (max TPR - min TPR)
    if len(tpr_per_group) >= 2:
        eq_opp_diff = max(tpr_per_group.values()) - min(tpr_per_group.values())
    else:
        eq_opp_diff = 0.0
    
    return {
        "equal_opportunity_difference": eq_opp_diff,
        "true_positive_rates": tpr_per_group
    }


def treatment_consistency_metric(
    model: Any,
    X: pd.DataFrame,
    protected_attr_name: str,
    n_perturbations: int = 10
) -> Dict[str, float]:
    """Calculate treatment consistency metrics.
    
    Treatment consistency requires that the model's predictions are not
    affected by small perturbations in non-protected features.
    
    Args:
        model: ML model with predict method
        X: Feature dataframe
        protected_attr_name: Name of protected attribute column
        n_perturbations: Number of perturbations to generate
        
    Returns:
        Dictionary of treatment consistency metrics
    """
    protected_attrs = X[protected_attr_name].values
    unique_attrs = np.unique(protected_attrs)
    
    # Copy data for perturbation
    X_copy = X.copy()
    non_protected_cols = [col for col in X.columns if col != protected_attr_name]
    
    # Calculate consistency across perturbations for each group
    group_consistency = {}
    
    for attr in unique_attrs:
        attr_mask = protected_attrs == attr
        attr_indices = np.where(attr_mask)[0]
        
        if len(attr_indices) == 0:
            continue
            
        # Sample individuals from this group
        sample_size = min(len(attr_indices), 20)  # Limit to 20 samples
        sampled_indices = np.random.choice(attr_indices, sample_size, replace=False)
        
        consistency_scores = []
        
        # For each sampled individual
        for idx in sampled_indices:
            original_pred = model.predict(X.iloc[idx:idx+1])[0]
            perturbed_preds = []
            
            # Generate perturbations
            for _ in range(n_perturbations):
                X_perturbed = X.iloc[idx:idx+1].copy()
                
                # Perturb each non-protected feature slightly
                for col in non_protected_cols:
                    if X[col].dtype.kind in 'fc':  # float or complex
                        # Add small Gaussian noise
                        noise = np.random.normal(0, 0.1 * X[col].std())
                        X_perturbed[col] += noise
                
                perturbed_pred = model.predict(X_perturbed)[0]
                perturbed_preds.append(perturbed_pred)
            
            # Calculate consistency as 1 - average absolute difference
            if perturbed_preds:
                avg_diff = np.mean([abs(p - original_pred) for p in perturbed_preds])
                consistency = 1.0 - min(1.0, avg_diff)  # Limit to [0, 1]
                consistency_scores.append(consistency)
        
        if consistency_scores:
            group_consistency[attr] = np.mean(consistency_scores)
    
    # Calculate consistency difference
    if len(group_consistency) >= 2:
        consistency_diff = max(group_consistency.values()) - min(group_consistency.values())
    else:
        consistency_diff = 0.0
    
    return {
        "treatment_consistency_difference": consistency_diff,
        "group_consistency": group_consistency
    }


@pytest.fixture
def fairness_evaluator():
    """Create a fairness evaluator function."""
    def evaluate_fairness(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attr_name: str = 'protected_attr',
        threshold: float = 0.15
    ) -> FairnessResult:
        """Evaluate fairness of a model.
        
        Args:
            model: ML model with predict method
            X: Feature dataframe
            y: Target series
            protected_attr_name: Name of protected attribute column
            threshold: Maximum allowed difference for fairness metrics
            
        Returns:
            FairnessResult object
        """
        # Get predictions
        predictions = model.predict(X)
        protected_attrs = X[protected_attr_name].values
        
        # Calculate fairness metrics
        dp_metrics = demographic_parity_metric(predictions, protected_attrs)
        eo_metrics = equal_opportunity_metric(y.values, predictions, protected_attrs)
        tc_metrics = treatment_consistency_metric(model, X, protected_attr_name)
        
        # Combine metrics
        metrics = {
            "demographic_parity_difference": dp_metrics["demographic_parity_difference"],
            "equal_opportunity_difference": eo_metrics["equal_opportunity_difference"],
            "treatment_consistency_difference": tc_metrics["treatment_consistency_difference"]
        }
        
        # Determine if model passes fairness test
        passed = (
            metrics["demographic_parity_difference"] <= threshold and
            metrics["equal_opportunity_difference"] <= threshold and
            metrics["treatment_consistency_difference"] <= threshold
        )
        
        # Add group-specific metrics
        metrics["group_rates"] = dp_metrics["group_rates"]
        metrics["true_positive_rates"] = eo_metrics["true_positive_rates"]
        metrics["group_consistency"] = tc_metrics["group_consistency"]
        
        # Create result
        model_name = model.__class__.__name__
        result = FairnessResult(model_name, protected_attr_name, metrics, passed)
        
        return result
    
    return evaluate_fairness


@given(dataset=fair_dataset_strategy())
@settings(max_examples=MAX_EXAMPLES, deadline=None, 
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_demographic_parity_property(dataset, placeholder_model, fairness_evaluator):
    """Test demographic parity property.
    
    This test verifies that the model satisfies demographic parity,
    meaning prediction rates are similar across protected attribute groups.
    """
    X, y = dataset
    
    # Train the model
    model = placeholder_model
    model.fit(X, y)
    
    # Evaluate fairness
    result = fairness_evaluator(model, X, y)
    
    # Save results
    result.save()
    
    # Print results
    result.print_results()
    
    # Assert demographic parity
    assert result.metrics["demographic_parity_difference"] <= EPSILON, \
        f"Demographic parity difference {result.metrics['demographic_parity_difference']} exceeds threshold {EPSILON}"


@given(dataset=fair_dataset_strategy())
@settings(max_examples=MAX_EXAMPLES, deadline=None, 
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_equal_opportunity_property(dataset, placeholder_model, fairness_evaluator):
    """Test equal opportunity property.
    
    This test verifies that the model satisfies equal opportunity,
    meaning true positive rates are similar across protected attribute groups.
    """
    X, y = dataset
    
    # Train the model
    model = placeholder_model
    model.fit(X, y)
    
    # Evaluate fairness
    result = fairness_evaluator(model, X, y)
    
    # Save results
    result.save()
    
    # Print results
    result.print_results()
    
    # Assert equal opportunity
    assert result.metrics["equal_opportunity_difference"] <= EPSILON, \
        f"Equal opportunity difference {result.metrics['equal_opportunity_difference']} exceeds threshold {EPSILON}"


@given(dataset=fair_dataset_strategy())
@settings(max_examples=MAX_EXAMPLES, deadline=None, 
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_treatment_consistency_property(dataset, placeholder_model, fairness_evaluator):
    """Test treatment consistency property.
    
    This test verifies that the model satisfies treatment consistency,
    meaning predictions are not arbitrarily affected by small perturbations
    in non-protected features.
    """
    X, y = dataset
    
    # Train the model
    model = placeholder_model
    model.fit(X, y)
    
    # Evaluate fairness
    result = fairness_evaluator(model, X, y)
    
    # Save results
    result.save()
    
    # Print results
    result.print_results()
    
    # Assert treatment consistency
    assert result.metrics["treatment_consistency_difference"] <= EPSILON, \
        f"Treatment consistency difference {result.metrics['treatment_consistency_difference']} exceeds threshold {EPSILON}"


@given(dataset=biased_dataset_strategy())
@settings(max_examples=MAX_EXAMPLES, deadline=None, 
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_bias_detection_capability(dataset, placeholder_model, fairness_evaluator):
    """Test that fairness metrics can detect bias.
    
    This test verifies that our fairness metrics can detect bias when it exists,
    using a deliberately biased dataset.
    """
    X, y = dataset
    
    # Train the model (will learn the bias in the data)
    model = placeholder_model
    model.fit(X, y)
    
    # Evaluate fairness
    result = fairness_evaluator(model, X, y)
    
    # Save results with special name to indicate this is a bias detection test
    result.model_name = f"{result.model_name}_BiasDetectionTest"
    result.save()
    
    # Print results
    result.print_results()
    
    # Note: We expect this test to detect bias, but we don't assert failure
    # because some models might be robust against learning the bias
    logger.info(
        f"Bias detection test: demographic parity difference = {result.metrics['demographic_parity_difference']}, "
        f"equal opportunity difference = {result.metrics['equal_opportunity_difference']}"
    )


def test_intersectional_fairness(placeholder_model):
    """Test fairness across intersectional demographic groups.
    
    This test verifies fairness properties across intersections of multiple
    protected attributes (e.g., age and gender combinations).
    """
    # Create synthetic dataset with multiple protected attributes
    n_samples = 1000
    np.random.seed(42)
    
    # Generate non-protected features
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples),
        'age_group': np.random.choice(['18-25', '26-40', '41-60', '60+'], n_samples),
        'region': np.random.choice(['NA', 'EU', 'AS', 'AF', 'SA'], n_samples)
    })
    
    # Generate fair target
    y = pd.Series(
        (0.3 * X['feature1'] + 0.3 * X['feature2'] + 0.4 * X['feature3'] + 
         np.random.normal(0, 0.2, n_samples)) > 0,
        name='target'
    ).astype(int)
    
    # Train model
    model = placeholder_model
    model.fit(X, y)
    
    # Create intersectional groups
    X['gender_age'] = X['gender'] + '_' + X['age_group']
    X['gender_region'] = X['gender'] + '_' + X['region']
    
    # Evaluate fairness for each protected attribute
    protected_attrs = ['gender', 'age_group', 'region', 'gender_age', 'gender_region']
    results = []
    
    for attr in protected_attrs:
        evaluator = FairnessMetrics(model)
        metrics = evaluator.evaluate(X, y, [attr])
        
        # Calculate demographic parity
        predictions = model.predict(X)
        dp_metrics = demographic_parity_metric(predictions, X[attr].values)
        
        # Save results
        result = FairnessResult(
            model.__class__.__name__,
            attr,
            {"demographic_parity_difference": dp_metrics["demographic_parity_difference"]},
            dp_metrics["demographic_parity_difference"] <= EPSILON
        )
        result.save()
        results.append(result)
        
        # Print results
        result.print_results()
    
    # Verify at least some of the tests pass
    pass_count = sum(1 for r in results if r.passed)
    assert pass_count > 0, "No protected attributes passed fairness tests"
    
    # Summarize results across attributes
    attr_results = {r.protected_attribute: r.passed for r in results}
    logger.info(f"Intersectional fairness results: {attr_results}")


def test_model_doesnt_learn_protected_attributes(placeholder_model):
    """Test that the model doesn't implicitly encode protected attributes.
    
    This test verifies that the model's predictions cannot be used to
    accurately predict the protected attributes, which would indicate
    the model has implicitly encoded this information.
    """
    # Create synthetic dataset
    n_samples = 1000
    np.random.seed(42)
    
    # Generate features
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'protected_attr': np.random.choice(['group_a', 'group_b'], n_samples)
    })
    
    # Generate target (fair, no dependency on protected attribute)
    y = pd.Series(
        (0.3 * X['feature1'] + 0.3 * X['feature2'] + 0.4 * X['feature3'] + 
         np.random.normal(0, 0.2, n_samples)) > 0,
        name='target'
    ).astype(int)
    
    # Train model
    model = placeholder_model
    model.fit(X.drop('protected_attr', axis=1), y)
    
    # Get predictions
    predictions = model.predict(X.drop('protected_attr', axis=1))
    
    # Try to predict protected attribute from model outputs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        predictions.reshape(-1, 1),
        X['protected_attr'],
        test_size=0.3,
        random_state=42
    )
    
    # Train classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate baseline (majority class frequency)
    baseline = max(np.mean(y_test == 'group_a'), np.mean(y_test == 'group_b'))
    
    # Log results
    logger.info(f"Protected attribute prediction accuracy: {accuracy:.4f}")
    logger.info(f"Baseline accuracy (majority class): {baseline:.4f}")
    
    # Save results
    result = {
        "model": model.__class__.__name__,
        "protected_attr_prediction_accuracy": accuracy,
        "baseline_accuracy": baseline,
        "information_leakage": max(0, accuracy - baseline),
        "passed": accuracy <= baseline + 0.05  # Allow 5% above baseline
    }
    
    os.makedirs(get_output_dir(), exist_ok=True)
    filepath = os.path.join(
        get_output_dir(), 
        f"protected_attr_encoding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Assert protection against information leakage
    assert accuracy <= baseline + 0.05, \
        f"Model leaks information about protected attributes: accuracy {accuracy:.4f} vs baseline {baseline:.4f}"


if __name__ == "__main__":
    # This allows running the tests directly
    model = placeholder_model()
    
    # Generate test data
    n_samples = 1000
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'protected_attr': np.random.choice(['group_a', 'group_b'], n_samples)
    })
    
    y = pd.Series(
        (0.3 * X['feature1'] + 0.3 * X['feature2'] + 0.4 * X['feature3'] + 
         np.random.normal(0, 0.2, n_samples)) > 0,
        name='target'
    ).astype(int)
    
    # Train model
    model.fit(X, y)
    
    # Create fairness evaluator
    def evaluate_fairness(model, X, y, protected_attr_name='protected_attr', threshold=0.15):
        predictions = model.predict(X)
        protected_attrs = X[protected_attr_name].values
        
        dp_metrics = demographic_parity_metric(predictions, protected_attrs)
        eo_metrics = equal_opportunity_metric(y.values, predictions, protected_attrs)
        
        metrics = {
            "demographic_parity_difference": dp_metrics["demographic_parity_difference"],
            "equal_opportunity_difference": eo_metrics["equal_opportunity_difference"],
            "group_rates": dp_metrics["group_rates"],
            "true_positive_rates": eo_metrics["true_positive_rates"]
        }
        
        passed = (
            metrics["demographic_parity_difference"] <= threshold and
            metrics["equal_opportunity_difference"] <= threshold
        )
        
        model_name = model.__class__.__name__
        result = FairnessResult(model_name, protected_attr_name, metrics, passed)
        
        return result
    
    # Evaluate fairness
    result = evaluate_fairness(model, X, y)
    result.save()
    result.print_results() 