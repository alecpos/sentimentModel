# Ethical AI Implementation

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document details the implementation of ethical AI practices within the WITHIN ML platform. It covers fairness assessment, bias mitigation techniques, privacy-preserving mechanisms, and explainability components integrated throughout the ML pipeline.

## Table of Contents

1. [Fairness Assessment Framework](#fairness-assessment-framework)
2. [Bias Mitigation Techniques](#bias-mitigation-techniques)
3. [Privacy-Preserving Machine Learning](#privacy-preserving-machine-learning)
4. [Explainability Components](#explainability-components)
5. [Monitoring and Feedback](#monitoring-and-feedback)
6. [Ethical Review Process](#ethical-review-process)
7. [Implementation Examples](#implementation-examples)

## Fairness Assessment Framework

Our fairness assessment framework is designed to systematically evaluate models for potential bias across multiple dimensions and protected attributes.

### Fairness Metrics

The system implements these key fairness metrics:

| Metric | Implementation | Description |
|--------|----------------|-------------|
| Demographic Parity | `metrics.demographic_parity()` | Predictions should be independent of protected attributes |
| Equal Opportunity | `metrics.equal_opportunity()` | True positive rates should be equal across protected groups |
| Equalized Odds | `metrics.equalized_odds()` | False positive and true positive rates should be equal across groups |
| Disparate Impact | `metrics.disparate_impact()` | Ratio of positive prediction rates between privileged and unprivileged groups |
| Intersectional Fairness | `metrics.intersectional_fairness()` | Evaluates fairness across combinations of protected attributes |

### Implementation Details

The fairness assessment is implemented in the evaluation phase using the `FairnessEvaluator` class:

```python
class FairnessEvaluator:
    """Evaluates model fairness across protected attributes"""
    
    def __init__(self, 
                 protected_attributes: List[str],
                 metrics: List[str] = ["demographic_parity", "equal_opportunity"],
                 threshold: float = 0.05):
        """
        Args:
            protected_attributes: List of protected attribute column names
            metrics: List of fairness metrics to calculate
            threshold: Threshold for acceptable fairness disparity
        """
        self.protected_attributes = protected_attributes
        self.metrics = metrics
        self.threshold = threshold
        
    def evaluate(self, 
                model: Any, 
                data: pd.DataFrame, 
                target_column: str) -> Dict[str, Any]:
        """Evaluate model fairness on dataset
        
        Args:
            model: Model to evaluate
            data: Dataset containing features and protected attributes
            target_column: Name of target column
            
        Returns:
            Dictionary of fairness metrics
        """
        results = {}
        
        # Get model predictions
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(data.drop(columns=[target_column]))
            y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
        else:
            y_pred = model.predict(data.drop(columns=[target_column]))
        
        y_true = data[target_column]
        
        # Evaluate each protected attribute
        for attr in self.protected_attributes:
            attr_results = {}
            
            # Calculate metrics for this attribute
            if "demographic_parity" in self.metrics:
                attr_results["demographic_parity"] = self._calculate_demographic_parity(
                    y_pred, data[attr]
                )
                
            if "equal_opportunity" in self.metrics:
                attr_results["equal_opportunity"] = self._calculate_equal_opportunity(
                    y_pred, y_true, data[attr]
                )
                
            if "equalized_odds" in self.metrics:
                attr_results["equalized_odds"] = self._calculate_equalized_odds(
                    y_pred, y_true, data[attr]
                )
            
            # Add intersectional metrics if multiple attributes are provided
            if len(self.protected_attributes) > 1 and "intersectional_fairness" in self.metrics:
                other_attrs = [a for a in self.protected_attributes if a != attr]
                for other_attr in other_attrs:
                    attr_results[f"intersectional_with_{other_attr}"] = self._calculate_intersectional_fairness(
                        y_pred, y_true, data[attr], data[other_attr]
                    )
            
            results[attr] = attr_results
            
        # Overall fairness evaluation
        all_disparities = []
        for attr, metrics in results.items():
            for metric, value in metrics.items():
                if isinstance(value, dict) and "disparity" in value:
                    all_disparities.append(value["disparity"])
        
        results["overall"] = {
            "max_disparity": max(all_disparities) if all_disparities else 0,
            "mean_disparity": sum(all_disparities) / len(all_disparities) if all_disparities else 0,
            "passes_threshold": all(d <= self.threshold for d in all_disparities) if all_disparities else True
        }
        
        return results
```

### Usage in the Training Pipeline

The fairness assessment is integrated into the training pipeline in two ways:

1. **Training-time Fairness Constraints**: Implemented as custom loss functions that penalize unfair models
2. **Evaluation-time Fairness Metrics**: Used to generate comprehensive fairness reports during model evaluation

## Bias Mitigation Techniques

The platform implements the following bias mitigation techniques:

### Pre-processing Techniques

- **Resampling**: Implemented in `mitigations.FairResampler` to balance protected attributes
- **Variable Repair**: Implemented in `mitigations.VariableRepair` to remove correlations between features and protected attributes

### In-processing Techniques

- **Adversarial Debiasing**: Implemented in `mitigations.AdversarialDebiasing` to train a fair model using an adversary
- **Fairness Constraints**: Implemented in `mitigations.FairnessConstraints` to add fairness constraints to the optimization objective

Example implementation of adversarial debiasing:

```python
class AdversarialDebiasing(nn.Module):
    """Implements adversarial debiasing for fairness
    
    Mitigates bias by training a discriminator to predict protected attributes
    from the model's internal representations, and training the main model
    to fool the discriminator.
    """
    
    def __init__(self, 
                 predictor: nn.Module,
                 protected_dim: int,
                 lambda_param: float = 1.0):
        """
        Args:
            predictor: Main prediction model
            protected_dim: Dimensionality of protected attributes
            lambda_param: Weight for adversarial loss
        """
        super().__init__()
        self.predictor = predictor
        self.lambda_param = lambda_param
        
        # Adversary architecture
        hidden_dim = 128
        
        self.adversary = nn.Sequential(
            nn.Linear(predictor.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, protected_dim)
        )
        
    def forward(self, x):
        # Get intermediate representation and prediction
        features, prediction = self.predictor.get_features_and_prediction(x)
        
        # Adversary tries to predict protected attributes
        protected_pred = self.adversary(features.detach())
        
        return prediction, protected_pred
        
    def get_adversarial_loss(self, protected_pred, protected_true):
        """Calculate adversarial loss"""
        return F.binary_cross_entropy_with_logits(protected_pred, protected_true)
        
    def training_step(self, x, y, protected):
        """Perform adversarial training step
        
        Args:
            x: Input features
            y: Target values
            protected: Protected attributes
            
        Returns:
            Dictionary with losses and predictions
        """
        # Step 1: Train predictor to predict y while fooling adversary
        features, prediction = self.predictor.get_features_and_prediction(x)
        protected_pred = self.adversary(features)
        
        # Main task loss
        task_loss = F.binary_cross_entropy_with_logits(prediction, y)
        
        # Adversarial loss (note the gradient reversal by using negative lambda)
        adv_loss = -self.lambda_param * F.binary_cross_entropy_with_logits(
            protected_pred, protected
        )
        
        predictor_loss = task_loss + adv_loss
        
        # Step 2: Train adversary to predict protected attributes
        features_detached = features.detach()
        protected_pred_detached = self.adversary(features_detached)
        adversary_loss = F.binary_cross_entropy_with_logits(
            protected_pred_detached, protected
        )
        
        return {
            "predictor_loss": predictor_loss,
            "adversary_loss": adversary_loss,
            "task_loss": task_loss,
            "adv_loss": adv_loss,
            "prediction": prediction
        }
```

### Post-processing Techniques

- **Threshold Optimization**: Implemented in `mitigations.ThresholdOptimizer` to find optimal decision thresholds for different groups
- **Calibration**: Implemented in `mitigations.FairCalibration` to ensure calibration across protected groups

## Privacy-Preserving Machine Learning

The platform implements privacy-preserving machine learning through differential privacy:

### Differential Privacy Training

```python
class DPTrainingValidator:
    """Validates training for differential privacy compliance"""
    
    def __init__(self, 
                 epsilon: float = 2.0,
                 delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        """
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Probability of privacy failure
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
    def validate_batch_size(self, batch_size: int, dataset_size: int) -> bool:
        """Validates if batch size is appropriate for DP training
        
        Args:
            batch_size: Training batch size
            dataset_size: Total dataset size
            
        Returns:
            Whether batch size is appropriate
        """
        # Heuristic: For good privacy, batch size should be
        # small relative to dataset size
        return batch_size <= 0.01 * dataset_size
    
    def compute_noise_multiplier(self, 
                                batch_size: int,
                                dataset_size: int,
                                epochs: int) -> float:
        """Compute noise multiplier for DP-SGD
        
        Args:
            batch_size: Training batch size
            dataset_size: Total dataset size
            epochs: Number of training epochs
            
        Returns:
            Noise multiplier value
        """
        # Compute steps based on dataset size, batch size and epochs
        steps = (dataset_size // batch_size) * epochs
        
        # Compute sampling rate
        sampling_rate = batch_size / dataset_size
        
        # Use analytical formula to determine noise multiplier
        # This is a simplified version - the real implementation
        # would use a privacy accountant like RDP or zCDP
        return 1.0 * (
            np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        ) * np.sqrt(steps * sampling_rate)
    
    def estimate_epsilon(self, 
                        noise_multiplier: float,
                        batch_size: int,
                        dataset_size: int,
                        epochs: int) -> float:
        """Estimate privacy budget consumed
        
        Args:
            noise_multiplier: Noise multiplier used
            batch_size: Training batch size
            dataset_size: Total dataset size
            epochs: Number of training epochs
            
        Returns:
            Estimated epsilon value
        """
        # Simplified estimate - real implementation would use
        # a full privacy accountant
        steps = (dataset_size // batch_size) * epochs
        sampling_rate = batch_size / dataset_size
        
        return (
            noise_multiplier * self.epsilon / 
            (np.sqrt(2 * np.log(1.25 / self.delta)) * 
             np.sqrt(steps * sampling_rate))
        )
```

### Anonymous Feature Generation

The platform includes components for generating anonymized features with privacy guarantees:

1. **k-Anonymization**: Ensures each feature combination appears at least k times
2. **Differential Privacy for Feature Aggregation**: Adds calibrated noise when computing aggregate features
3. **Federated Computation**: Enables computing features across organizations without sharing raw data

## Explainability Components

The platform includes multiple explainability components:

### Local Explanations

- **SHAP Integration**: Implemented in `explainers.SHAPExplainer` for feature attribution
- **LIME Integration**: Implemented in `explainers.LIMEExplainer` for local explanations
- **Counterfactual Explanations**: Implemented in `explainers.CounterfactualExplainer` for actionable insights

### Global Explanations

- **Feature Importance**: Implemented in `explainers.FeatureImportance` for model-level importance
- **Partial Dependence Plots**: Implemented in `explainers.PartialDependence` for feature impact analysis
- **Global Surrogate Models**: Implemented in `explainers.GlobalSurrogate` for interpretable approximations

Example of the SHAP integration:

```python
class SHAPExplainer:
    """Provides SHAP explanations for model predictions"""
    
    def __init__(self, model, features: List[str], background_data: Optional[pd.DataFrame] = None):
        """
        Args:
            model: Model to explain
            features: List of feature names
            background_data: Background data for SHAP (optional)
        """
        self.model = model
        self.features = features
        self.background_data = background_data
        
        # Initialize the appropriate explainer based on model type
        if hasattr(model, "predict_proba"):
            # For sklearn-like models
            self.explainer = shap.Explainer(model, background_data)
        elif isinstance(model, torch.nn.Module):
            # For PyTorch models
            self.explainer = shap.DeepExplainer(
                model, 
                torch.FloatTensor(background_data.values) if background_data is not None else None
            )
        else:
            # Generic explainer
            self.explainer = shap.Explainer(model)
    
    def explain_instance(self, 
                         instance: Union[pd.DataFrame, np.ndarray],
                         return_format: str = "values") -> Dict[str, Any]:
        """Generate explanation for a single instance
        
        Args:
            instance: Instance to explain
            return_format: Format for returned explanation ('values', 'plot', or 'both')
            
        Returns:
            Dictionary with explanation
        """
        # Convert instance to appropriate format
        if isinstance(instance, pd.DataFrame):
            instance_values = instance.values
        else:
            instance_values = instance
            
        # Generate SHAP values
        shap_values = self.explainer.shap_values(instance_values)
        
        # Prepare results based on return format
        result = {"format": return_format}
        
        if return_format in ["values", "both"]:
            # Create dictionary mapping features to SHAP values
            if isinstance(shap_values, list):
                # For multi-class output
                result["values"] = []
                for class_idx, class_shap_values in enumerate(shap_values):
                    if len(class_shap_values.shape) > 1:
                        class_shap_values = class_shap_values[0]
                    result["values"].append({
                        feature: float(value) 
                        for feature, value in zip(self.features, class_shap_values)
                    })
            else:
                # For single-class output
                values_to_use = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                result["values"] = {
                    feature: float(value) 
                    for feature, value in zip(self.features, values_to_use)
                }
        
        if return_format in ["plot", "both"]:
            # Create visualization
            result["plot"] = self._generate_plot(instance_values, shap_values)
            
        return result
    
    def explain_dataset(self, 
                       data: pd.DataFrame, 
                       return_format: str = "values") -> Dict[str, Any]:
        """Generate explanations for a dataset
        
        Args:
            data: Dataset to explain
            return_format: Format for returned explanation ('values', 'plot', or 'both')
            
        Returns:
            Dictionary with explanations
        """
        # Generate SHAP values for the entire dataset
        shap_values = self.explainer.shap_values(data.values)
        
        # Prepare results based on return format
        result = {"format": return_format}
        
        if return_format in ["values", "both"]:
            # For multi-class output
            if isinstance(shap_values, list):
                result["values"] = []
                for class_idx, class_shap_values in enumerate(shap_values):
                    result["values"].append({
                        "class": class_idx,
                        "global_importance": {
                            feature: float(np.abs(class_shap_values[:, i]).mean())
                            for i, feature in enumerate(self.features)
                        },
                        "instance_values": class_shap_values.tolist()
                    })
            else:
                # For single-class output
                result["values"] = {
                    "global_importance": {
                        feature: float(np.abs(shap_values[:, i]).mean())
                        for i, feature in enumerate(self.features)
                    },
                    "instance_values": shap_values.tolist()
                }
        
        if return_format in ["plot", "both"]:
            # Create visualization
            result["plot"] = self._generate_summary_plot(data.values, shap_values)
            
        return result
    
    def _generate_plot(self, instance_values, shap_values):
        """Generate force plot for a single instance"""
        # In a real implementation, this would generate and return
        # a serialized version of the plot
        return {"type": "force_plot", "generated": True}
    
    def _generate_summary_plot(self, data_values, shap_values):
        """Generate summary plot for a dataset"""
        # In a real implementation, this would generate and return
        # a serialized version of the plot
        return {"type": "summary_plot", "generated": True}
```

## Monitoring and Feedback

The platform implements continuous monitoring for ethical concerns:

### Fairness Monitoring

- Real-time tracking of fairness metrics in production
- Alerts for fairness degradation
- Drift detection for demographic distributions

### Feedback Mechanisms

- User feedback collection on model predictions
- Specialized reporting for ethical concerns
- Integration with incident management system

## Ethical Review Process

The platform supports a formalized ethical review process:

1. **Initial Assessment**: Automated scanning for high-risk scenarios
2. **Documentation**: Standardized ethical impact assessments
3. **Review**: Multi-disciplinary review for high-risk models
4. **Approval**: Tiered approval process based on risk level
5. **Monitoring**: Ongoing ethical monitoring

## Implementation Examples

### Implementing Fairness Constraints in Training

```python
def train_with_fairness_constraints(
    model, 
    train_loader, 
    protected_attr,
    fairness_lambda=0.1,
    fairness_metric="demographic_parity"
):
    """Train model with fairness constraints
    
    Args:
        model: Model to train
        train_loader: DataLoader with training data
        protected_attr: Protected attribute to enforce fairness on
        fairness_lambda: Weight for fairness constraint
        fairness_metric: Type of fairness metric to use
    """
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        for batch in train_loader:
            inputs, targets, protected = batch
            
            # Forward pass
            outputs = model(inputs)
            
            # Task loss
            task_loss = F.binary_cross_entropy_with_logits(outputs, targets)
            
            # Fairness constraint
            if fairness_metric == "demographic_parity":
                # Group predictions by protected attribute
                protected_pos = protected == 1
                protected_neg = protected == 0
                
                # Calculate positive prediction rates for each group
                pred_probs = torch.sigmoid(outputs)
                pos_rate_prot = pred_probs[protected_pos].mean()
                pos_rate_unprot = pred_probs[protected_neg].mean()
                
                # Demographic parity penalty
                fairness_loss = torch.abs(pos_rate_prot - pos_rate_unprot)
            
            # Total loss
            loss = task_loss + fairness_lambda * fairness_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Print metrics
        print(f"Epoch {epoch}: Task Loss = {task_loss.item():.4f}, "
              f"Fairness Loss = {fairness_loss.item():.4f}")
```

### Generating Explanations

```python
def explain_ad_score_prediction(model, ad_data):
    """Generate explanation for ad score prediction
    
    Args:
        model: Ad score prediction model
        ad_data: Ad data to explain
        
    Returns:
        Dictionary with explanation
    """
    # Create explainer
    explainer = SHAPExplainer(
        model=model,
        features=list(ad_data.columns),
        background_data=get_background_dataset()
    )
    
    # Generate explanation
    explanation = explainer.explain_instance(
        instance=ad_data,
        return_format="both"
    )
    
    # Convert to user-friendly format
    user_friendly = {
        "overall_score": model.predict(ad_data)[0],
        "key_factors": [],
        "improvement_areas": []
    }
    
    # Get feature impacts
    if isinstance(explanation["values"], dict):
        feature_impacts = explanation["values"]
    else:
        feature_impacts = explanation["values"][0]  # First class
    
    # Sort features by impact
    sorted_features = sorted(
        feature_impacts.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Extract key factors (positive impact)
    user_friendly["key_factors"] = [
        {"factor": feature, "impact": impact}
        for feature, impact in sorted_features
        if impact > 0 and abs(impact) > 0.05
    ][:3]  # Top 3 positive factors
    
    # Extract improvement areas (negative impact)
    user_friendly["improvement_areas"] = [
        {"factor": feature, "impact": abs(impact)}
        for feature, impact in sorted_features
        if impact < 0 and abs(impact) > 0.05
    ][:3]  # Top 3 negative factors
    
    return user_friendly
```

## Conclusion

This document provides a comprehensive overview of the ethical AI implementation in the WITHIN ML platform. By integrating fairness assessments, bias mitigation, privacy-preserving techniques, and explainability components throughout the ML pipeline, the platform aims to support responsible AI development and deployment.

The implementation reflects our commitment to ethical AI principles, while providing flexible, configurable components that can be tailored to specific use cases and requirements. 