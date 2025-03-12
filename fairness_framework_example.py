#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fairness Framework Integration Example

This script demonstrates how all components of the WITHIN fairness framework
work together to provide comprehensive fairness evaluation, mitigation, 
monitoring, explainability, and regulatory compliance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

# Import fairness framework components
try:
    # Assume these are implemented
    from fairness_evaluator import FairnessEvaluator
    from fairness_explainability import FairnessExplainer
    from fairness_monitoring import FairnessMonitor
    from model_card_generator import ModelCardGenerator, generate_model_card_for_ad_score_predictor
except ImportError:
    print("Using mock implementations for demonstration purposes.")
    # Mock implementations for demo purposes
    
    class FairnessEvaluator:
        def __init__(self, protected_attributes, fairness_threshold=0.05, output_dir='fairness_results'):
            self.protected_attributes = protected_attributes
            self.fairness_threshold = fairness_threshold
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            print(f"Initialized FairnessEvaluator with protected attributes: {protected_attributes}")
            
        def evaluate(self, X, y_true, y_pred, y_prob=None, calculate_intersectional=False):
            print(f"Evaluating fairness for {len(y_true)} samples")
            print(f"Calculating intersectional metrics: {calculate_intersectional}")
            
            # Create mock fairness results
            results = {
                'overall': {
                    'accuracy': 0.85,
                    'positive_rate': 0.65
                },
                'fairness_metrics': {},
                'group_metrics': {},
                'fairness_threshold': self.fairness_threshold
            }
            
            # Add metrics for each protected attribute
            for attr in self.protected_attributes:
                # Get values for this attribute
                values = X[attr].unique()
                results['group_metrics'][attr] = {}
                
                for value in values:
                    # Add metrics for this group
                    results['group_metrics'][attr][value] = {
                        'count': X[X[attr] == value].shape[0],
                        'positive_rate': 0.5 + np.random.uniform(-0.1, 0.1),
                        'accuracy': 0.8 + np.random.uniform(-0.1, 0.1)
                    }
                
                # Add fairness metrics
                results['fairness_metrics'][f"{attr}_demographic_parity"] = {
                    'difference': np.random.uniform(0, 0.1),
                    'passes_threshold': True
                }
                
                results['fairness_metrics'][f"{attr}_equal_opportunity"] = {
                    'difference': np.random.uniform(0, 0.1),
                    'passes_threshold': True
                }
            
            # Add intersectional metrics if requested
            if calculate_intersectional:
                results['intersectional'] = {
                    'fairness_metrics': {},
                    'group_metrics': {}
                }
                
                # Add metrics for pairs of protected attributes
                for i, attr1 in enumerate(self.protected_attributes):
                    for attr2 in self.protected_attributes[i+1:]:
                        intersection = f"{attr1}+{attr2}"
                        
                        # Add intersection fairness metrics
                        results['intersectional']['fairness_metrics'][f"{intersection}_demographic_parity"] = {
                            'difference': np.random.uniform(0, 0.1),
                            'passes_threshold': True
                        }
                        
                        # Add intersection group metrics
                        results['intersectional']['group_metrics'][intersection] = {}
                        
                        for val1 in X[attr1].unique():
                            for val2 in X[attr2].unique():
                                group_key = f"{val1}+{val2}"
                                results['intersectional']['group_metrics'][intersection][group_key] = {
                                    'count': 100,  # Mock value
                                    'positive_rate': 0.5 + np.random.uniform(-0.1, 0.1)
                                }
            
            return results
        
        def plot_fairness_metrics(self, results, output_dir=None):
            print("Generating fairness metrics visualizations...")
            # Create a mock visualization for demo purposes
            plt.figure(figsize=(10, 6))
            plt.title("Fairness Metrics by Protected Attribute")
            plt.bar(['Gender DP', 'Gender EO', 'Age DP', 'Age EO'], 
                   [0.05, 0.03, 0.08, 0.04])
            plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
            plt.ylabel('Disparity')
            plt.legend()
            
            output_dir = output_dir or self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/fairness_metrics.png")
            plt.close()
            
            return f"{output_dir}/fairness_metrics.png"
        
        def plot_group_metrics(self, results, output_dir=None):
            print("Generating group metrics visualizations...")
            # Create a mock visualization for demo purposes
            plt.figure(figsize=(10, 6))
            plt.title("Positive Rate by Group")
            plt.bar(['Male', 'Female', '18-25', '26-35', '36-50', '51+'], 
                   [0.62, 0.68, 0.58, 0.64, 0.70, 0.63])
            plt.ylabel('Positive Rate')
            
            output_dir = output_dir or self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/group_metrics.png")
            plt.close()
            
            return f"{output_dir}/group_metrics.png"
        
        def plot_intersectional_fairness(self, results, output_dir=None):
            print("Generating intersectional fairness visualizations...")
            if 'intersectional' not in results:
                print("No intersectional metrics available")
                return None
                
            # Create a mock visualization for demo purposes
            plt.figure(figsize=(10, 6))
            plt.title("Intersectional Fairness Analysis")
            plt.imshow(np.random.uniform(0.5, 0.7, (4, 2)))
            plt.colorbar(label='Positive Rate')
            plt.xlabel('Gender')
            plt.ylabel('Age Group')
            plt.xticks([0, 1], ['Male', 'Female'])
            plt.yticks([0, 1, 2, 3], ['18-25', '26-35', '36-50', '51+'])
            
            output_dir = output_dir or self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/intersectional_fairness.png")
            plt.close()
            
            return f"{output_dir}/intersectional_fairness.png"
    
    class FairnessExplainer:
        def __init__(self, protected_attributes, shap_explainer_type='tree', output_dir='fairness_explanations'):
            self.protected_attributes = protected_attributes
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            print(f"Initialized FairnessExplainer with protected attributes: {protected_attributes}")
            
        def explain_model(self, model, X, y=None, feature_names=None):
            print(f"Generating SHAP explanations for {len(X)} samples")
            
            # Create mock explanations
            os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
            
            # Create mock summary plot
            plt.figure(figsize=(10, 6))
            plt.title("SHAP Summary Plot")
            plt.barh(['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'], 
                    [0.3, 0.25, 0.2, 0.15, 0.1])
            plt.xlabel('SHAP Value')
            plt.savefig(f"{self.output_dir}/plots/shap_summary.png")
            plt.close()
            
            # Create mock group comparison plot
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance by Gender")
            x = np.arange(5)
            width = 0.35
            plt.bar(x - width/2, [0.3, 0.25, 0.2, 0.15, 0.1], width, label='Male')
            plt.bar(x + width/2, [0.25, 0.3, 0.15, 0.2, 0.1], width, label='Female')
            plt.xticks(x, ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
            plt.xlabel('Feature')
            plt.ylabel('SHAP Value')
            plt.legend()
            plt.savefig(f"{self.output_dir}/plots/group_comparison.png")
            plt.close()
            
            # Create markdown report
            with open(f"{self.output_dir}/fairness_explanation_report.md", 'w') as f:
                f.write("# Fairness Explanation Report\n\n")
                f.write("## Overview\n\n")
                f.write("This report provides explanations of model predictions through a fairness lens.\n\n")
                f.write("## Feature Importance\n\n")
                f.write("![SHAP Summary Plot](plots/shap_summary.png)\n\n")
                f.write("## Group Comparison\n\n")
                f.write("![Group Comparison](plots/group_comparison.png)\n\n")
                f.write("## Conclusion\n\n")
                f.write("The model uses similar features for different demographic groups, with slight variations in importance.\n")
            
            return {
                'overall_importance': {
                    'feature_1': 0.3,
                    'feature_2': 0.25,
                    'feature_3': 0.2,
                    'feature_4': 0.15,
                    'feature_5': 0.1
                },
                'group_importance': {
                    'gender': {
                        'male': {
                            'feature_1': 0.3,
                            'feature_2': 0.25
                        },
                        'female': {
                            'feature_1': 0.25,
                            'feature_2': 0.3
                        }
                    }
                },
                'plots': {
                    'summary': f"{self.output_dir}/plots/shap_summary.png",
                    'group_comparison': f"{self.output_dir}/plots/group_comparison.png"
                },
                'report': f"{self.output_dir}/fairness_explanation_report.md"
            }
    
    class FairnessMonitor:
        """Mock implementation of FairnessMonitor."""
        
        def __init__(self, protected_attributes, fairness_metrics, alert_threshold=0.05, 
                    monitoring_dir='fairness_monitoring', rolling_window=3):
            self.protected_attributes = protected_attributes
            self.fairness_metrics = fairness_metrics
            self.alert_threshold = alert_threshold
            self.monitoring_dir = monitoring_dir
            self.rolling_window = rolling_window
            
            # Create monitoring directory
            os.makedirs(monitoring_dir, exist_ok=True)
            
            print(f"Initialized FairnessMonitor with protected attributes: {protected_attributes}")
        
        def set_baseline_metrics(self, baseline_metrics):
            """Set baseline fairness metrics."""
            self.baseline_metrics = baseline_metrics
            
            if not os.path.exists(f"{self.monitoring_dir}/baseline"):
                os.makedirs(f"{self.monitoring_dir}/baseline", exist_ok=True)
                with open(f"{self.monitoring_dir}/baseline/metrics.json", 'w') as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "metrics": "baseline_metrics"}, f)
            
            print("Baseline metrics set")
        
        def update(self, X, y_pred, y_true, batch_id="batch_1"):
            """Update monitoring with new predictions."""
            # Create batch directory
            batch_dir = f"{self.monitoring_dir}/{batch_id}"
            os.makedirs(batch_dir, exist_ok=True)
            
            print(f"Updating monitor with batch {batch_id}")
            
            # Create visualizations for the batch
            self._create_visualizations(batch_id)
            
            # Check for fairness alerts
            alerts = []
            for attr in self.protected_attributes:
                for metric in self.fairness_metrics:
                    # Mock alert generation (would be based on actual metrics in real implementation)
                    if attr == 'gender' and metric == 'demographic_parity':
                        alerts.append({
                            'metric': f"{attr}_{metric}",
                            'current': 0.0758,
                            'threshold': self.alert_threshold,
                            'severity': 'high' if 0.0758 > self.alert_threshold else 'low'
                        })
                    elif attr == 'age_group' and metric == 'equal_opportunity':
                        alerts.append({
                            'metric': f"{attr}_{metric}",
                            'current': 0.0925,
                            'threshold': self.alert_threshold,
                            'severity': 'high' if 0.0925 > self.alert_threshold else 'low'
                        })
            
            # Return batch metrics (mock implementation)
            batch_metrics = {attr: {metric: {'value': 0.05} for metric in self.fairness_metrics} 
                           for attr in self.protected_attributes}
            
            return batch_metrics, alerts
        
        def _create_visualizations(self, batch_id):
            """Create visualizations for the specified batch."""
            print(f"Creating visualizations for batch {batch_id}")
            
            # In a real implementation, this would create actual visualizations
            # Mock visualization creation
            batch_dir = f"{self.monitoring_dir}/{batch_id}"
            for attr in self.protected_attributes:
                for metric in self.fairness_metrics:
                    with open(f"{batch_dir}/{attr}_{metric}_trend.png", 'w') as f:
                        f.write(f"Mock visualization for {attr} {metric}")
        
        def get_trend_analysis(self, metric_name):
            """Get trend analysis for a specific metric."""
            print(f"Generating trend analysis for {metric_name}")
            
            # Mock trend analysis
            return {
                'metric': metric_name,
                'trend': 'stable',
                'alerts_count': 3,
                'visualization': f"{self.monitoring_dir}/trends/{metric_name}.png"
            }

# Mock implementation of ModelCardGenerator
class ModelCardGenerator:
    """Generate model cards with regulatory compliance information and fairness metrics."""
    
    def __init__(self, model_info, output_dir="model_cards", template_dir=None):
        """
        Initialize the model card generator.
        
        Args:
            model_info (dict): Information about the model (name, version, type, etc.)
            output_dir (str): Directory to save model cards
            template_dir (str): Directory containing template files (optional)
        """
        self.model_info = model_info
        self.output_dir = output_dir
        self.template_dir = template_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Initialized ModelCardGenerator for {model_info.get('name', 'unnamed model')}")
    
    def generate_model_card(self, performance_metrics, fairness_results, mitigation_results=None, 
                           card_type="comprehensive", visualizations=None):
        """
        Generate a model card based on provided data.
        
        Args:
            performance_metrics (dict): Model performance metrics
            fairness_results (dict): Results of fairness evaluation
            mitigation_results (dict, optional): Results after applying fairness mitigation
            card_type (str): Type of model card (comprehensive, regulatory, technical)
            visualizations (list, optional): Paths to visualization files to include
            
        Returns:
            dict: Paths to generated model card files
        """
        # Generate filename based on model info and card type
        base_filename = f"{self.model_info.get('name', 'model').replace(' ', '_').lower()}_-_{card_type}"
        version_str = f"{self.model_info.get('version', '1.0.0')}".replace('.', '_')
        filename = f"{base_filename}_{version_str}"
        
        # Create content for model card (simplified for demo)
        content = self._create_model_card_content(
            performance_metrics, fairness_results, mitigation_results, card_type
        )
        
        # Save model card in different formats
        file_paths = {}
        
        # Save as markdown
        md_path = os.path.join(self.output_dir, f"{filename}_model_card.md")
        with open(md_path, 'w') as f:
            f.write(content)
        file_paths['markdown'] = md_path
        
        # Save metadata as JSON
        metadata = self._create_metadata(
            performance_metrics, fairness_results, mitigation_results
        )
        json_path = os.path.join(self.output_dir, f"{filename}_model_card_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        file_paths['json'] = json_path
        
        # Create HTML version (simplified for demo)
        html_content = f"<html><head><title>{self.model_info.get('name')} Model Card</title></head>"
        html_content += f"<body><pre>{content}</pre></body></html>"
        html_path = os.path.join(self.output_dir, f"{filename}_model_card.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        file_paths['html'] = html_path
        
        print(f"Generated {card_type} model card for {self.model_info.get('name')}:")
        for format_name, path in file_paths.items():
            print(f"  - {format_name.capitalize()}: {path}")
            
        return file_paths
    
    def _create_model_card_content(self, performance_metrics, fairness_results, 
                                  mitigation_results, card_type):
        """Create the content for the model card based on the card type."""
        model_name = self.model_info.get('name', 'Unnamed Model')
        version = self.model_info.get('version', '1.0.0')
        model_type = self.model_info.get('model_type', 'Unknown')
        
        content = f"# Model Card: {model_name} (v{version})\n\n"
        content += f"## Model Details\n\n"
        content += f"- **Name:** {model_name}\n"
        content += f"- **Version:** {version}\n"
        content += f"- **Type:** {model_type}\n"
        content += f"- **Date:** {self.model_info.get('date', 'Not specified')}\n"
        content += f"- **Authors:** {self.model_info.get('authors', 'Not specified')}\n\n"
        
        content += f"## Model Performance\n\n"
        content += f"- **Accuracy:** {performance_metrics.get('accuracy', 'Not evaluated'):.4f}\n"
        content += f"- **Precision:** {performance_metrics.get('precision', 'Not evaluated'):.4f}\n"
        content += f"- **Recall:** {performance_metrics.get('recall', 'Not evaluated'):.4f}\n"
        content += f"- **F1 Score:** {performance_metrics.get('f1', 'Not evaluated'):.4f}\n\n"
        
        content += f"## Fairness Evaluation\n\n"
        
        for attr in fairness_results.get('group_metrics', {}).keys():
            content += f"### {attr.capitalize()} Group Metrics\n\n"
            metrics = fairness_results['group_metrics'][attr]
            dp = metrics.get('demographic_parity', {}).get('value', 'Not calculated')
            eo = metrics.get('equal_opportunity', {}).get('value', 'Not calculated')
            
            if isinstance(dp, (int, float)):
                dp = f"{dp:.4f}"
            if isinstance(eo, (int, float)):
                eo = f"{eo:.4f}"
                
            content += f"- **Demographic Parity:** {dp}\n"
            content += f"- **Equal Opportunity:** {eo}\n\n"
        
        if mitigation_results:
            content += f"## Fairness Mitigation\n\n"
            content += f"- **Mitigation Technique:** {mitigation_results.get('technique', 'Not specified')}\n"
            
            for attr in mitigation_results.get('mitigated_metrics', {}).keys():
                content += f"### {attr.capitalize()} Group Metrics After Mitigation\n\n"
                metrics = mitigation_results['mitigated_metrics'][attr]
                dp = metrics.get('demographic_parity', {}).get('value', 'Not calculated')
                eo = metrics.get('equal_opportunity', {}).get('value', 'Not calculated')
                
                if isinstance(dp, (int, float)):
                    dp = f"{dp:.4f}"
                if isinstance(eo, (int, float)):
                    eo = f"{eo:.4f}"
                    
                content += f"- **Demographic Parity:** {dp}\n"
                content += f"- **Equal Opportunity:** {eo}\n\n"
        
        if card_type == "regulatory" or card_type == "comprehensive":
            content += f"## Ethical Considerations\n\n"
            content += f"- **Protected Attributes:** {', '.join(fairness_results.get('protected_attributes', []))}\n"
            content += f"- **Potential Risks:** Bias in model predictions across protected groups\n"
            content += f"- **Mitigation Strategies:** {mitigation_results.get('technique', 'None applied') if mitigation_results else 'None applied'}\n\n"
            
            content += f"## Regulatory Compliance\n\n"
            content += f"- **Compliance Frameworks:**\n"
            content += f"  - EU AI Act (2023)\n"
            content += f"  - NIST AI RMF (AI Risk Management Framework)\n\n"
            
            content += f"- **Documentation Standards:**\n"
            content += f"  - NIST AI RMF ID.GV 1.2: Explainability requirements are established\n"
            content += f"  - NIST AI RMF ID.RA 2.9: Fairness considerations are evaluated\n\n"
            
            content += f"## Limitations and Warnings\n\n"
            content += f"- This model shows some bias across protected attributes that requires ongoing monitoring\n"
            content += f"- Fairness metrics should be continuously evaluated in production\n"
            content += f"- Mitigation techniques may impact overall model performance\n\n"
        
        return content
    
    def _create_metadata(self, performance_metrics, fairness_results, mitigation_results):
        """Create structured metadata for the model card."""
        metadata = {
            "model_info": self.model_info,
            "performance_metrics": performance_metrics,
            "fairness": {
                "protected_attributes": fairness_results.get('protected_attributes', []),
                "group_metrics": fairness_results.get('group_metrics', {}),
                "intersectional_metrics": fairness_results.get('intersectional_metrics', {})
            },
            "mitigation": None if not mitigation_results else {
                "technique": mitigation_results.get('technique', 'Not specified'),
                "mitigated_metrics": mitigation_results.get('mitigated_metrics', {})
            },
            "regulatory_compliance": {
                "frameworks": ["EU AI Act", "NIST AI RMF"],
                "documentation_standards": [
                    "NIST AI RMF ID.GV 1.2", 
                    "NIST AI RMF ID.RA 2.9"
                ]
            }
        }
        return metadata

# Helper function to generate synthetic data
def generate_synthetic_data(n_samples=1000, n_features=5, bias_factor=0.7):
    """
    Generate synthetic data with bias for fairness evaluation.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features to generate
        bias_factor (float): Factor determining bias strength (0-1)
        
    Returns:
        tuple: (data DataFrame with features and target, list of protected attribute names)
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create protected attributes: gender (binary) and age_group (multi-category)
    gender = np.random.binomial(1, 0.5, size=n_samples)  # 0: male, 1: female
    age_group = np.random.choice(['18-25', '26-35', '36-50', '51+'], size=n_samples, 
                               p=[0.25, 0.35, 0.25, 0.15])
    
    # Create DataFrame with features and protected attributes
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    X_df['gender'] = gender
    X_df['age_group'] = age_group
    
    # Create target with bias - females (gender=1) are less likely to get positive outcome
    # and older age groups also have lower probability
    base_score = np.sum(X[:, :2], axis=1)  # Base prediction from first two features
    
    # Add bias based on gender
    gender_bias = -bias_factor * gender  # Negative bias for females
    
    # Add bias based on age group
    age_bias = np.zeros(n_samples)
    age_bias[age_group == '36-50'] = -bias_factor * 0.5  # Moderate negative bias
    age_bias[age_group == '51+'] = -bias_factor  # Strong negative bias
    
    # Combine scores and convert to probability
    score = base_score + gender_bias + age_bias
    prob = 1 / (1 + np.exp(-score))  # Sigmoid to get probability
    
    # Generate binary target
    y = np.random.binomial(1, prob)
    
    # Add target to DataFrame
    data = X_df.copy()
    data['target'] = y
    
    # List of protected attributes
    protected_attributes = ['gender', 'age_group']
    
    return data, protected_attributes

def main():
    """Run the fairness framework integration example."""
    print("\n================================================================================")
    print("WITHIN Fairness Framework Integration Example")
    print("================================================================================\n")
    
    # Define output directories
    fairness_results_dir = "fairness_results"
    fairness_explanations_dir = "fairness_explanations"
    fairness_monitoring_dir = "fairness_monitoring"
    
    # 1. Generate synthetic data
    print("1. Generating synthetic data with bias...")
    data, protected_attributes = generate_synthetic_data()
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Generated {len(data)} samples with features: {list(X.columns)}")
    print(f"Protected attributes: {', '.join(protected_attributes)}")
    print(f"Positive rate overall: {y.mean():.2f}")
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Step 2: Train a simple model
    print("\n2. Training a model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.drop(['gender', 'age_group'], axis=1), y_train)
    
    # Make predictions
    y_pred = model.predict(X_test.drop(['gender', 'age_group'], axis=1))
    y_prob = model.predict_proba(X_test.drop(['gender', 'age_group'], axis=1))[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score: {f1_score(y_test, y_pred):.4f}")
    
    # Step 3: Evaluate fairness
    print("\n3. Evaluating fairness metrics...")
    fairness_evaluator = FairnessEvaluator(
        protected_attributes=['gender', 'age_group'],
        fairness_threshold=0.05,
        output_dir='fairness_results'
    )
    
    fairness_results = fairness_evaluator.evaluate(
        X_test, y_test, y_pred, y_prob, 
        calculate_intersectional=True
    )
    
    # Generate visualizations
    fairness_metrics_viz = fairness_evaluator.plot_fairness_metrics(fairness_results)
    group_metrics_viz = fairness_evaluator.plot_group_metrics(fairness_results)
    intersectional_viz = fairness_evaluator.plot_intersectional_fairness(fairness_results)
    
    # Step 4: Generate explainability analysis
    print("\n4. Generating fairness explanations...")
    fairness_explainer = FairnessExplainer(
        protected_attributes=['gender', 'age_group'],
        output_dir='fairness_explanations'
    )
    
    explanation_results = fairness_explainer.explain_model(
        model, X_test.drop(['gender', 'age_group'], axis=1), 
        feature_names=[f'feature_{i}' for i in range(5)]
    )
    
    # Step 5: Set up fairness monitoring
    print("\n5. Setting up fairness monitoring...")
    monitor = FairnessMonitor(
        protected_attributes=['gender', 'age_group'],
        fairness_metrics=['demographic_parity', 'equal_opportunity'],
        alert_threshold=0.05,
        monitoring_dir='fairness_monitoring'
    )
    
    # Set baseline metrics
    monitor.set_baseline_metrics(fairness_results)
    
    # 6. Simulate fairness monitoring over time
    print("\n6. Simulating fairness monitoring over time...")
    
    # Create a function to generate new batches of data with varying bias
    def generate_monitoring_batch(batch_id, n_samples=200, bias_shift=0.1):
        # Generate data with slightly different bias for monitoring
        new_data, _ = generate_synthetic_data(n_samples=n_samples, 
                                           bias_factor=0.7 + bias_shift * batch_id)
        return new_data
    
    # Simulate monitoring for multiple batches
    for batch_id in range(1, 4):
        print(f"\nBatch {batch_id}:")
        
        # Generate new batch of data
        batch_data = generate_monitoring_batch(batch_id)
        
        # Get features and target
        X_new = batch_data.drop('target', axis=1)
        y_new = batch_data['target']
        
        # Get only the feature columns that were used during training
        # (exclude protected attributes and target)
        X_new_features = X_new.drop(protected_attributes, axis=1)
        
        # Make predictions
        y_new_pred = model.predict(X_new_features)
        
        # Update the monitor with new predictions
        batch_metrics, alerts = monitor.update(X_new, y_new_pred, y_new, batch_id=f"batch_{batch_id}")
        
        if alerts:
            print(f"Alerts detected: {len(alerts)}")
            for alert in alerts:
                print(f"  - {alert['metric']}: {alert['current']:.4f} (threshold: {alert['threshold']:.4f}, severity: {alert['severity']})")
    
    # Get trend analysis for a specific metric
    trend_analysis = monitor.get_trend_analysis('gender_demographic_parity')
    print(f"\nTrend analysis for gender_demographic_parity: {trend_analysis['trend']}")
    
    # Step 7: Generate model card for regulatory compliance
    print("\n7. Generating model card for regulatory compliance...")
    
    model_info = {
        "name": "Ad Score Predictor",
        "version": "2.1.0",
        "model_type": "Random Forest Classifier",
        "date": "2023-10-15",
        "authors": "WITHIN AI Team",
        "purpose": "Predict ad performance scores with fairness considerations"
    }
    
    generator = ModelCardGenerator(model_info, output_dir="model_cards")
    
    # Performance metrics from model
    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    # Create fairness results structure
    fairness_results = {
        "protected_attributes": ['gender', 'age_group'],
        "group_metrics": {
            'gender': {
                'demographic_parity': {'value': 0.02, 'threshold': 0.05},
                'equal_opportunity': {'value': 0.03, 'threshold': 0.05}
            },
            'age_group': {
                'demographic_parity': {'value': 0.03, 'threshold': 0.05},
                'equal_opportunity': {'value': 0.04, 'threshold': 0.05}
            }
        },
        "intersectional_metrics": {}  # In a real implementation, this would contain actual intersectional metrics
    }
    
    # Mock mitigation results
    mitigation_results = {
        "technique": "Reweighing",
        "mitigated_metrics": {
            "gender": {
                "demographic_parity": {"value": 0.02, "threshold": 0.05},
                "equal_opportunity": {"value": 0.03, "threshold": 0.05}
            },
            "age_group": {
                "demographic_parity": {"value": 0.03, "threshold": 0.05},
                "equal_opportunity": {"value": 0.04, "threshold": 0.05}
            }
        }
    }
    
    # Generate model cards
    generator.generate_model_card(
        performance_metrics=performance_metrics,
        fairness_results=fairness_results,
        mitigation_results=mitigation_results,
        card_type="regulatory"
    )
    
    # Also generate a technical model card without mitigation info
    generator.generate_model_card(
        performance_metrics=performance_metrics,
        fairness_results=fairness_results,
        card_type="technical"
    )
    
    print("\nFairness Framework Integration Example Complete!")
    print("===============================================================")
    print("Output files:")
    print(f"- Fairness results: {fairness_results_dir}")
    print(f"- Fairness explanations: {fairness_explanations_dir}")
    print(f"- Fairness monitoring: {fairness_monitoring_dir}")
    print(f"- Model cards: model_cards/")
    print("===============================================================")

if __name__ == "__main__":
    main() 