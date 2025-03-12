#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fairness Explainability Module

This module provides tools for explaining model decisions through a fairness lens,
using SHAP values to understand how different features impact predictions for 
various demographic groups.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairnessExplainer:
    """
    Explains model predictions with a focus on fairness across protected groups.
    
    This class integrates SHAP explanations with fairness metrics to provide
    insights into how models make predictions for different demographic groups
    and whether there are concerning patterns that may lead to bias.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        shap_explainer_type: str = 'tree',  # 'tree', 'kernel', 'deep'
        output_dir: str = 'fairness_explanations'
    ):
        """
        Initialize the FairnessExplainer.
        
        Args:
            protected_attributes: List of protected attribute names
            shap_explainer_type: Type of SHAP explainer to use
            output_dir: Directory to save explanations and visualizations
        """
        self.protected_attributes = protected_attributes
        self.shap_explainer_type = shap_explainer_type
        self.output_dir = output_dir
        self.explainer = None
        self.feature_names = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    def _create_explainer(self, model, X_background):
        """
        Create a SHAP explainer based on the model type.
        
        Args:
            model: Trained model to explain
            X_background: Background data for the explainer
            
        Returns:
            SHAP explainer object
        """
        if self.shap_explainer_type == 'tree':
            try:
                return shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"Could not create TreeExplainer: {e}")
                logger.info("Falling back to KernelExplainer")
                return shap.KernelExplainer(model.predict, X_background)
                
        elif self.shap_explainer_type == 'kernel':
            if hasattr(model, 'predict_proba'):
                return shap.KernelExplainer(model.predict_proba, X_background)
            else:
                return shap.KernelExplainer(model.predict, X_background)
                
        elif self.shap_explainer_type == 'deep':
            return shap.DeepExplainer(model, X_background)
            
        else:
            logger.warning(f"Unknown explainer type: {self.shap_explainer_type}")
            logger.info("Using KernelExplainer as fallback")
            if hasattr(model, 'predict_proba'):
                return shap.KernelExplainer(model.predict_proba, X_background)
            else:
                return shap.KernelExplainer(model.predict, X_background)
    
    def explain_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray = None,
        protected_attributes_data: Dict[str, np.ndarray] = None,
        feature_names: List[str] = None,
        class_names: List[str] = None,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Generate fairness-aware model explanations.
        
        Args:
            model: Trained model to explain
            X: Feature data
            y: Target data (optional)
            protected_attributes_data: Dictionary mapping attribute names to values
            feature_names: Names of the features (optional)
            class_names: Names of the classes for classification (optional)
            sample_size: Number of samples to use for SHAP calculation (for large datasets)
            
        Returns:
            Dictionary containing explanation results
        """
        # Prepare data
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names = feature_names or [f"Feature {i}" for i in range(X.shape[1])]
            X_values = X
        
        # Sample data if needed
        if len(X_values) > sample_size:
            indices = np.random.choice(len(X_values), sample_size, replace=False)
            X_sample = X_values[indices]
            y_sample = y[indices] if y is not None else None
            
            # Sample protected attributes
            protected_samples = {}
            if protected_attributes_data:
                for attr, values in protected_attributes_data.items():
                    protected_samples[attr] = values[indices]
        else:
            X_sample = X_values
            y_sample = y
            protected_samples = protected_attributes_data
        
        # Create background data for SHAP
        X_background = X_sample
        
        # Create explainer
        try:
            self.explainer = self._create_explainer(model, X_background)
            logger.info(f"Created {self.shap_explainer_type} explainer")
        except Exception as e:
            logger.error(f"Error creating explainer: {e}")
            return {"error": f"Failed to create explainer: {str(e)}"}
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X_sample)
            logger.info("Calculated SHAP values")
            
            # If multi-class, shap_values will be a list of arrays
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Take the positive class for binary classification
                shap_values_for_analysis = shap_values[1] if len(shap_values) == 2 else shap_values
            else:
                shap_values_for_analysis = shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {"error": f"Failed to calculate SHAP values: {str(e)}"}
        
        # Initialize results
        results = {
            "overall": {
                "mean_shap_values": np.mean(np.abs(shap_values_for_analysis), axis=0).tolist(),
                "feature_importance": dict(zip(self.feature_names, np.mean(np.abs(shap_values_for_analysis), axis=0).tolist()))
            },
            "group_explanations": {}
        }
        
        # Create overall SHAP plots
        self._create_overall_plots(shap_values_for_analysis, X_sample)
        
        # Analyze by protected attributes
        if protected_samples:
            for attr_name, attr_values in protected_samples.items():
                if attr_name in self.protected_attributes:
                    group_results = self._analyze_by_group(
                        shap_values_for_analysis, X_sample, attr_values, attr_name
                    )
                    results["group_explanations"][attr_name] = group_results
        
        # Create fairness report
        self._create_fairness_report(results)
        
        return results
    
    def _create_overall_plots(self, shap_values, X):
        """
        Create overall SHAP plots for the model.
        
        Args:
            shap_values: SHAP values
            X: Feature data
        """
        try:
            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'overall_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create bar plot of feature importance
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'overall_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create dependence plots for top features
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(-feature_importance)[:5]  # Top 5 features
            
            for i in top_indices:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(i, shap_values, X, feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir, 'plots', f'dependence_{self.feature_names[i]}.png'),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
            
            logger.info("Created overall SHAP plots")
        except Exception as e:
            logger.error(f"Error creating overall plots: {e}")
    
    def _analyze_by_group(self, shap_values, X, group_values, attribute_name):
        """
        Analyze SHAP values for different protected groups.
        
        Args:
            shap_values: SHAP values
            X: Feature data
            group_values: Protected attribute values
            attribute_name: Name of the protected attribute
            
        Returns:
            Dictionary of group analysis results
        """
        unique_groups = np.unique(group_values)
        group_results = {}
        
        # Set up directory for this attribute
        attr_dir = os.path.join(self.output_dir, 'plots', attribute_name)
        os.makedirs(attr_dir, exist_ok=True)
        
        # Process each group
        for group in unique_groups:
            group_mask = group_values == group
            
            if not np.any(group_mask):
                continue
            
            group_shap = shap_values[group_mask]
            group_X = X[group_mask]
            
            # Calculate mean absolute SHAP values for this group
            mean_abs_shap = np.mean(np.abs(group_shap), axis=0)
            
            group_results[str(group)] = {
                "count": int(np.sum(group_mask)),
                "mean_absolute_shap": mean_abs_shap.tolist(),
                "feature_importance": dict(zip(self.feature_names, mean_abs_shap.tolist()))
            }
            
            # Create group-specific plots
            try:
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(group_shap, group_X, feature_names=self.feature_names, show=False)
                plt.title(f"{attribute_name} = {group}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(attr_dir, f'{group}_summary.png'),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
                
                # Bar plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(group_shap, group_X, feature_names=self.feature_names, plot_type="bar", show=False)
                plt.title(f"{attribute_name} = {group}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(attr_dir, f'{group}_importance.png'),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error creating plots for group {group}: {e}")
        
        # Create comparison plots
        self._create_group_comparison(group_results, attribute_name)
        
        return group_results
    
    def _create_group_comparison(self, group_results, attribute_name):
        """
        Create comparison plots for different groups.
        
        Args:
            group_results: Results for each group
            attribute_name: Name of the protected attribute
        """
        try:
            # Only proceed if we have multiple groups
            if len(group_results) < 2:
                return
            
            attr_dir = os.path.join(self.output_dir, 'plots', attribute_name)
            
            # Extract feature importance for each group
            groups = list(group_results.keys())
            feature_importance = {
                group: np.array([v for k, v in group_results[group]["feature_importance"].items()])
                for group in groups
            }
            
            # Find top features across all groups
            all_importance = np.zeros(len(self.feature_names))
            for group in groups:
                all_importance += feature_importance[group]
            
            top_indices = np.argsort(-all_importance)[:10]  # Top 10 features
            top_features = [self.feature_names[i] for i in top_indices]
            
            # Create bar chart comparing feature importance across groups
            plt.figure(figsize=(15, 8))
            x = np.arange(len(top_features))
            width = 0.8 / len(groups)
            
            for i, group in enumerate(groups):
                group_importance = [group_results[group]["feature_importance"][feature] for feature in top_features]
                plt.bar(x + (i - len(groups)/2 + 0.5) * width, group_importance, width, label=group)
            
            plt.xlabel('Features')
            plt.ylabel('Mean |SHAP value|')
            plt.title(f'Feature Importance Comparison by {attribute_name}')
            plt.xticks(x, top_features, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(attr_dir, 'feature_importance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create heatmap of feature importance differences
            if len(groups) == 2:
                # For binary attributes, create a difference plot
                group1, group2 = groups
                diff = np.array([
                    group_results[group1]["feature_importance"][f] - group_results[group2]["feature_importance"][f]
                    for f in self.feature_names
                ])
                
                # Sort by absolute difference
                sorted_indices = np.argsort(-np.abs(diff))[:20]  # Top 20 features with biggest difference
                sorted_features = [self.feature_names[i] for i in sorted_indices]
                sorted_diff = [diff[i] for i in sorted_indices]
                
                # Create horizontal bar chart
                plt.figure(figsize=(10, 12))
                colors = ['red' if x < 0 else 'blue' for x in sorted_diff]
                plt.barh(np.arange(len(sorted_features)), sorted_diff, color=colors)
                plt.yticks(np.arange(len(sorted_features)), sorted_features)
                plt.axvline(x=0, color='black', linestyle='-')
                plt.title(f'Feature Importance Difference ({group1} - {group2})')
                plt.xlabel(f'Difference in importance (positive = more important for {group1})')
                plt.tight_layout()
                plt.savefig(os.path.join(attr_dir, 'feature_importance_difference.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save the differential importance to a file
                diff_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'difference': diff,
                    'importance_for_' + group1: [group_results[group1]["feature_importance"][f] for f in self.feature_names],
                    'importance_for_' + group2: [group_results[group2]["feature_importance"][f] for f in self.feature_names]
                })
                diff_df.to_csv(os.path.join(self.output_dir, 'data', f'{attribute_name}_importance_difference.csv'), index=False)
            
            logger.info(f"Created group comparison plots for {attribute_name}")
        except Exception as e:
            logger.error(f"Error creating group comparison: {e}")
    
    def _create_fairness_report(self, results):
        """
        Create a comprehensive fairness explanation report.
        
        Args:
            results: Analysis results
        """
        report_path = os.path.join(self.output_dir, 'fairness_explanation_report.md')
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Fairness Explanation Report\n\n")
                
                f.write("## Overview\n\n")
                f.write("This report provides explanations of model behavior with a focus on fairness across protected groups.\n")
                f.write("It analyzes how the model makes decisions for different demographic groups and identifies potential sources of bias.\n\n")
                
                f.write("## Overall Model Behavior\n\n")
                f.write("The following features have the most influence on model predictions:\n\n")
                
                # Top 10 most important features
                overall_importance = results["overall"]["feature_importance"]
                top_features = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                f.write("| Feature | Importance |\n")
                f.write("|---------|------------|\n")
                for feature, importance in top_features:
                    f.write(f"| {feature} | {importance:.4f} |\n")
                
                f.write("\n### Feature Importance Visualization\n\n")
                f.write("![Overall Feature Importance](plots/overall_importance.png)\n\n")
                
                f.write("### SHAP Summary Plot\n\n")
                f.write("This plot shows how each feature affects model predictions.\n\n")
                f.write("![SHAP Summary Plot](plots/overall_summary.png)\n\n")
                
                f.write("## Fairness Analysis\n\n")
                
                # Process each protected attribute
                for attr_name, attr_results in results.get("group_explanations", {}).items():
                    f.write(f"### Analysis by {attr_name}\n\n")
                    
                    groups = list(attr_results.keys())
                    group_counts = {group: attr_results[group]["count"] for group in groups}
                    
                    f.write("#### Group Distribution\n\n")
                    f.write("| Group | Count | Percentage |\n")
                    f.write("|-------|-------|------------|\n")
                    
                    total_count = sum(group_counts.values())
                    for group, count in group_counts.items():
                        percentage = (count / total_count) * 100
                        f.write(f"| {group} | {count} | {percentage:.1f}% |\n")
                    
                    f.write("\n#### Feature Importance by Group\n\n")
                    f.write("![Feature Importance Comparison](plots/{}/feature_importance_comparison.png)\n\n".format(attr_name))
                    
                    # If there are exactly two groups, show the difference plot
                    if len(groups) == 2:
                        group1, group2 = groups
                        f.write("#### Feature Importance Difference\n\n")
                        f.write(f"This plot shows the difference in feature importance between **{group1}** and **{group2}**.\n")
                        f.write("Positive values indicate features that are more important for the first group.\n\n")
                        f.write("![Feature Importance Difference](plots/{}/feature_importance_difference.png)\n\n".format(attr_name))
                        
                        # Add interpretation
                        f.write("#### Interpretation\n\n")
                        
                        # Load the difference data
                        diff_file = os.path.join(self.output_dir, 'data', f'{attr_name}_importance_difference.csv')
                        if os.path.exists(diff_file):
                            diff_df = pd.read_csv(diff_file)
                            diff_df = diff_df.sort_values(by='difference', key=abs, ascending=False)
                            
                            # Get top 5 features with largest absolute difference
                            top_diff = diff_df.head(5)
                            
                            f.write("The model relies on different features when making predictions for different groups:\n\n")
                            
                            for _, row in top_diff.iterrows():
                                feature = row['feature']
                                diff = row['difference']
                                
                                if diff > 0:
                                    f.write(f"- **{feature}** is {abs(diff):.4f} more important for **{group1}** than for **{group2}**\n")
                                else:
                                    f.write(f"- **{feature}** is {abs(diff):.4f} more important for **{group2}** than for **{group1}**\n")
                            
                            f.write("\nThese differences might indicate potential sources of disparate treatment in the model's decision-making process.\n")
                    
                    f.write("\n#### Group-Specific SHAP Summary Plots\n\n")
                    
                    for group in groups:
                        f.write(f"##### {attr_name} = {group}\n\n")
                        f.write(f"![{group} Summary](plots/{attr_name}/{group}_summary.png)\n\n")
                
                f.write("## Conclusion and Recommendations\n\n")
                f.write("### Key Findings\n\n")
                
                # Add some general conclusions
                f.write("1. The model relies on different features for different demographic groups.\n")
                f.write("2. These differences in feature importance might contribute to disparities in model outcomes.\n")
                f.write("3. Understanding these differences can help identify potential sources of bias.\n\n")
                
                f.write("### Recommendations\n\n")
                f.write("1. **Targeted Data Collection**: Address potential representational bias in the training data.\n")
                f.write("2. **Feature Engineering**: Consider developing more equitable features based on this analysis.\n")
                f.write("3. **Model Adjustments**: Apply fairness constraints to mitigate the identified disparities.\n")
                f.write("4. **Monitoring**: Continuously monitor these patterns in production.\n")
                f.write("5. **Further Investigation**: Explore intersectional effects across multiple protected attributes.\n")
            
            logger.info(f"Created fairness explanation report at {report_path}")
        except Exception as e:
            logger.error(f"Error creating fairness report: {e}")


def main():
    """
    Run a demonstration of the fairness explainability tools.
    """
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Create output directory
    output_dir = 'fairness_explanations_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data with bias
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    gender = np.random.choice(['male', 'female'], size=n_samples, p=[0.6, 0.4])
    location = np.random.choice(['urban', 'suburban', 'rural'], size=n_samples)
    
    # Create gender-based differences
    education_level = np.zeros(n_samples)
    for i, g in enumerate(gender):
        if g == 'male':
            education_level[i] = np.random.normal(16, 2)  # Higher education for males
        else:
            education_level[i] = np.random.normal(14, 2)  # Lower education for females
    
    # Inject bias: males with higher income more likely to get approved
    gender_bias = np.zeros(n_samples)
    gender_bias[gender == 'male'] = 0.2
    
    # Location bias: urban locations more likely to get approved
    location_bias = np.zeros(n_samples)
    location_bias[location == 'urban'] = 0.1
    location_bias[location == 'rural'] = -0.1
    
    # Create target with bias
    approval_score = (
        0.3 * (age / 50) +
        0.4 * (income / 100000) +
        0.2 * (education_level / 20) +
        gender_bias +
        location_bias +
        0.1 * np.random.random(n_samples)
    )
    
    # Binary outcome
    approved = (approval_score >= 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'education_level': education_level,
        'approved': approved
    })
    
    # Add one-hot encoding for categorical variables
    gender_dummies = pd.get_dummies(gender, prefix='gender')
    location_dummies = pd.get_dummies(location, prefix='location')
    
    data = pd.concat([data, gender_dummies, location_dummies], axis=1)
    
    # Split data
    X = data.drop('approved', axis=1)
    y = data['approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Save protected attributes
    protected_attributes = {
        'gender': gender,
        'location': location
    }
    
    # Split protected attributes
    train_indices = X_train.index
    test_indices = X_test.index
    
    protected_train = {
        'gender': gender[train_indices],
        'location': location[train_indices]
    }
    
    protected_test = {
        'gender': gender[test_indices],
        'location': location[test_indices]
    }
    
    # Train a model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create explainer
    print("Creating fairness explainer...")
    explainer = FairnessExplainer(
        protected_attributes=['gender', 'location'],
        shap_explainer_type='tree',
        output_dir=output_dir
    )
    
    # Generate explanations
    print("Generating explanations...")
    start_time = time.time()
    explanation_results = explainer.explain_model(
        model,
        X_test,
        y_test,
        protected_attributes_data=protected_test
    )
    end_time = time.time()
    
    print(f"Explanation generation took {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_dir}")
    print(f"See {output_dir}/fairness_explanation_report.md for the full report")

if __name__ == "__main__":
    main() 