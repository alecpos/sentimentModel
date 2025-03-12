#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Fairness Visualizations

This script generates multiple synthetic datasets with different bias patterns,
trains models on them, evaluates fairness, and creates visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
os.makedirs('fairness_visualizations', exist_ok=True)
os.makedirs('fairness_visualizations/datasets', exist_ok=True)
os.makedirs('fairness_visualizations/metrics', exist_ok=True)
os.makedirs('fairness_visualizations/group_metrics', exist_ok=True)
os.makedirs('fairness_visualizations/intersectional', exist_ok=True)

# Define colors for visualizations
colors = plt.cm.tab10(np.linspace(0, 1, 10))

def generate_synthetic_data(n_samples=1000, n_features=5, bias_config=None):
    """
    Generate synthetic data with various bias configurations.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        bias_config: Dictionary with bias configuration
            - 'type': Type of bias ('gender', 'age', 'location', 'intersectional')
            - 'strength': Strength of bias (0-1)
            - 'direction': Direction of bias (which group is favored)
            
    Returns:
        data: DataFrame with features and target
        protected_attributes: List of protected attribute names
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create protected attributes based on configuration
    protected_attributes = []
    bias_type = bias_config.get('type', 'gender')
    bias_strength = bias_config.get('strength', 0.5)
    bias_direction = bias_config.get('direction', 1)  # 1 = bias against females/older/etc.
    
    # Create DataFrame with features
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Base score from features (no bias)
    base_score = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    
    # Add bias term (will be modified based on protected attributes)
    bias_term = np.zeros(n_samples)
    
    # Add gender if needed
    if bias_type in ['gender', 'intersectional']:
        gender = np.random.binomial(1, 0.5, size=n_samples)  # 0: male, 1: female
        X_df['gender'] = gender
        protected_attributes.append('gender')
        
        # Add gender bias
        if bias_direction == 1:  # Bias against females
            bias_term -= bias_strength * gender
        else:  # Bias against males
            bias_term -= bias_strength * (1 - gender)
    
    # Add age group if needed
    if bias_type in ['age', 'intersectional']:
        age_raw = np.random.normal(35, 12, size=n_samples)  # Generate ages with mean 35, std 12
        age_raw = np.clip(age_raw, 18, 75)  # Clip to reasonable age range
        
        # Convert to age groups
        age_group = pd.cut(age_raw, bins=[18, 25, 35, 50, 75], 
                          labels=['18-25', '26-35', '36-50', '51+'])
        X_df['age_group'] = age_group
        X_df['age_raw'] = age_raw
        protected_attributes.append('age_group')
        
        # Add age bias - against older people if bias_direction = 1
        if bias_direction == 1:
            bias_term[age_group == '36-50'] -= bias_strength * 0.5
            bias_term[age_group == '51+'] -= bias_strength
        else:  # Against younger people
            bias_term[age_group == '18-25'] -= bias_strength
            bias_term[age_group == '26-35'] -= bias_strength * 0.5
    
    # Add location if needed
    if bias_type in ['location', 'intersectional']:
        location = np.random.choice(['urban', 'suburban', 'rural'], size=n_samples, 
                                   p=[0.5, 0.3, 0.2])
        X_df['location'] = location
        protected_attributes.append('location')
        
        # Add location bias
        if bias_direction == 1:  # Bias against rural
            bias_term[location == 'rural'] -= bias_strength
            bias_term[location == 'suburban'] -= bias_strength * 0.3
        else:  # Bias against urban
            bias_term[location == 'urban'] -= bias_strength
    
    # Calculate final score
    score = base_score + bias_term
    
    # Convert to probability
    prob = 1 / (1 + np.exp(-score))  # Sigmoid
    
    # Generate binary target
    y = np.random.binomial(1, prob)
    
    # Add target to DataFrame
    data = X_df.copy()
    data['target'] = y
    
    return data, protected_attributes

class FairnessVisualizer:
    """Create visualizations for fairness metrics."""
    
    def __init__(self, output_dir='fairness_visualizations'):
        """Initialize the visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_dataset_distributions(self, dataset, protected_attributes, title):
        """
        Plot distributions of protected attributes and outcomes in the dataset.
        
        Args:
            dataset: DataFrame with data
            protected_attributes: List of protected attribute names
            title: Title for the plot
        """
        n_attrs = len(protected_attributes)
        fig, axes = plt.subplots(1, n_attrs, figsize=(n_attrs * 5, 4))
        
        if n_attrs == 1:
            axes = [axes]
        
        for i, attr in enumerate(protected_attributes):
            # Count positive outcomes by group
            group_counts = dataset.groupby([attr, 'target']).size().unstack(fill_value=0)
            
            # Calculate proportions
            total = group_counts.sum(axis=1)
            props = group_counts.div(total, axis=0)
            
            # Plot stacked bar chart
            props.plot(kind='bar', stacked=True, ax=axes[i], 
                      color=['lightcoral', 'cornflowerblue'])
            axes[i].set_title(f'Outcomes by {attr}')
            axes[i].set_ylabel('Proportion')
            axes[i].set_ylim(0, 1)
            axes[i].legend(['Negative (0)', 'Positive (1)'])
            
            # Add count labels
            for j, p in enumerate(props.index):
                total_count = total.loc[p]
                axes[i].text(j, 1.05, f'n={total_count}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/datasets/{title.replace(" ", "_").lower()}.png')
        plt.close()
    
    def plot_fairness_metrics(self, fairness_results, title):
        """
        Plot fairness metrics.
        
        Args:
            fairness_results: Dictionary with fairness evaluation results
            title: Title for the plot
        """
        metrics = fairness_results.get('fairness_metrics', {})
        if not metrics:
            return
        
        # Extract metric values and names
        metric_names = []
        metric_values = []
        thresholds = []
        
        for name, values in metrics.items():
            if 'difference' in values:
                metric_names.append(name)
                metric_values.append(values['difference'])
                thresholds.append(fairness_results.get('fairness_threshold', 0.2))
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, color='skyblue')
        
        # Add threshold line
        if thresholds:
            plt.axhline(y=thresholds[0], color='red', linestyle='--', alpha=0.7, 
                       label=f'Threshold ({thresholds[0]})')
        
        # Highlight bars that exceed threshold
        for i, (bar, value, threshold) in enumerate(zip(bars, metric_values, thresholds)):
            if value > threshold:
                bar.set_color('salmon')
        
        plt.ylabel('Disparity')
        plt.title(f'Fairness Metrics: {title}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{self.output_dir}/metrics/{title.replace(" ", "_").lower()}_metrics.png')
        plt.close()
    
    def plot_group_metrics(self, fairness_results, title):
        """
        Plot metrics by group.
        
        Args:
            fairness_results: Dictionary with fairness evaluation results
            title: Title for the plot
        """
        group_metrics = fairness_results.get('group_metrics', {})
        if not group_metrics:
            return
        
        # Create subplots for each protected attribute
        n_attrs = len(group_metrics)
        fig, axes = plt.subplots(1, n_attrs, figsize=(n_attrs * 6, 5))
        
        if n_attrs == 1:
            axes = [axes]
        
        for i, (attr, metrics) in enumerate(group_metrics.items()):
            # Extract group names and metrics
            groups = list(metrics.keys())
            pos_rates = [metrics[g]['positive_rate'] for g in groups]
            tpr_values = [metrics[g]['true_positive_rate'] for g in groups]
            counts = [metrics[g]['count'] for g in groups]
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Group': groups,
                'Positive Rate': pos_rates,
                'True Positive Rate': tpr_values,
                'Count': counts
            })
            
            # Create grouped bar chart
            ax = axes[i]
            bar_width = 0.35
            x = np.arange(len(groups))
            
            # Plot bars
            ax.bar(x - bar_width/2, df['Positive Rate'], bar_width, label='Positive Rate', color='skyblue')
            ax.bar(x + bar_width/2, df['True Positive Rate'], bar_width, label='True Positive Rate', color='lightgreen')
            
            # Add labels and legend
            ax.set_title(f'Metrics by {attr}')
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.set_ylabel('Rate')
            ax.set_ylim(0, 1)
            ax.legend()
            
            # Add count labels
            for j, count in enumerate(counts):
                ax.text(j, 0.05, f'n={count}', ha='center', color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/group_metrics/{title.replace(" ", "_").lower()}_group_metrics.png')
        plt.close()
    
    def plot_intersectional_heatmap(self, fairness_results, title):
        """
        Plot intersectional fairness heatmap.
        
        Args:
            fairness_results: Dictionary with fairness evaluation results
            title: Title for the plot
        """
        # Check if intersectional results exist
        if 'intersectional' not in fairness_results:
            return
        
        intersectional = fairness_results['intersectional']
        group_metrics = intersectional.get('group_metrics', {})
        
        for intersection, metrics in group_metrics.items():
            # Only process 2-attribute intersections for heatmaps
            if len(intersection) != 2:
                continue
            
            # Get attribute names
            attr1, attr2 = intersection
            
            # Extract unique values from the group keys
            groups = list(metrics.keys())
            
            if not groups:  # Skip if no groups
                continue
                
            # Group keys should be in the format 'value1_value2'
            # First, extract all unique values for attr1 and attr2
            all_values = []
            for group_key in groups:
                parts = group_key.split('_')
                if len(parts) == 2:
                    all_values.append(parts)
            
            if not all_values:  # Skip if parsing failed
                continue
                
            # Extract unique values for each attribute
            values1 = sorted(set(parts[0] for parts in all_values))
            values2 = sorted(set(parts[1] for parts in all_values))
            
            # Create data structure for heatmap
            heatmap_data = np.zeros((len(values1), len(values2)))
            
            # Fill in heatmap data
            for val1_idx, val1 in enumerate(values1):
                for val2_idx, val2 in enumerate(values2):
                    group_key = f"{val1}_{val2}"
                    if group_key in metrics:
                        heatmap_data[val1_idx, val2_idx] = metrics[group_key]['positive_rate']
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r',
                       xticklabels=values2, yticklabels=values1,
                       vmin=0, vmax=1, center=0.5)
            
            plt.title(f'Positive Rate by {attr1} and {attr2}\n{title}')
            plt.xlabel(attr2)
            plt.ylabel(attr1)
            plt.tight_layout()
            
            # Save heatmap
            intersection_name = f"{attr1}_{attr2}"
            plt.savefig(f'{self.output_dir}/intersectional/{title.replace(" ", "_").lower()}_{intersection_name}_heatmap.png')
            plt.close()

def evaluate_fairness(X_test, y_test, y_pred, protected_attributes, fairness_threshold=0.1):
    """
    Evaluate fairness for the model predictions.
    
    Args:
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels
        protected_attributes: List of protected attribute names
        fairness_threshold: Threshold for fairness violations
        
    Returns:
        Dictionary with fairness evaluation results
    """
    # Initialize results
    results = {
        "overall": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "positive_rate": float(np.mean(y_pred)),
            "true_positive_rate": float(np.mean(y_pred[y_test == 1]) if np.any(y_test == 1) else 0),
            "false_positive_rate": float(np.mean(y_pred[y_test == 0]) if np.any(y_test == 0) else 0)
        },
        "group_metrics": {},
        "fairness_metrics": {},
        "fairness_threshold": fairness_threshold
    }
    
    # Calculate metrics for each protected attribute
    for attr in protected_attributes:
        results["group_metrics"][attr] = {}
        
        # Get unique values for this attribute
        unique_values = X_test[attr].unique()
        
        for value in unique_values:
            # Get indices for this group
            mask = (X_test[attr] == value)
            if not np.any(mask):
                continue
                
            # Get predictions and labels for this group
            group_preds = y_pred[mask]
            group_labels = y_test[mask]
            
            # Calculate metrics
            group_accuracy = accuracy_score(group_labels, group_preds)
            group_positive_rate = np.mean(group_preds)
            
            # Calculate true positive rate
            group_positives = (group_labels == 1)
            if np.any(group_positives):
                true_positive_rate = np.mean(group_preds[group_positives])
            else:
                true_positive_rate = 0
                
            # Calculate false positive rate
            group_negatives = (group_labels == 0)
            if np.any(group_negatives):
                false_positive_rate = np.mean(group_preds[group_negatives])
            else:
                false_positive_rate = 0
            
            # Store metrics
            results["group_metrics"][attr][value] = {
                "count": int(np.sum(mask)),
                "accuracy": float(group_accuracy),
                "positive_rate": float(group_positive_rate),
                "true_positive_rate": float(true_positive_rate),
                "false_positive_rate": float(false_positive_rate)
            }
        
        # Calculate fairness metrics
        pos_rates = [m["positive_rate"] for m in results["group_metrics"][attr].values()]
        tpr_values = [m["true_positive_rate"] for m in results["group_metrics"][attr].values()]
        
        dp_difference = max(pos_rates) - min(pos_rates)
        eo_difference = max(tpr_values) - min(tpr_values)
        
        results["fairness_metrics"][f"{attr}_demographic_parity"] = {
            "difference": float(dp_difference),
            "passes_threshold": dp_difference <= fairness_threshold
        }
        
        results["fairness_metrics"][f"{attr}_equal_opportunity"] = {
            "difference": float(eo_difference),
            "passes_threshold": eo_difference <= fairness_threshold
        }
    
    # Calculate intersectional metrics if we have multiple protected attributes
    if len(protected_attributes) > 1:
        results["intersectional"] = {
            "group_metrics": {},
            "fairness_metrics": {}
        }
        
        # Only do pairs for simplicity
        from itertools import combinations
        for attr1, attr2 in combinations(protected_attributes, 2):
            intersection = (attr1, attr2)
            results["intersectional"]["group_metrics"][intersection] = {}
            
            # Get unique values for each attribute
            values1 = X_test[attr1].unique()
            values2 = X_test[attr2].unique()
            
            # Calculate metrics for each combination
            for val1 in values1:
                for val2 in values2:
                    # Create combined mask
                    mask = (X_test[attr1] == val1) & (X_test[attr2] == val2)
                    if not np.any(mask):
                        continue
                    
                    # Get group data
                    group_preds = y_pred[mask]
                    group_labels = y_test[mask]
                    
                    # Calculate metrics
                    group_positive_rate = np.mean(group_preds)
                    group_accuracy = accuracy_score(group_labels, group_preds)
                    
                    # Calculate TPR and FPR
                    group_positives = (group_labels == 1)
                    true_positive_rate = np.mean(group_preds[group_positives]) if np.any(group_positives) else 0
                    
                    group_negatives = (group_labels == 0)
                    false_positive_rate = np.mean(group_preds[group_negatives]) if np.any(group_negatives) else 0
                    
                    # Store metrics
                    group_key = f"{val1}_{val2}"
                    results["intersectional"]["group_metrics"][intersection][group_key] = {
                        "count": int(np.sum(mask)),
                        "positive_rate": float(group_positive_rate),
                        "accuracy": float(group_accuracy),
                        "true_positive_rate": float(true_positive_rate),
                        "false_positive_rate": float(false_positive_rate)
                    }
            
            # Calculate fairness metrics for this intersection
            group_metrics = results["intersectional"]["group_metrics"][intersection]
            if len(group_metrics) < 2:
                continue
                
            pos_rates = [m["positive_rate"] for m in group_metrics.values()]
            tpr_values = [m["true_positive_rate"] for m in group_metrics.values()]
            
            max_diff_dp = max(pos_rates) - min(pos_rates)
            max_diff_eo = max(tpr_values) - min(tpr_values)
            
            metric_key_dp = f"{attr1}+{attr2}_demographic_parity"
            metric_key_eo = f"{attr1}+{attr2}_equal_opportunity"
            
            results["intersectional"]["fairness_metrics"][metric_key_dp] = {
                "difference": float(max_diff_dp),
                "passes_threshold": max_diff_dp <= fairness_threshold
            }
            
            results["intersectional"]["fairness_metrics"][metric_key_eo] = {
                "difference": float(max_diff_eo),
                "passes_threshold": max_diff_eo <= fairness_threshold
            }
    
    return results

def run_experiment(dataset_config, fairness_threshold=0.1):
    """
    Run a fairness experiment with the given configuration.
    
    Args:
        dataset_config: Configuration for synthetic data generation
        fairness_threshold: Threshold for fairness violations
        
    Returns:
        Dictionary with experiment results
    """
    # Generate dataset
    data, protected_attributes = generate_synthetic_data(
        n_samples=dataset_config.get('n_samples', 2000),
        n_features=dataset_config.get('n_features', 5),
        bias_config=dataset_config.get('bias_config', {})
    )
    
    # Create a title for this experiment
    bias_type = dataset_config['bias_config']['type']
    bias_strength = dataset_config['bias_config']['strength']
    bias_direction = dataset_config['bias_config']['direction']
    
    direction_label = "Against " + {
        'gender': 'Females' if bias_direction == 1 else 'Males',
        'age': 'Older' if bias_direction == 1 else 'Younger',
        'location': 'Rural' if bias_direction == 1 else 'Urban',
        'intersectional': 'Multiple Groups' if bias_direction == 1 else 'Multiple Groups (Reversed)'
    }[bias_type]
    
    strength_label = {
        0.0: 'No',
        0.2: 'Very Weak',
        0.5: 'Moderate',
        0.8: 'Strong',
        1.2: 'Very Strong'
    }[bias_strength]
    
    title = f"{bias_type.title()} Bias ({strength_label}) {direction_label}"
    
    # Split the data
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create visualizer
    visualizer = FairnessVisualizer()
    
    # Plot dataset distributions
    visualizer.plot_dataset_distributions(data, protected_attributes, title)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Get feature columns excluding protected attributes
    feature_cols = [col for col in X_train.columns if col not in protected_attributes 
                   and col != 'age_raw']  # Exclude age_raw if present
    
    # Train the model
    model.fit(X_train[feature_cols], y_train)
    
    # Make predictions
    y_pred = model.predict(X_test[feature_cols])
    
    # Evaluate fairness
    fairness_results = evaluate_fairness(
        X_test, y_test, y_pred, 
        protected_attributes, 
        fairness_threshold=fairness_threshold
    )
    
    # Create visualizations
    visualizer.plot_fairness_metrics(fairness_results, title)
    visualizer.plot_group_metrics(fairness_results, title)
    visualizer.plot_intersectional_heatmap(fairness_results, title)
    
    return {
        'title': title,
        'accuracy': accuracy_score(y_test, y_pred),
        'fairness_results': fairness_results,
        'config': dataset_config
    }

def main():
    """Run various fairness experiments and generate visualizations."""
    print("Generating fairness visualizations for different bias patterns...")
    
    # Define dataset configurations
    dataset_configs = [
        # No bias baseline
        {
            'n_samples': 2000,
            'bias_config': {
                'type': 'gender',
                'strength': 0.0,
                'direction': 1
            }
        },
        # Gender bias (binary)
        {
            'n_samples': 2000,
            'bias_config': {
                'type': 'gender',
                'strength': 0.8,
                'direction': 1  # Against females
            }
        },
        {
            'n_samples': 2000,
            'bias_config': {
                'type': 'gender',
                'strength': 0.8,
                'direction': 0  # Against males
            }
        },
        # Age bias (multi-category)
        {
            'n_samples': 2000,
            'bias_config': {
                'type': 'age',
                'strength': 0.8,
                'direction': 1  # Against older
            }
        },
        {
            'n_samples': 2000,
            'bias_config': {
                'type': 'age',
                'strength': 0.5,
                'direction': 0  # Against younger (moderate)
            }
        },
        # Location bias
        {
            'n_samples': 2000,
            'bias_config': {
                'type': 'location',
                'strength': 0.8,
                'direction': 1  # Against rural
            }
        },
        # Intersectional bias
        {
            'n_samples': 3000,
            'bias_config': {
                'type': 'intersectional',
                'strength': 1.2,
                'direction': 1  # Against females & older & rural
            }
        }
    ]
    
    # Run experiments
    results = []
    for config in dataset_configs:
        print(f"Running experiment: {config['bias_config']['type']} bias with strength {config['bias_config']['strength']}")
        result = run_experiment(config, fairness_threshold=0.1)
        results.append(result)
        print(f"  - Model accuracy: {result['accuracy']:.4f}")
        
        # Print fairness metrics
        fairness_metrics = result['fairness_results']['fairness_metrics']
        for metric, values in fairness_metrics.items():
            passes = "PASS" if values.get('passes_threshold', False) else "FAIL"
            print(f"  - {metric}: {values['difference']:.4f} ({passes})")
        print()
    
    print("\nAll visualizations have been generated in the 'fairness_visualizations' directory.")
    print("\nDirectory structure:")
    print(" - fairness_visualizations/datasets/: Dataset distributions")
    print(" - fairness_visualizations/metrics/: Fairness metrics")
    print(" - fairness_visualizations/group_metrics/: Metrics by group")
    print(" - fairness_visualizations/intersectional/: Intersectional analysis")

if __name__ == "__main__":
    main() 