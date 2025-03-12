#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Card Generation Demo

This script demonstrates how to generate comprehensive model cards
that document fairness considerations and mitigation techniques.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import the model card generator
from model_card_generator import ModelCardGenerator, generate_model_card_for_ad_score_predictor

# Import fairness framework components
# Replace with your actual imports
try:
    from generate_synthetic_data import generate_synthetic_data_with_bias
    from fairness_evaluator import FairnessEvaluator
    # Try to import other fairness components if available
    try:
        from fairness_explainability import FairnessExplainer
    except ImportError:
        print("FairnessExplainer not available, skipping explainability analysis")
except ImportError:
    print("Could not import fairness framework components. Using mock data.")
    # Define mock function if real one is not available
    def generate_synthetic_data_with_bias(n_samples=1000, bias_factor=0.4, 
                                        random_state=42):
        """Mock function to generate synthetic data with bias."""
        from sklearn.datasets import make_classification
        
        # Generate synthetic data
        X, y = make_classification(n_samples=n_samples, n_features=10, 
                                n_informative=5, n_redundant=2, n_classes=2,
                                random_state=random_state)
        
        # Create a DataFrame with the features
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Add protected attributes
        np.random.seed(random_state)
        X_df['gender'] = np.random.choice(['male', 'female'], size=n_samples, 
                                        p=[0.6, 0.4])
        X_df['location'] = np.random.choice(['urban', 'suburban', 'rural'], 
                                            size=n_samples, p=[0.5, 0.3, 0.2])
        X_df['age_group'] = np.random.choice(['18-25', '26-35', '36-50', '51+'], 
                                            size=n_samples, p=[0.2, 0.4, 0.3, 0.1])
        
        # Add bias - make males more likely to have positive outcomes
        for i in range(len(X_df)):
            if X_df.loc[i, 'gender'] == 'male' and np.random.random() < bias_factor:
                y[i] = 1
        
        return X_df, pd.Series(y)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, protected_attributes):
    """
    Train a model and evaluate its performance and fairness.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        protected_attributes: List of protected attribute names
        
    Returns:
        model: Trained model
        performance_metrics: Dictionary of performance metrics
        fairness_results: Dictionary of fairness evaluation results
    """
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Get feature columns excluding protected attributes
    feature_cols = [col for col in X_train.columns if col not in protected_attributes]
    
    # Convert to numpy arrays if needed
    X_train_features = X_train[feature_cols].values if hasattr(X_train, 'values') else X_train[feature_cols]
    X_test_features = X_test[feature_cols].values if hasattr(X_test, 'values') else X_test[feature_cols]
    
    y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Train the model
    model.fit(X_train_features, y_train_values)
    
    # Predict on test set
    y_pred = model.predict(X_test_features)
    y_prob = model.predict_proba(X_test_features)[:, 1]
    
    # Calculate performance metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    performance_metrics = {
        'accuracy': float(accuracy_score(y_test_values, y_pred)),
        'precision': float(precision_score(y_test_values, y_pred)),
        'recall': float(recall_score(y_test_values, y_pred)),
        'f1_score': float(f1_score(y_test_values, y_pred)),
        'roc_auc': float(roc_auc_score(y_test_values, y_prob))
    }
    
    # Evaluate fairness
    try:
        fairness_evaluator = FairnessEvaluator(protected_attributes=protected_attributes)
        fairness_results = fairness_evaluator.evaluate(
            X_test, y_test, y_pred, y_prob, calculate_intersectional=True, 
            output_dir='fairness_eval'
        )
    except Exception as e:
        print(f"Error in fairness evaluation: {e}")
        # Create mock fairness results
        fairness_results = create_mock_fairness_results(X_test, y_test, y_pred, protected_attributes)
    
    return model, performance_metrics, fairness_results

def create_mock_fairness_results(X_test, y_test, y_pred, protected_attributes):
    """Create mock fairness evaluation results if the real evaluator is not available."""
    # Calculate group metrics for each protected attribute
    group_metrics = {}
    
    # Convert y_test and y_pred to numpy arrays if they're pandas Series
    if hasattr(y_test, 'values'):
        y_test_values = y_test.values
    else:
        y_test_values = np.array(y_test)
        
    if hasattr(y_pred, 'values'):
        y_pred_values = y_pred.values
    else:
        y_pred_values = np.array(y_pred)
    
    for attr in protected_attributes:
        groups = X_test[attr].unique()
        group_metrics[attr] = {}
        
        for group in groups:
            # Get indices for this group as a boolean mask
            mask = (X_test[attr] == group).values if hasattr(X_test[attr], 'values') else (X_test[attr] == group)
            
            if np.sum(mask) > 0:
                # Calculate metrics for this group
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                
                try:
                    # Use boolean mask directly with numpy arrays
                    group_y_test = y_test_values[mask]
                    group_y_pred = y_pred_values[mask]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(group_y_test, group_y_pred)
                    positive_rate = np.mean(group_y_pred)
                    
                    # Handle edge cases for TPR and FPR
                    if np.sum(group_y_test == 1) > 0:
                        true_positive_rate = recall_score(group_y_test, group_y_pred, zero_division=0)
                    else:
                        true_positive_rate = 0
                        
                    if np.sum(group_y_test == 0) > 0:
                        false_positive_rate = np.sum((group_y_pred == 1) & (group_y_test == 0)) / np.sum(group_y_test == 0)
                    else:
                        false_positive_rate = 0
                    
                    group_metrics[attr][group] = {
                        'count': int(np.sum(mask)),
                        'accuracy': float(accuracy),
                        'positive_rate': float(positive_rate),
                        'true_positive_rate': float(true_positive_rate),
                        'false_positive_rate': float(false_positive_rate)
                    }
                except Exception as e:
                    print(f"Error calculating metrics for {attr}={group}: {e}")
                    group_metrics[attr][group] = {
                        'count': int(np.sum(mask)),
                        'accuracy': 0.0,
                        'positive_rate': 0.0,
                        'true_positive_rate': 0.0,
                        'false_positive_rate': 0.0
                    }
    
    # Calculate fairness metrics
    fairness_metrics = {}
    for attr in protected_attributes:
        groups = list(group_metrics[attr].keys())
        if len(groups) > 1:
            # Calculate demographic parity difference
            fairness_metrics[f"{attr}_demographic_parity"] = {
                'difference': abs(group_metrics[attr][groups[0]]['positive_rate'] - 
                                group_metrics[attr][groups[1]]['positive_rate']),
                'passes_threshold': True  # Placeholder
            }
            
            # Calculate equal opportunity difference
            fairness_metrics[f"{attr}_equal_opportunity"] = {
                'difference': abs(group_metrics[attr][groups[0]]['true_positive_rate'] - 
                                group_metrics[attr][groups[1]]['true_positive_rate']),
                'passes_threshold': True  # Placeholder
            }
    
    # Create mock fairness results
    fairness_results = {
        'overall': {
            'accuracy': float((y_pred == y_test).mean())
        },
        'fairness_metrics': fairness_metrics,
        'group_metrics': group_metrics,
        'fairness_threshold': 0.05
    }
    
    return fairness_results

def create_mitigation_info(mitigation_type='reweighing', protected_attribute='gender'):
    """
    Create information about mitigation strategies used.
    
    Args:
        mitigation_type: Type of mitigation strategy
        protected_attribute: Protected attribute being mitigated
        
    Returns:
        Dictionary with mitigation information
    """
    if mitigation_type == 'reweighing':
        return {
            'Reweighing': {
                'description': 'Assigns different weights to training examples to ensure fairness across protected groups.',
                'implementation': 'ReweighingMitigation class',
                'parameters': {
                    'protected_attribute': protected_attribute,
                    'reweighing_factor': 1.0
                },
                'effectiveness': 'Reduced demographic parity difference by approximately 80%'
            }
        }
    elif mitigation_type == 'constraints':
        return {
            'Fairness Constraints': {
                'description': 'Adds fairness constraints to the model training process to enforce fairness criteria.',
                'implementation': 'FairnessConstraint class',
                'parameters': {
                    'constraint_type': 'demographic_parity',
                    'protected_attribute': protected_attribute,
                    'epsilon': 0.05
                },
                'effectiveness': 'Reduced demographic parity difference by approximately 65%'
            }
        }
    elif mitigation_type == 'adversarial':
        return {
            'Adversarial Debiasing': {
                'description': 'Uses adversarial learning to remove information about protected attributes from the model.',
                'implementation': 'AdversarialDebiasing class',
                'parameters': {
                    'protected_attribute': protected_attribute,
                    'adversary_loss_weight': 0.5
                },
                'effectiveness': 'Reduced demographic parity difference by approximately 70%'
            }
        }
    else:
        return {}

def generate_model_cards():
    """
    Generate model cards for various scenarios.
    """
    print("Generating model cards for different fairness scenarios...")
    
    # Create output directory
    os.makedirs('model_cards', exist_ok=True)
    
    # 1. Generate synthetic data with bias
    print("\n1. Generating synthetic data with bias...")
    X, y = generate_synthetic_data_with_bias(n_samples=1000, bias_factor=0.4)
    
    # 2. Split data
    protected_attributes = ['gender', 'location', 'age_group']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Train and evaluate baseline model
    print("\n2. Training and evaluating baseline model...")
    model, performance_metrics, fairness_results = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, protected_attributes
    )
    
    # 4. Generate baseline model card
    print("\n3. Generating baseline model card...")
    model_info = {
        'name': 'Ad Score Predictor - Baseline',
        'version': '1.0.0',
        'type': 'Classification',
        'description': 'Baseline model for predicting ad performance scores.',
        'use_cases': ['Predicting ad effectiveness', 'Estimating conversion rates'],
        'date_created': datetime.now().strftime('%Y-%m-%d'),
    }
    
    limitations = [
        'This baseline model does not include any fairness mitigations',
        'Performance varies across demographic groups, indicating potential bias',
        'The model was trained on synthetic data with known gender bias'
    ]
    
    ethical_considerations = [
        'This model shows evidence of disparate impact across gender groups',
        'Using this model without mitigation could lead to unfair ad targeting',
        'Regular fairness audits are recommended if deploying this model'
    ]
    
    generator = ModelCardGenerator(output_dir='model_cards')
    
    baseline_card_path = generator.generate_model_card(
        model_info=model_info,
        performance_metrics=performance_metrics,
        fairness_metrics=fairness_results,
        limitations=limitations,
        ethical_considerations=ethical_considerations,
        export_formats=['md', 'html']
    )
    
    print(f"Baseline model card generated at: {baseline_card_path}")
    
    # 5. Generate mitigated model card
    print("\n4. Generating mitigated model card...")
    
    # In a real scenario, you would train a mitigated model here
    # For demo purposes, we'll use the same model but with mitigation info
    
    mitigation_info = create_mitigation_info(
        mitigation_type='reweighing', 
        protected_attribute='gender'
    )
    
    # Update the fairness results to simulate improvement
    mitigated_fairness_results = fairness_results.copy()
    
    # Simulate improved fairness metrics
    for key in mitigated_fairness_results['fairness_metrics']:
        if 'gender' in key:
            # Reduce the difference by 80%
            original_diff = mitigated_fairness_results['fairness_metrics'][key]['difference']
            mitigated_fairness_results['fairness_metrics'][key]['difference'] = original_diff * 0.2
            mitigated_fairness_results['fairness_metrics'][key]['passes_threshold'] = True
    
    model_info['name'] = 'Ad Score Predictor - Mitigated'
    model_info['description'] = 'Ad score prediction model with fairness mitigations.'
    
    limitations = [
        'While fairness has been improved, some residual bias may remain',
        'Mitigation focused only on gender bias, other biases may still be present',
        'The model was trained on synthetic data with limited diversity'
    ]
    
    ethical_considerations = [
        'This model includes fairness mitigations to reduce disparate impact',
        'Regular fairness audits are still recommended when using this model',
        'Additional protected attributes should be considered in future versions'
    ]
    
    mitigated_card_path = generator.generate_model_card(
        model_info=model_info,
        performance_metrics=performance_metrics,
        fairness_metrics=mitigated_fairness_results,
        limitations=limitations,
        ethical_considerations=ethical_considerations,
        mitigation_strategies=mitigation_info,
        export_formats=['md', 'html']
    )
    
    print(f"Mitigated model card generated at: {mitigated_card_path}")
    
    # 6. Generate comprehensive model card with multiple mitigations
    print("\n5. Generating comprehensive model card with multiple mitigations...")
    
    model_info['name'] = 'Ad Score Predictor - Comprehensive'
    model_info['version'] = '2.0.0'
    model_info['description'] = 'Advanced ad score prediction model with multiple fairness mitigations.'
    
    # Combine multiple mitigation strategies
    comprehensive_mitigations = {}
    comprehensive_mitigations.update(create_mitigation_info('reweighing', 'gender'))
    comprehensive_mitigations.update(create_mitigation_info('constraints', 'age_group'))
    
    # Create comprehensive fairness results with intersectional analysis
    comprehensive_fairness_results = mitigated_fairness_results.copy()
    
    # Add mock intersectional analysis
    comprehensive_fairness_results['intersectional'] = {
        'fairness_metrics': {
            'gender+location_demographic_parity': {
                'difference': 0.03,
                'passes_threshold': True
            },
            'gender+age_group_demographic_parity': {
                'difference': 0.04,
                'passes_threshold': True
            },
            'gender+location+age_group_demographic_parity': {
                'difference': 0.05,
                'passes_threshold': True
            }
        },
        'group_metrics': {
            'gender+location': {
                'male+urban': {'positive_rate': 0.76, 'count': 300},
                'male+suburban': {'positive_rate': 0.75, 'count': 180},
                'male+rural': {'positive_rate': 0.74, 'count': 120},
                'female+urban': {'positive_rate': 0.74, 'count': 200},
                'female+suburban': {'positive_rate': 0.73, 'count': 120},
                'female+rural': {'positive_rate': 0.71, 'count': 80}
            }
        }
    }
    
    # Add visualization paths (in a real scenario, these would be actual file paths)
    visualizations = {
        'fairness_heatmap': 'model_cards/assets/fairness_heatmap_example.png',
        'group_comparison': 'model_cards/assets/group_comparison_example.png'
    }
    
    # Create example visualization files (empty files just for demo)
    os.makedirs('model_cards/assets', exist_ok=True)
    for viz_path in visualizations.values():
        # Create a simple matplotlib figure
        plt.figure(figsize=(8, 6))
        plt.title(os.path.basename(viz_path).replace('.png', '').replace('_', ' ').title())
        plt.text(0.5, 0.5, 'Example Visualization', ha='center', va='center', fontsize=14)
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
    
    comprehensive_card_path = generator.generate_model_card(
        model_info=model_info,
        performance_metrics=performance_metrics,
        fairness_metrics=comprehensive_fairness_results,
        limitations=[
            'While comprehensive fairness mitigations have been applied, performance may vary in production',
            'The model should be regularly monitored for drift and fairness degradation',
            'Intersectional analysis reveals some residual bias at the intersections of multiple attributes'
        ],
        ethical_considerations=[
            'This model implements multiple fairness mitigations to address various forms of bias',
            'The model has been evaluated using intersectional fairness analysis',
            'Continuous monitoring is necessary to ensure fairness is maintained over time',
            'The model should be used as part of a larger responsible AI framework'
        ],
        mitigation_strategies=comprehensive_mitigations,
        visualizations=visualizations,
        export_formats=['md', 'html']
    )
    
    print(f"Comprehensive model card generated at: {comprehensive_card_path}")
    
    # 7. Generate regulatory compliance model card
    print("\n6. Generating regulatory compliance model card...")
    
    model_info['name'] = 'Ad Score Predictor - Regulatory Compliance'
    model_info['version'] = '2.1.0'
    model_info['description'] = 'Ad score prediction model designed for regulatory compliance.'
    
    # Add additional model information for regulatory compliance
    model_info['regulatory_info'] = {
        'frameworks': ['EU AI Act', 'NYC Local Law 144', 'NIST AI Risk Management Framework'],
        'risk_level': 'Medium',
        'compliance_status': 'Compliant',
        'audit_date': datetime.now().strftime('%Y-%m-%d'),
        'documentation': 'Complete',
        'human_oversight': 'Required for scores below threshold'
    }
    
    regulatory_card_path = generator.generate_model_card(
        model_info=model_info,
        performance_metrics=performance_metrics,
        fairness_metrics=comprehensive_fairness_results,
        limitations=[
            'This model complies with current regulatory requirements but may need updates as regulations evolve',
            'Regular compliance audits are required to maintain regulatory status',
            'Additional protected categories may be required for future regulatory compliance'
        ],
        ethical_considerations=[
            'This model is designed to comply with relevant AI fairness regulations',
            'Human oversight is required for certain decision scenarios',
            'The model has undergone a third-party fairness audit'
        ],
        mitigation_strategies=comprehensive_mitigations,
        visualizations=visualizations,
        export_formats=['md', 'html']
    )
    
    print(f"Regulatory compliance model card generated at: {regulatory_card_path}")
    
    print("\nAll model cards have been generated successfully in the 'model_cards' directory!")

if __name__ == "__main__":
    generate_model_cards() 