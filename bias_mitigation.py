#!/usr/bin/env python
"""
Bias Mitigation Module for Sentiment Analysis

This module provides techniques for mitigating bias in sentiment analysis models:
1. Data preprocessing techniques to reduce bias
2. Training techniques that incorporate fairness constraints
3. Post-processing methods to adjust predictions for fairness

Usage:
    from bias_mitigation import reweight_training_data, calibrate_predictions
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reweight_training_data(df, prediction_col, protected_cols, label_col=None):
    """
    Reweight the training data to mitigate bias across protected groups.
    
    Args:
        df: DataFrame with training data
        prediction_col: Column with model predictions
        protected_cols: List of columns with protected attributes
        label_col: Column with ground truth labels (if available)
    
    Returns:
        DataFrame with sample weights
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Initialize sample weights to 1.0
    result_df['sample_weight'] = 1.0
    
    # Calculate overall positive rate
    overall_positive_rate = df[prediction_col].mean()
    
    # Calculate weights for each combination of protected attributes
    # Create a combined column for intersection
    intersection_col = '_'.join(protected_cols)
    result_df[intersection_col] = result_df[protected_cols[0]].astype(str)
    for col in protected_cols[1:]:
        result_df[intersection_col] += '_' + result_df[col].astype(str)
    
    # Calculate positive rate for each group
    group_positive_rates = result_df.groupby(intersection_col)[prediction_col].mean()
    
    # Calculate weights
    for group, positive_rate in group_positive_rates.items():
        if positive_rate > 0:  # Avoid division by zero
            weight = overall_positive_rate / positive_rate
            result_df.loc[result_df[intersection_col] == group, 'sample_weight'] = weight
    
    # Adjust weights based on ground truth if available
    if label_col is not None:
        # Calculate conditional weights to ensure equal opportunity
        # For each group, balance the weights for true positives and true negatives
        for group in result_df[intersection_col].unique():
            group_mask = result_df[intersection_col] == group
            group_df = result_df[group_mask]
            
            # True positive rate for this group
            group_positives = group_df[group_df[label_col] == 1]
            if len(group_positives) > 0:
                group_tpr = group_positives[prediction_col].mean()
                
                # True negative rate for this group
                group_negatives = group_df[group_df[label_col] == 0]
                if len(group_negatives) > 0:
                    group_tnr = 1 - group_negatives[prediction_col].mean()
                    
                    # Calculate overall TPR and TNR
                    overall_tpr = df[df[label_col] == 1][prediction_col].mean()
                    overall_tnr = 1 - df[df[label_col] == 0][prediction_col].mean()
                    
                    # Adjust weights for positive and negative examples
                    if group_tpr > 0 and group_tnr > 0:
                        tp_weight = overall_tpr / group_tpr
                        tn_weight = overall_tnr / group_tnr
                        
                        # Apply adjusted weights
                        result_df.loc[(group_mask) & (df[label_col] == 1), 'sample_weight'] *= tp_weight
                        result_df.loc[(group_mask) & (df[label_col] == 0), 'sample_weight'] *= tn_weight
    
    # Normalize weights to have a mean of 1.0
    result_df['sample_weight'] = result_df['sample_weight'] / result_df['sample_weight'].mean()
    
    logger.info(f"Reweighted data with weight range: [{result_df['sample_weight'].min():.4f}, {result_df['sample_weight'].max():.4f}]")
    
    return result_df

def balance_dataset(df, protected_cols, label_col, method='upsample'):
    """
    Balance the dataset across protected groups using resampling.
    
    Args:
        df: DataFrame with training data
        protected_cols: List of columns with protected attributes
        label_col: Column with labels
        method: Resampling method ('upsample' or 'downsample')
    
    Returns:
        Balanced DataFrame
    """
    # Create a combined column for intersection
    intersection_col = '_'.join(protected_cols)
    df[intersection_col] = df[protected_cols[0]].astype(str)
    for col in protected_cols[1:]:
        df[intersection_col] += '_' + df[col].astype(str)
    
    # Get counts for each intersection and label
    group_counts = df.groupby([intersection_col, label_col]).size().reset_index(name='count')
    
    # Find the target size for each group
    if method == 'upsample':
        # Upsample all groups to match the largest group
        target_sizes = group_counts.groupby(label_col)['count'].max().to_dict()
    else:  # 'downsample'
        # Downsample all groups to match the smallest group
        target_sizes = group_counts.groupby(label_col)['count'].min().to_dict()
    
    # Resample each group to the target size
    balanced_dfs = []
    for (group, label), count in group_counts.set_index([intersection_col, label_col])['count'].items():
        group_df = df[(df[intersection_col] == group) & (df[label_col] == label)]
        target_size = target_sizes[label]
        
        if method == 'upsample' and count < target_size:
            # Upsample
            resampled_df = resample(
                group_df,
                replace=True,
                n_samples=target_size,
                random_state=42
            )
            balanced_dfs.append(resampled_df)
        elif method == 'downsample' and count > target_size:
            # Downsample
            resampled_df = resample(
                group_df,
                replace=False,
                n_samples=target_size,
                random_state=42
            )
            balanced_dfs.append(resampled_df)
        else:
            # No resampling needed
            balanced_dfs.append(group_df)
    
    # Combine all resampled groups
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Log resampling results
    orig_counts = df.groupby([intersection_col, label_col]).size()
    new_counts = balanced_df.groupby([intersection_col, label_col]).size()
    
    logger.info(f"Original dataset size: {len(df)}")
    logger.info(f"Balanced dataset size: {len(balanced_df)}")
    logger.info(f"Resampling method: {method}")
    
    # Check if resampling was effective
    orig_std = orig_counts.groupby(level=1).std().mean()
    new_std = new_counts.groupby(level=1).std().mean()
    logger.info(f"Standard deviation of group sizes before: {orig_std:.2f}, after: {new_std:.2f}")
    
    return balanced_df

def counterfactual_data_augmentation(df, text_col, protected_terms_map):
    """
    Perform counterfactual data augmentation by replacing protected terms.
    
    Args:
        df: DataFrame with text data
        text_col: Column containing text
        protected_terms_map: Dictionary mapping protected terms to their replacements
            Format: {protected_group: {original_term: replacement_term, ...}, ...}
    
    Returns:
        Augmented DataFrame
    """
    # Start with the original data
    augmented_df = df.copy()
    
    # For each protected group, create counterfactual examples
    for group, term_map in protected_terms_map.items():
        # Create a copy of the DataFrame to modify
        modified_df = df.copy()
        
        # Replace terms in the text
        for original, replacement in term_map.items():
            # Create a regex pattern to match the term as a whole word
            pattern = r'\b' + original + r'\b'
            modified_df[text_col] = modified_df[text_col].str.replace(
                pattern, replacement, regex=True
            )
        
        # Add the modified rows to the augmented DataFrame
        augmented_df = pd.concat([augmented_df, modified_df], ignore_index=True)
    
    logger.info(f"Original dataset size: {len(df)}")
    logger.info(f"Augmented dataset size: {len(augmented_df)}")
    logger.info(f"Created {len(augmented_df) - len(df)} counterfactual examples")
    
    return augmented_df

def calibrate_predictions(predictions, protected_attributes, calibration_type='equal_odds'):
    """
    Post-process predictions to satisfy fairness constraints.
    
    Args:
        predictions: Array of prediction probabilities
        protected_attributes: DataFrame with protected attributes
        calibration_type: Type of calibration to apply ('equal_odds', 'demographic_parity')
    
    Returns:
        Calibrated predictions
    """
    # Convert predictions to numpy array if needed
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Get unique protected groups
    if isinstance(protected_attributes, pd.DataFrame):
        # If multiple protected attributes, combine them
        if protected_attributes.shape[1] > 1:
            # Create a combined attribute
            combined = protected_attributes.iloc[:, 0].astype(str)
            for i in range(1, protected_attributes.shape[1]):
                combined += '_' + protected_attributes.iloc[:, i].astype(str)
            groups = combined.unique()
            group_indices = {group: (combined == group) for group in groups}
        else:
            # Single protected attribute
            groups = protected_attributes.iloc[:, 0].unique()
            group_indices = {group: (protected_attributes.iloc[:, 0] == group) for group in groups}
    else:
        # Single numpy array
        groups = np.unique(protected_attributes)
        group_indices = {group: (protected_attributes == group) for group in groups}
    
    # Calculate mean prediction for each group
    group_means = {group: predictions[indices].mean() for group, indices in group_indices.items()}
    
    # Calculate overall mean
    overall_mean = predictions.mean()
    
    # Calibrate predictions based on fairness type
    calibrated = np.copy(predictions)
    
    if calibration_type == 'demographic_parity':
        # Calibrate for demographic parity (equal positive rate)
        # For each group, shift predictions to match overall mean
        for group, indices in group_indices.items():
            if group_means[group] > 0:  # Avoid division by zero
                calibrated[indices] = predictions[indices] * (overall_mean / group_means[group])
    
    elif calibration_type == 'equal_odds':
        # Simple approximation of equal odds calibration
        # This is a simplified version that works better with actual ground truth
        # For demonstration purposes, we'll just normalize within each predicted class
        positive_indices = predictions >= 0.5
        negative_indices = ~positive_indices
        
        # Calibrate positive predictions
        for group, indices in group_indices.items():
            group_pos = indices & positive_indices
            if group_pos.any():
                group_pos_mean = predictions[group_pos].mean()
                overall_pos_mean = predictions[positive_indices].mean()
                
                if group_pos_mean > 0:  # Avoid division by zero
                    calibration_factor = overall_pos_mean / group_pos_mean
                    calibrated[group_pos] = predictions[group_pos] * calibration_factor
        
        # Calibrate negative predictions
        for group, indices in group_indices.items():
            group_neg = indices & negative_indices
            if group_neg.any():
                group_neg_mean = predictions[group_neg].mean()
                overall_neg_mean = predictions[negative_indices].mean()
                
                if group_neg_mean > 0:  # Avoid division by zero
                    calibration_factor = overall_neg_mean / group_neg_mean
                    calibrated[group_neg] = predictions[group_neg] * calibration_factor
    
    # Ensure calibrated predictions are in [0, 1]
    calibrated = np.clip(calibrated, 0, 1)
    
    # Log calibration results
    for group in groups:
        before_mean = group_means[group]
        after_mean = calibrated[group_indices[group]].mean()
        logger.info(f"Group '{group}': mean prediction before={before_mean:.4f}, after={after_mean:.4f}")
    
    return calibrated

def adversarial_training_weights(df, prediction_col, protected_cols, label_col=None):
    """
    Calculate sample weights for adversarial training to improve fairness.
    
    Args:
        df: DataFrame with training data
        prediction_col: Column with model predictions
        protected_cols: List of columns with protected attributes
        label_col: Column with ground truth labels (if available)
    
    Returns:
        DataFrame with sample weights for adversarial training
    """
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Initialize sample weights to 1.0
    result_df['adv_sample_weight'] = 1.0
    
    # Create a combined column for intersection
    intersection_col = '_'.join(protected_cols)
    result_df[intersection_col] = result_df[protected_cols[0]].astype(str)
    for col in protected_cols[1:]:
        result_df[intersection_col] += '_' + df[col].astype(str)
    
    # Calculate positive rate for each group
    group_positive_rates = result_df.groupby(intersection_col)[prediction_col].mean()
    
    # Calculate overall positive rate
    overall_positive_rate = df[prediction_col].mean()
    
    # Calculate adversarial weights - higher weights for groups with rates
    # further from overall rate
    for group, positive_rate in group_positive_rates.items():
        # Calculate deviation from overall rate
        deviation = abs(positive_rate - overall_positive_rate)
        
        # Scale deviation to get weight (more deviated = higher weight)
        weight = 1.0 + deviation / overall_positive_rate
        
        # Apply weight to group
        result_df.loc[result_df[intersection_col] == group, 'adv_sample_weight'] = weight
    
    # Normalize weights to have a mean of 1.0
    result_df['adv_sample_weight'] = (
        result_df['adv_sample_weight'] / result_df['adv_sample_weight'].mean()
    )
    
    # Log weight statistics
    logger.info(f"Adversarial weights range: [{result_df['adv_sample_weight'].min():.4f}, {result_df['adv_sample_weight'].max():.4f}]")
    
    return result_df

def gender_neutral_preprocessing(text):
    """
    Preprocess text to make it more gender-neutral.
    
    Args:
        text: Input text
    
    Returns:
        Gender-neutralized text
    """
    # Define gender-specific terms and their neutralized replacements
    gender_terms = {
        # Pronouns
        r'\b(he|she)\b': 'they',
        r'\b(his|her)\b': 'their',
        r'\b(him|her)\b': 'them',
        r'\b(himself|herself)\b': 'themselves',
        
        # Common gender-specific terms
        r'\b(man|woman)\b': 'person',
        r'\b(men|women)\b': 'people',
        r'\b(guy|gal)\b': 'individual',
        r'\b(guys|gals)\b': 'individuals',
        r'\b(businessman|businesswoman)\b': 'businessperson',
        r'\b(businessmen|businesswomen)\b': 'businesspeople',
        r'\b(chairman|chairwoman)\b': 'chairperson',
        r'\b(mr\.|mrs\.|ms\.)\b': '',
        r'\b(actor|actress)\b': 'performer',
        r'\b(waiter|waitress)\b': 'server',
        r'\b(male|female)\b': 'person',
        r'\b(boy|girl)\b': 'child',
        r'\b(boys|girls)\b': 'children',
        r'\b(husband|wife)\b': 'spouse',
        r'\b(husbands|wives)\b': 'spouses'
    }
    
    # Apply replacements
    for pattern, replacement in gender_terms.items():
        import re
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def apply_bias_mitigation_pipeline(df, text_col, protected_cols, label_col=None, 
                                  methods=None, return_weights=True):
    """
    Apply a comprehensive bias mitigation pipeline to training data.
    
    Args:
        df: DataFrame with training data
        text_col: Column containing text
        protected_cols: List of columns with protected attributes
        label_col: Column with ground truth labels (if available)
        methods: List of methods to apply ('reweight', 'balance', 'counterfactual', 'gender_neutral')
        return_weights: Whether to return sample weights or the modified DataFrame
    
    Returns:
        Modified DataFrame or DataFrame with sample weights
    """
    result_df = df.copy()
    
    # Default to all methods if not specified
    if methods is None:
        methods = ['reweight', 'balance', 'gender_neutral']
    
    logger.info(f"Applying bias mitigation pipeline with methods: {methods}")
    
    # Apply gender-neutral preprocessing to text
    if 'gender_neutral' in methods:
        logger.info("Applying gender-neutral text preprocessing...")
        result_df[text_col] = result_df[text_col].apply(gender_neutral_preprocessing)
    
    # Apply balancing if requested
    if 'balance' in methods:
        logger.info("Balancing dataset across protected groups...")
        if label_col is not None:
            result_df = balance_dataset(result_df, protected_cols, label_col, method='upsample')
    
    # Apply counterfactual data augmentation if requested
    if 'counterfactual' in methods:
        logger.info("Applying counterfactual data augmentation...")
        # Example protected terms map (would need to be customized for real-world use)
        protected_terms_map = {
            'gender': {
                'he': 'she', 'him': 'her', 'his': 'hers', 'himself': 'herself',
                'she': 'he', 'her': 'him', 'hers': 'his', 'herself': 'himself',
                'man': 'woman', 'woman': 'man', 'men': 'women', 'women': 'men',
                'boy': 'girl', 'girl': 'boy', 'boys': 'girls', 'girls': 'boys'
            }
        }
        result_df = counterfactual_data_augmentation(result_df, text_col, protected_terms_map)
    
    # Calculate sample weights for the adjusted dataset
    if 'reweight' in methods or return_weights:
        logger.info("Calculating reweighted sample weights...")
        # Create a temporary prediction column if needed
        pred_col = label_col if label_col is not None else 'temp_pred'
        if pred_col not in result_df.columns:
            result_df[pred_col] = 0.5  # Default neutral prediction
        
        # Calculate weights
        weighted_df = reweight_training_data(result_df, pred_col, protected_cols, label_col)
        
        # Also calculate adversarial weights
        adv_weighted_df = adversarial_training_weights(result_df, pred_col, protected_cols, label_col)
        
        # Combine weights
        weighted_df['adv_sample_weight'] = adv_weighted_df['adv_sample_weight']
        
        # Clean up temporary columns
        if pred_col == 'temp_pred':
            weighted_df = weighted_df.drop(columns=[pred_col])
        
        if return_weights:
            return weighted_df
    
    return result_df

if __name__ == "__main__":
    # Example usage
    print("This module provides bias mitigation functions for sentiment analysis models.")
    print("Import this module in your main script to use its functionality.") 