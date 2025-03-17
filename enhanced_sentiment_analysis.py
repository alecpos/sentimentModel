#!/usr/bin/env python
"""
Enhanced Sentiment Analysis with Fairness Evaluation

This script builds upon our existing sentiment analysis implementation by:
1. Enhancing model accuracy through improved features and architecture
2. Adding synthetic demographic features for intersectional fairness analysis
3. Implementing bias detection and visualization tools
4. Providing fairness metrics across demographic intersections

Usage:
    python enhanced_sentiment_analysis.py --dataset_path path/to/sentiment140.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom text feature extractors
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract custom text features beyond simple TF-IDF"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to DataFrame if it's a list
        if isinstance(X, list):
            X = pd.Series(X)
        
        features = pd.DataFrame()
        
        # Text length
        features['text_length'] = X.apply(len)
        
        # Count special patterns
        features['exclamation_count'] = X.apply(lambda x: x.count('!'))
        features['question_count'] = X.apply(lambda x: x.count('?'))
        features['hashtag_count'] = X.apply(lambda x: len(re.findall(r'#\w+', x)))
        features['mention_count'] = X.apply(lambda x: len(re.findall(r'@\w+', x)))
        features['url_count'] = X.apply(lambda x: len(re.findall(r'https?://\S+', x)))
        
        # Sentiment-related word counts
        positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'awesome', 'best']
        negative_words = ['bad', 'worst', 'hate', 'awful', 'terrible', 'sad', 'disappointed']
        
        features['positive_word_count'] = X.apply(
            lambda x: sum(1 for word in positive_words if word in x.lower().split()))
        features['negative_word_count'] = X.apply(
            lambda x: sum(1 for word in negative_words if word in x.lower().split()))
        
        # Convert to numpy array
        return features.values

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text: Raw text
            
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
            
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keeping the text after #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def add_synthetic_demographics(df, random_seed=42):
    """
    Add synthetic demographic features to the dataset for fairness evaluation.
    
    Args:
        df: DataFrame to augment
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic demographic features
    """
    np.random.seed(random_seed)
    
    # Add demographics (synthetic for demonstration)
    df['age_group'] = np.random.choice(['18-25', '26-35', '36-50', '51+'], size=len(df))
    df['gender'] = np.random.choice([0, 1], size=len(df))
    df['location'] = np.random.choice(['rural', 'suburban', 'urban'], size=len(df))
    
    # Create intersectional groups for later analysis
    df['gender_age'] = df['gender'].astype(str) + '_' + df['age_group']
    df['gender_location'] = df['gender'].astype(str) + '_' + df['location']
    df['age_location'] = df['age_group'] + '_' + df['location']
    
    return df

def load_sentiment140(file_path, max_samples=None):
    """
    Load the Sentiment140 dataset.
    
    Args:
        file_path: Path to the dataset file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        DataFrame with the dataset
    """
    logger.info(f"Loading Sentiment140 dataset from {file_path}")
    
    # Define column names for the dataset (it doesn't have a header)
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    try:
        # Try loading with UTF-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8', header=None, names=column_names)
    except UnicodeDecodeError:
        # Fall back to latin1 encoding
        logger.info("UTF-8 encoding failed, trying latin1 encoding")
        df = pd.read_csv(file_path, encoding='latin1', header=None, names=column_names)
    
    # Convert target from 0/4 to 0/1
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Get class distribution
    class_dist = df['target'].value_counts()
    logger.info(f"Class distribution: {class_dist.to_dict()}")
    
    # Limit to max_samples if specified
    if max_samples is not None and len(df) > max_samples:
        # Ensure balanced sampling
        df_balanced = pd.DataFrame()
        for target in df['target'].unique():
            df_target = df[df['target'] == target]
            samples_per_class = max_samples // len(df['target'].unique())
            df_balanced = pd.concat([df_balanced, df_target.sample(samples_per_class, random_state=42)])
        
        df = df_balanced
        logger.info(f"Limited to {len(df)} samples ({samples_per_class} per class)")
    
    return df

def calculate_positive_rates(data, outcome_col, dem1_col, dem2_col):
    """
    Calculate positive rates across intersectional demographic groups.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - outcome_col: column name for the outcome variable (1 = positive, 0 = negative)
    - dem1_col: first demographic variable column name
    - dem2_col: second demographic variable column name
    
    Returns:
    - pivot table with positive rates for each intersection
    """
    # Group by demographic intersections and calculate positive rate
    positive_rates = data.groupby([dem1_col, dem2_col])[outcome_col].mean().reset_index()
    
    # Pivot the data for heatmap visualization
    pivot_data = positive_rates.pivot(index=dem1_col, columns=dem2_col, values=outcome_col)
    
    return pivot_data

def plot_intersectional_heatmap(pivot_data, title, save_path=None, cmap='Blues'):
    """
    Plot a heatmap of positive rates across intersectional groups.
    
    Parameters:
    - pivot_data: pivot table with rates for each intersection
    - title: title for the heatmap
    - save_path: path to save the figure (if None, just displays it)
    - cmap: colormap for the heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap)
    
    plt.title(f"{title}\nIntersectional Analysis of {pivot_data.index.name} and {pivot_data.columns.name}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Heatmap saved to {save_path}")
    else:
        plt.show()

def calculate_disparate_impact(data, outcome_col, protected_col, privileged_value):
    """
    Calculate disparate impact ratio between privileged and unprivileged groups.
    Values below 0.8 or above 1.25 typically indicate concerning bias.
    
    Parameters:
    - data: pandas DataFrame
    - outcome_col: binary outcome column (1 = positive)
    - protected_col: protected attribute column
    - privileged_value: value representing the privileged group
    
    Returns:
    - disparate impact ratio
    """
    privileged = data[data[protected_col] == privileged_value]
    unprivileged = data[data[protected_col] != privileged_value]
    
    priv_rate = privileged[outcome_col].mean()
    unpriv_rate = unprivileged[outcome_col].mean()
    
    # Avoid division by zero
    if priv_rate == 0:
        return float('inf')
    
    return unpriv_rate / priv_rate

def calculate_equalized_odds_difference(data, outcome_col, protected_col, privileged_value, actual_col):
    """
    Calculate the maximum difference in true positive and false positive rates.
    Values closer to 0 indicate better fairness.
    
    Parameters:
    - data: pandas DataFrame
    - outcome_col: predicted outcome column (1 = positive)
    - protected_col: protected attribute column
    - privileged_value: value representing the privileged group
    - actual_col: column with actual outcomes
    
    Returns:
    - maximum difference in error rates
    """
    privileged = data[data[protected_col] == privileged_value]
    unprivileged = data[data[protected_col] != privileged_value]
    
    # True positive rates
    priv_positives = privileged[privileged[actual_col] == 1]
    unpriv_positives = unprivileged[unprivileged[actual_col] == 1]
    
    # False positive rates
    priv_negatives = privileged[privileged[actual_col] == 0]
    unpriv_negatives = unprivileged[unprivileged[actual_col] == 0]
    
    # Handle empty groups
    if len(priv_positives) == 0 or len(unpriv_positives) == 0:
        tpr_diff = 1.0  # Maximum difference
    else:
        priv_tpr = priv_positives[outcome_col].mean()
        unpriv_tpr = unpriv_positives[outcome_col].mean()
        tpr_diff = abs(priv_tpr - unpriv_tpr)
    
    if len(priv_negatives) == 0 or len(unpriv_negatives) == 0:
        fpr_diff = 1.0  # Maximum difference
    else:
        priv_fpr = priv_negatives[outcome_col].mean()
        unpriv_fpr = unpriv_negatives[outcome_col].mean()
        fpr_diff = abs(priv_fpr - unpriv_fpr)
    
    return max(tpr_diff, fpr_diff)

def calculate_intersectional_metrics(data, outcome_col, dem1_col, dem2_col, actual_col=None):
    """
    Calculate fairness metrics for all intersectional groups.
    
    Parameters:
    - data: pandas DataFrame
    - outcome_col: predicted outcome column
    - dem1_col: first demographic column
    - dem2_col: second demographic column
    - actual_col: column with actual outcomes (optional)
    
    Returns:
    - DataFrame with fairness metrics for each intersection
    """
    results = []
    
    # Get all unique combinations of demographic variables
    intersections = data.groupby([dem1_col, dem2_col]).size().reset_index()
    
    # Calculate metrics for each intersection vs. rest
    for _, row in intersections.iterrows():
        dem1_val = row[dem1_col]
        dem2_val = row[dem2_col]
        
        # Create binary column for this intersection
        data['is_intersection'] = ((data[dem1_col] == dem1_val) & 
                                  (data[dem2_col] == dem2_val)).astype(int)
        
        # Calculate disparate impact
        di = calculate_disparate_impact(data, outcome_col, 'is_intersection', 0)
        
        # Calculate equalized odds difference if actual outcomes are available
        eod = None
        if actual_col is not None:
            eod = calculate_equalized_odds_difference(data, outcome_col, 'is_intersection', 0, actual_col)
        
        results.append({
            dem1_col: dem1_val,
            dem2_col: dem2_val,
            'disparate_impact': di,
            'equalized_odds_diff': eod,
            'group_size': data[data['is_intersection'] == 1].shape[0],
            'positive_rate': data[data['is_intersection'] == 1][outcome_col].mean()
        })
    
    return pd.DataFrame(results)

class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analysis model with fairness evaluation.
    
    This class provides methods for training, evaluating, and using
    sentiment analysis models on text data, with additional
    capabilities for fairness assessment.
    """
    
    def __init__(self, model_type='logistic', use_advanced_features=True):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type: Type of classifier ('logistic', 'svm', 'rf', or 'gb')
            use_advanced_features: Whether to use advanced text features beyond TF-IDF
        """
        self.model_type = model_type
        self.use_advanced_features = use_advanced_features
        self.model_path = None
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the appropriate ML pipeline based on configuration."""
        # Text preprocessing and feature extraction
        text_features = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                min_df=5,
                use_idf=True,
                sublinear_tf=True
            ))
        ])
        
        # Add custom text features if enabled
        if self.use_advanced_features:
            all_features = FeatureUnion([
                ('tfidf_features', text_features),
                ('custom_features', TextFeatureExtractor())
            ])
        else:
            all_features = text_features
        
        # Choose the classifier based on model_type
        if self.model_type == 'logistic':
            classifier = LogisticRegression(
                C=1.0,
                max_iter=5000,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'svm':
            classifier = LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'gb':
            classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create the full pipeline
        model = Pipeline([
            ('features', all_features),
            ('classifier', classifier)
        ])
        
        return model
    
    def train(self, texts, labels, test_size=0.2, perform_cv=True):
        """
        Train the sentiment analysis model.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0 for negative, 1 for positive)
            test_size: Proportion of data to use for testing
            perform_cv: Whether to perform cross-validation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must be the same")
        
        # Convert labels to numpy array
        labels_array = np.array(labels)
        
        # Split data into training and testing sets
        logger.info(f"Splitting {len(texts)} samples into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_array, test_size=test_size, random_state=42, stratify=labels_array
        )
        logger.info(f"Data split completed. Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train the model
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        start_time = time.time()
        
        # Perform cross-validation if requested
        if perform_cv:
            logger.info(f"Starting 5-fold cross-validation...")
            cv_scores = []
            # Custom cross-validation to log progress
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                fold_start = time.time()
                logger.info(f"Cross-validation fold {fold+1}/5 started...")
                
                # Get fold data
                X_fold_train = [X_train[i] for i in train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = [X_train[i] for i in val_idx]
                y_fold_val = y_train[val_idx]
                
                # Create a clone of the model for this fold
                from sklearn.base import clone
                fold_model = clone(self.model)
                
                # Fit and evaluate
                logger.info(f"  Fitting model on {len(X_fold_train)} samples...")
                fold_model.fit(X_fold_train, y_fold_train)
                logger.info(f"  Evaluating on {len(X_fold_val)} samples...")
                score = fold_model.score(X_fold_val, y_fold_val)
                cv_scores.append(score)
                
                fold_time = time.time() - fold_start
                logger.info(f"  Fold {fold+1} completed with accuracy: {score:.4f} (took {fold_time:.2f} seconds)")
            
            cv_scores = np.array(cv_scores)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f}")
        
        # Fit the model on the entire training set
        logger.info(f"Fitting final model on all {len(X_train)} training samples...")
        
        # Check if we're using a pipeline with feature extraction
        if hasattr(self.model, 'steps'):
            pipeline_steps = self.model.steps
            for i, (step_name, _) in enumerate(pipeline_steps):
                step_start = time.time()
                logger.info(f"Starting pipeline step {i+1}/{len(pipeline_steps)}: {step_name}")
                
                # If this is not the last step, we can partially fit the pipeline up to this point
                if i < len(pipeline_steps) - 1:
                    logger.info(f"  Processing features in {step_name}...")
                    partial_pipeline = Pipeline(pipeline_steps[:i+1])
                    _ = partial_pipeline.fit_transform(X_train, y_train)
                    step_time = time.time() - step_start
                    logger.info(f"  {step_name} completed in {step_time:.2f} seconds")
        
        # Actual model fitting
        fit_start = time.time()
        self.model.fit(X_train, y_train)
        fit_time = time.time() - fit_start
        logger.info(f"Model fitting completed in {fit_time:.2f} seconds")
        
        training_time = time.time() - start_time
        logger.info(f"Total training time: {training_time:.2f} seconds")
        
        # Evaluate the model
        logger.info("Evaluating model on test set...")
        eval_start = time.time()
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        eval_time = time.time() - eval_start
        
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Package evaluation metrics
        metrics = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "training_time_seconds": training_time
        }
        
        if perform_cv:
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["mean_cv_accuracy"] = float(cv_scores.mean())
        
        return metrics, X_test, y_test, y_pred
    
    def predict(self, text):
        """
        Predict the sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the text
        cleaned_text = preprocess_text(text)
        
        # Get predicted probability (for binary classification)
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba([cleaned_text])[0]
            predicted_class = self.model.predict([cleaned_text])[0]
            confidence = probs[predicted_class]
        else:
            # For models without predict_proba (like SVM)
            predicted_class = self.model.predict([cleaned_text])[0]
            confidence = 0.8  # Default confidence
        
        # Map predicted class to label
        sentiment_label = "positive" if predicted_class == 1 else "negative"
        
        # Map to a score between -1 and 1
        if sentiment_label == "positive":
            sentiment_score = 0.5 + (0.5 * confidence)  # 0.5 to 1.0
        else:
            sentiment_score = -0.5 - (0.5 * confidence)  # -0.5 to -1.0
        
        return {
            "text": text,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "class": int(predicted_class)
        }
    
    def batch_predict(self, texts):
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, path)
        self.model_path = path
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        self.model = joblib.load(path)
        self.model_path = path
        logger.info(f"Model loaded from {path}")
    
    def evaluate_fairness(self, texts, predictions, actual_labels, demographics_df):
        """
        Evaluate fairness metrics across demographic groups.
        
        Args:
            texts: List of text samples
            predictions: Predicted outcomes (0/1)
            actual_labels: Actual labels (0/1)
            demographics_df: DataFrame with demographic information
            
        Returns:
            Dictionary with fairness metrics
        """
        # Create a DataFrame with all necessary information
        eval_df = pd.DataFrame({
            'text': texts,
            'predicted': predictions,
            'actual': actual_labels
        })
        
        # Add demographic information
        for col in demographics_df.columns:
            if col not in eval_df.columns:
                eval_df[col] = demographics_df[col].values
        
        # Calculate positive rates by demographic
        gender_rates = eval_df.groupby('gender')['predicted'].mean()
        age_rates = eval_df.groupby('age_group')['predicted'].mean()
        location_rates = eval_df.groupby('location')['predicted'].mean()
        
        # Calculate intersectional positive rates
        gender_age_rates = calculate_positive_rates(eval_df, 'predicted', 'gender', 'age_group')
        gender_location_rates = calculate_positive_rates(eval_df, 'predicted', 'gender', 'location')
        age_location_rates = calculate_positive_rates(eval_df, 'predicted', 'age_group', 'location')
        
        # Calculate fairness metrics
        gender_metrics = calculate_intersectional_metrics(eval_df, 'predicted', 'gender', 'age_group', 'actual')
        location_metrics = calculate_intersectional_metrics(eval_df, 'predicted', 'gender', 'location', 'actual')
        
        # Return all metrics
        fairness_results = {
            'positive_rates': {
                'gender': gender_rates.to_dict(),
                'age_group': age_rates.to_dict(),
                'location': location_rates.to_dict(),
                'intersectional': {
                    'gender_age': gender_age_rates.to_dict(),
                    'gender_location': gender_location_rates.to_dict(),
                    'age_location': age_location_rates.to_dict()
                }
            },
            'fairness_metrics': {
                'gender_age': gender_metrics.to_dict(),
                'gender_location': location_metrics.to_dict()
            }
        }
        
        return fairness_results, eval_df

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train an enhanced sentiment analysis model on Sentiment140')
    parser.add_argument('--dataset_path', type=str, 
                      default='training.1600000.processed.noemoticon.csv',
                      help='Path to the Sentiment140 dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='enhanced_models/sentiment140',
                      help='Directory to save the model and analysis results')
    parser.add_argument('--max_samples', type=int, default=200000,
                      help='Maximum number of samples to use (None to use all)')
    parser.add_argument('--model_type', type=str, default='logistic',
                      choices=['logistic', 'svm', 'rf', 'gb'],
                      help='Type of ML model to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--advanced_features', action='store_true',
                       help='Use advanced text features beyond TF-IDF')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different outputs
        models_dir = output_dir / "models"
        fairness_dir = output_dir / "fairness"
        plots_dir = output_dir / "plots"
        
        for directory in [models_dir, fairness_dir, plots_dir]:
            directory.mkdir(exist_ok=True)
        
        # Load the dataset
        df = load_sentiment140(args.dataset_path, args.max_samples)
        
        # Preprocess text data
        logger.info("Preprocessing text data...")
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # Add synthetic demographic features
        logger.info("Adding synthetic demographic features for fairness analysis...")
        df = add_synthetic_demographics(df)
        
        # Initialize and train the model
        logger.info(f"Initializing {args.model_type} model with advanced_features={args.advanced_features}...")
        analyzer = EnhancedSentimentAnalyzer(
            model_type=args.model_type,
            use_advanced_features=args.advanced_features
        )
        
        # Train the model and get evaluation metrics
        metrics, X_test, y_test, y_pred = analyzer.train(
            texts=df['cleaned_text'].tolist(),
            labels=df['target'].tolist(),
            test_size=args.test_size,
            perform_cv=True
        )
        
        # Save the model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = models_dir / f"sentiment140_{args.model_type}_{timestamp}.joblib"
        analyzer.save_model(model_path)
        
        # Save metrics
        metrics_path = models_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create test dataset with demographic information
        test_indices = np.arange(len(df))[int(len(df) * (1 - args.test_size)):]
        test_demographics = df.iloc[test_indices].reset_index(drop=True)
        
        # Evaluate fairness
        logger.info("Evaluating fairness across demographic groups...")
        fairness_results, eval_df = analyzer.evaluate_fairness(
            texts=X_test,
            predictions=y_pred,
            actual_labels=y_test,
            demographics_df=test_demographics
        )
        
        # Save fairness results
        fairness_path = fairness_dir / "fairness_metrics.json"
        with open(fairness_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            import numpy as np
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            
            json.dump(fairness_results, f, indent=2, cls=NpEncoder)
        
        # Generate intersectional bias heatmaps
        logger.info("Generating intersectional bias visualizations...")
        
        # Gender-Age heatmap
        gender_age_rates = calculate_positive_rates(eval_df, 'predicted', 'gender', 'age_group')
        plot_intersectional_heatmap(
            gender_age_rates, 
            "Sentiment Prediction Rates by Gender and Age Group",
            save_path=plots_dir / "gender_age_heatmap.png"
        )
        
        # Gender-Location heatmap
        gender_location_rates = calculate_positive_rates(eval_df, 'predicted', 'gender', 'location')
        plot_intersectional_heatmap(
            gender_location_rates, 
            "Sentiment Prediction Rates by Gender and Location",
            save_path=plots_dir / "gender_location_heatmap.png"
        )
        
        # Age-Location heatmap
        age_location_rates = calculate_positive_rates(eval_df, 'predicted', 'age_group', 'location')
        plot_intersectional_heatmap(
            age_location_rates, 
            "Sentiment Prediction Rates by Age Group and Location",
            save_path=plots_dir / "age_location_heatmap.png"
        )
        
        # Test on example examples
        logger.info("\nTesting model on example texts...")
        test_examples = [
            "I love this product! It's amazing!",
            "This is absolutely terrible, would not recommend.",
            "It's okay, nothing special but it works.",
            "Having a great day with friends!",
            "Worst experience ever. Never going back there.",
            "The service was excellent, very attentive staff.",
            "Disappointed with the quality, not worth the money.",
            "Neutral opinion, neither good nor bad.",
            "Couldn't be happier with my purchase!",
            "Frustrating experience, wasted my time and money."
        ]
        
        for example in test_examples:
            result = analyzer.predict(example)
            logger.info(f"Text: {example}")
            logger.info(f"  Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.2f})")
            logger.info(f"  Confidence: {result['confidence']:.2f}")
            logger.info("")
        
        logger.info(f"Enhanced sentiment analysis completed successfully!")
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Fairness metrics and visualizations saved to {args.output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 