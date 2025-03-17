#!/usr/bin/env python
"""
Hybrid Sentiment Analyzer with XGBoost Integration

This module implements a hybrid sentiment analysis approach that combines:
1. Text feature extraction (TF-IDF, custom features)
2. XGBoost for classification
3. Fairness evaluation metrics

The implementation follows PEP 8 with strict type hints and Google docstring format.
"""

import os
import sys
import logging
import time
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
import joblib
from datetime import datetime

# ML libraries
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    precision_recall_fscore_support
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel

# Import processing utilities from sentiment_utils
from sentiment_utils import (
    preprocess_text, 
    enhanced_preprocess_text, 
    replace_emoticons, 
    handle_negation_context, 
    standardize_slang
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text: Raw text input
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r'https?://\S+', 'URL', text)
    
    # Replace user mentions with token
    text = re.sub(r'@\w+', 'USER', text)
    
    # Replace hashtags with token
    text = re.sub(r'#(\w+)', r'HASHTAG_\1', text)
    
    # Replace repeated punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Replace extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add specific handling for Twitter-style content
    text = re.sub(r'(\s)#(\w+)', r'\1hashtag_\2', text)  # Better hashtag handling
    text = re.sub(r'(\s)@(\w+)', r'\1user_\2', text)     # Better mention handling
    text = re.sub(r'&amp;', '&', text)                   # HTML entity decoding
    text = re.sub(r'rt\s+', '', text.lower())            # Remove RT prefix
    
    return text


def enhanced_preprocess_text(text):
    # Current preprocessing
    text = preprocess_text(text)
    
    # Additional context-aware processing
    # Replace emoticons with sentiment words
    text = replace_emoticons(text)
    
    # Handle negation patterns more effectively
    text = handle_negation_context(text)
    
    # Convert slang to standard forms
    text = standardize_slang(text)
    
    return text


def replace_emoticons(text: str) -> str:
    """
    Replace common emoticons with their sentiment word equivalents.
    
    Args:
        text: Input text with potential emoticons
        
    Returns:
        Text with emoticons replaced by sentiment words
    """
    # Define common emoticons and their sentiment word replacements
    positive_emoticons = {
        ':)': ' happy ',
        ':-)': ' happy ',
        ':D': ' very_happy ',
        ':-D': ' very_happy ',
        ':p': ' playful ',
        ':-p': ' playful ',
        ';)': ' wink ',
        ';-)': ' wink ',
        '(^_^)': ' happy ',
        '<3': ' love ',
        '=)': ' happy ',
        '^^': ' happy '
    }
    
    negative_emoticons = {
        ':(': ' sad ',
        ':-(': ' sad ',
        ':/': ' skeptical ',
        ':-/': ' skeptical ',
        ':|': ' neutral ',
        ':-|': ' neutral ',
        ":'(": ' crying ',
        '>:(': ' angry ',
        '>:-(': ' angry ',
        ':(': ' sad ',
        'D:': ' scared ',
        ':-c': ' sad ',
        ':c': ' sad '
    }
    
    # Replace positive emoticons
    for emoticon, replacement in positive_emoticons.items():
        text = text.replace(emoticon, replacement)
    
    # Replace negative emoticons
    for emoticon, replacement in negative_emoticons.items():
        text = text.replace(emoticon, replacement)
    
    return text


def handle_negation_context(text: str) -> str:
    """
    Handle negation patterns in text to better capture sentiment reversals.
    
    Args:
        text: Input text potentially containing negations
        
    Returns:
        Text with negation patterns marked
    """
    # Define negation words
    negation_words = ['not', 'no', 'never', "n't", 'cannot', "can't", 'couldn\'t',
                      'wouldn\'t', 'shouldn\'t', 'won\'t', 'doesn\'t', 'didn\'t', 'isn\'t',
                      'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t']
    
    # Split the text into words
    words = text.split()
    negation_scope = False
    result = []
    
    # Process each word for negation scope
    for i, word in enumerate(words):
        # Check if word is a negation trigger
        if word.lower() in negation_words or word.lower().endswith("n't"):
            negation_scope = True
            result.append(word)
        # Check if word ends negation scope (punctuation usually ends scope)
        elif word.endswith(('.', '!', '?', ',', ';', ':', ')', ']', '}')) and negation_scope:
            negation_scope = False
            result.append(word)
        # Apply negation marking to words in negation scope
        elif negation_scope:
            result.append('NEG_' + word)
        else:
            result.append(word)
        
        # End negation after 3-5 words if no punctuation
        if negation_scope and i > 0 and (i - list(map(lambda x: x.lower(), words)).index(words[i-1].lower()) > 4):
            negation_scope = False
    
    # Join the words back into a string
    return ' '.join(result)


def standardize_slang(text: str) -> str:
    """
    Convert common social media slang to standard forms.
    
    Args:
        text: Input text with potential slang
        
    Returns:
        Text with standardized slang
    """
    # Define common slang terms and their standard equivalents
    slang_dict = {
        'u': 'you',
        'r': 'are',
        'ur': 'your',
        '2': 'to',
        '4': 'for',
        'y': 'why',
        'btw': 'by the way',
        'dm': 'direct message',
        'fb': 'facebook',
        'fomo': 'fear of missing out',
        'fyi': 'for your information',
        'idk': 'i do not know',
        'imo': 'in my opinion',
        'imho': 'in my humble opinion',
        'lol': 'laughing',
        'lmao': 'laughing',
        'rofl': 'laughing',
        'omg': 'oh my god',
        'tbh': 'to be honest',
        'thx': 'thanks',
        'thnx': 'thanks',
        'ty': 'thank you',
        'ttyl': 'talk to you later',
        'afaik': 'as far as i know',
        'bff': 'best friend',
        'brb': 'be right back',
        'b4': 'before',
        'cuz': 'because',
        'bc': 'because',
        'gr8': 'great',
        'idc': 'i do not care',
        'jk': 'just kidding',
        'k': 'okay',
        'msg': 'message',
        'ppl': 'people',
        'rn': 'right now',
        'smh': 'shaking my head',
        'tmi': 'too much information',
        'ttys': 'talk to you soon',
        'w/': 'with',
        'w/o': 'without',
        'wfh': 'work from home',
        'tho': 'though',
        'ya': 'you',
        'gonna': 'going to',
        'wanna': 'want to',
        'gotta': 'got to',
        'gimme': 'give me'
    }
    
    # First replace whole words (with spaces around them)
    for slang, standard in slang_dict.items():
        text = re.sub(r'\b' + slang + r'\b', standard, text, flags=re.IGNORECASE)
    
    # Special handling for contractions
    contractions = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how's": "how is",
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "shouldn't": "should not",
        "that'd": "that would",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'd": "what did",
        "what's": "what is",
        "where'd": "where did",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
    
    return text


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract custom text features beyond simple TF-IDF"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to pandas Series/DataFrame if it's not already
        if isinstance(X, np.ndarray):
            if X.ndim == 2 and X.shape[1] == 1:
                # This is from our column transformer, extract the text column
                X = pd.Series([x[0] if isinstance(x[0], str) else str(x[0]) for x in X])
            elif X.ndim == 1:
                # This is a 1D array
                X = pd.Series(X)
            else:
                raise ValueError(f"Unexpected shape for numpy array: {X.shape}")
        elif isinstance(X, list):
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
        positive_words = [
            'good', 'great', 'happy', 'love', 'excellent', 'awesome', 'best',
            'amazing', 'perfect', 'wonderful', 'brilliant', 'fantastic',
            'superb', 'recommend', 'recommended', 'impressed', 'favorite',
            'worth', 'pleased', 'outstanding', 'superior', 'glad'
        ]
        
        negative_words = [
            'bad', 'worst', 'hate', 'awful', 'terrible', 'sad', 'disappointed',
            'poor', 'useless', 'waste', 'horrible', 'disappointing', 'frustrating',
            'annoying', 'regret', 'avoid', 'failure', 'awful', 'unfortunately',
            'mediocre', 'not good', 'not worth', 'overpriced', 'broken'
        ]
        
        # Common negation words that can flip sentiment
        negation_words = [
            'not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t',
            'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t',
            'cannot', 'can\'t', 'couldn\'t', 'shouldn\'t', 'won\'t', 'wouldn\'t'
        ]
        
        # Count positive and negative word occurrences
        features['positive_word_count'] = X.apply(
            lambda x: sum(1 for word in positive_words if word in x.lower().split()))
        features['negative_word_count'] = X.apply(
            lambda x: sum(1 for word in negative_words if word in x.lower().split()))
        
        # Count negation words
        features['negation_count'] = X.apply(
            lambda x: sum(1 for word in negation_words if word in x.lower().split()))
        
        # Detect common negative phrases
        negative_phrases = [
            'waste of money', 'waste of time', 'don\'t buy', 'do not buy',
            'not worth', 'would not recommend', 'wouldn\'t recommend', 'stay away',
            'save your money', 'save your time', 'money back', 'refund',
            'not happy', 'not satisfied', 'not impressed', 'not pleased',
            'rip off', 'ripoff', 'scam'
        ]
        
        features['negative_phrases'] = X.apply(
            lambda x: sum(1 for phrase in negative_phrases if phrase in x.lower())
        )
        
        # Detect negated positive phrases (e.g., "not good")
        features['negated_positive'] = X.apply(
            lambda x: self._count_negated_positives(x.lower())
        )
        
        # Sentiment ratio with additional weighting for negations and phrases
        features['sentiment_score'] = features.apply(
            lambda row: (row['positive_word_count'] - row['negative_word_count'] 
                        - row['negative_phrases'] * 2 - row['negated_positive']),
            axis=1
        )
        
        # Calculate sentiment ratio (modified to handle complex cases better)
        features['sentiment_ratio'] = features.apply(
            lambda row: (row['positive_word_count'] + 0.1) / 
                       (max(0.1, row['negative_word_count'] + 
                           (row['negative_phrases'] * 2) + 
                           row['negated_positive'])),
            axis=1
        )
        
        # Add Twitter-specific features
        features['tweet_length'] = X.apply(len)
        features['word_count'] = X.apply(lambda x: len(x.split()))
        features['avg_word_length'] = features.apply(
            lambda row: row['tweet_length'] / max(1, row['word_count']), axis=1)
        features['capital_words'] = X.apply(
            lambda x: sum(1 for word in x.split() if word.isupper()))
        features['url_ratio'] = features.apply(
            lambda row: row['url_count'] / max(1, row['word_count']), axis=1)
        features['hashtag_ratio'] = features.apply(
            lambda row: row['hashtag_count'] / max(1, row['word_count']), axis=1)
        
        # Convert to numpy array
        return features.values
    
    def _count_negated_positives(self, text: str) -> int:
        """Count instances of negation words followed by positive words.
        
        Args:
            text: Lowercased text to analyze
            
        Returns:
            Count of negated positive words
        """
        # Define pattern of negation word followed by 0-2 words, then a positive word
        negation_words = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t']
        positive_words = ['good', 'great', 'happy', 'excellent', 'awesome', 'best', 'recommend']
        
        count = 0
        words = text.split()
        for i, word in enumerate(words):
            if word in negation_words and i < len(words) - 3:
                # Check next 3 words for a positive word
                for j in range(1, 4):
                    if i + j < len(words) and words[i + j] in positive_words:
                        count += 1
                        break
                        
        return count
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for output features.
        
        This method is required for extracting feature importance from the transformer.
        
        Args:
            input_features: Names of the input features (not used)
            
        Returns:
            Array of feature names for the transformed features
        """
        return np.array([
            'text_length',
            'exclamation_count',
            'question_count',
            'hashtag_count', 
            'mention_count',
            'url_count',
            'positive_word_count',
            'negative_word_count',
            'negation_count',
            'negative_phrases',
            'negated_positive',
            'sentiment_score',
            'sentiment_ratio',
            'tweet_length',
            'word_count',
            'avg_word_length',
            'capital_words',
            'url_ratio',
            'hashtag_ratio'
        ])


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    text: str
    sentiment_score: float  # -1 to 1 scale, negative to positive
    sentiment_label: str    # "negative", "neutral", "positive"
    confidence: float
    feature_importance: Dict[str, float]  # Feature -> importance mapping
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "text": self.text,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "feature_importance": self.feature_importance
        }


class DataFormatAdapter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Ensure X is in the right format for downstream transformers
        if isinstance(X, np.ndarray) and X.ndim == 1:
            return pd.DataFrame({'text': X})
        elif isinstance(X, list):
            return pd.DataFrame({'text': X})
        elif isinstance(X, pd.Series):
            return pd.DataFrame({'text': X})
        else:
            return X


class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analysis combining multiple techniques.
    
    This class implements a hybrid approach to sentiment analysis,
    combining TF-IDF features, custom text features, and XGBoost
    for classification.
    """
    
    def __init__(self, use_feature_selection=True):
        self.use_feature_selection = use_feature_selection
        self.model_path = None
        self.use_xgboost = True
        self.model = None
        self.feature_names = []
        self._build_model()
        
        # Load pre-trained model if path is provided
        if self.model_path:
            self.load_model(self.model_path)
    
    def _build_model(self) -> None:
        """Build an optimized hybrid sentiment model with feature selection."""
        # Build pipeline with or without feature selection based on self.use_feature_selection
        if self.use_feature_selection:
            # Include feature selection step
            # Step 1: Use TfidfVectorizer and CountVectorizer together with FeatureUnion
            text_features = FeatureUnion([
                ('tfidf', TfidfVectorizer(
                    max_features=4000,
                    ngram_range=(1, 3),
                    min_df=5
                )),
                ('count', CountVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('custom', TextFeatureExtractor())
            ])
            
            # Step 2: Create pipeline with feature selection using SelectFromModel
            self.model = Pipeline([
                ('features', text_features),
                # Add feature selection step
                ('feature_selection', SelectFromModel(
                    estimator=xgb.XGBClassifier(
                        n_estimators=50,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42
                    ),
                    threshold='median'  # Use median feature importance as threshold
                )),
                ('classifier', xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ))
            ])
        else:
            # Skip feature selection step
            self.model = Pipeline([
                ('classifier', xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ))
            ])
    
    def train(
        self, 
        texts: List[str], 
        labels: List[int], 
        test_size: float = 0.2,
        perform_cv: bool = True,
        optimize: bool = False
    ) -> Tuple[Dict[str, float], List[str], List[int], List[int]]:
        """
        Train the sentiment analysis model with F1 optimization.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0 for negative, 1 for positive)
            test_size: Proportion of data to use for testing
            perform_cv: Whether to perform cross-validation
            optimize: Whether to optimize hyperparameters using F1 score
            
        Returns:
            Tuple of (metrics, X_test, y_test, y_pred)
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must be the same")
        
        # Preprocess texts
        cleaned_texts = [enhanced_preprocess_text(text) for text in texts]
        
        # Convert labels to numpy array
        labels_array = np.array(labels)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_texts, labels_array, test_size=test_size, 
            random_state=42, stratify=labels_array
        )
        
        # Optimize hyperparameters if requested
        if optimize:
            logger.info("Optimizing hyperparameters using F1 score...")
            optimization_results = self.optimize_with_f1_score(X_train, y_train)
            logger.info(f"Optimization complete. Best F1: {optimization_results['best_score']:.4f}")
        
        # Train the model
        logger.info(f"Training hybrid sentiment model on {len(X_train)} samples...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Cross-validation if requested
        cv_scores = None
        if perform_cv:
            logger.info("Performing 10-fold cross-validation...")
            # Use stratified k-fold with shuffling
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            
            # Use a more diverse parameter grid
            param_grid = {
                'features__tfidf__max_features': [3000, 5000, 7000],
                'features__tfidf__ngram_range': [(1, 2), (1, 3), (2, 4)],
                'classifier__max_depth': [3, 5, 7, 9],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring='f1_weighted',
                cv=cv,
                verbose=1,
                n_jobs=-1
            )
            
            # Find best parameters
            logger.info("Starting hyperparameter optimization using F1 score...")
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.model, cleaned_texts, labels_array, 
                cv=10, scoring='f1_weighted'
            )
            logger.info(f"Cross-validation F1 scores: {cv_scores}")
            logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f}")
        
        # Evaluate using F1 score explicitly
        metrics = self.evaluate_with_f1(X_test, y_test)
        metrics['training_time_seconds'] = training_time
        metrics['train_samples'] = len(X_train)
        metrics['test_samples'] = len(X_test)
        
        if perform_cv and cv_scores is not None:
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["mean_cv_f1"] = float(cv_scores.mean())
        
        # Get predictions for return value
        y_pred = self.model.predict(X_test)
        
        return metrics, X_test, y_test, y_pred
    
    def predict(self, text: str) -> SentimentResult:
        """
        Predict the sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with prediction details
        """
        if not self.model:
            raise ValueError("Model has not been trained or loaded")
        
        # Preprocess the text
        cleaned_text = enhanced_preprocess_text(text)
        
        # Get predicted probability
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba([cleaned_text])[0]
            predicted_class = self.model.predict([cleaned_text])[0]
            confidence = probs[predicted_class]
        else:
            # For models without predict_proba
            predicted_class = self.model.predict([cleaned_text])[0]
            confidence = 0.8  # Default confidence
        
        # Map predicted class to label
        sentiment_label = "positive" if predicted_class == 1 else "negative"
        
        # Map to a score between -1 and 1
        if sentiment_label == "positive":
            sentiment_score = 0.5 + (0.5 * confidence)  # 0.5 to 1.0
        else:
            sentiment_score = -0.5 - (0.5 * confidence)  # -0.5 to -1.0
        
        # Get feature importance if available (XGBoost)
        feature_importance = {}
        if self.use_xgboost:
            try:
                # Extract the XGBoost model from the pipeline
                xgb_model = self.model.named_steps['classifier']
                
                # Get feature names if available
                if hasattr(self.model.named_steps['features'], 'get_feature_names_out'):
                    feature_names = self.model.named_steps['features'].get_feature_names_out()
                else:
                    # Fallback to generic names
                    feature_names = [f"feature_{i}" for i in range(xgb_model.feature_importances_.shape[0])]
                
                # Get top 10 features by importance
                importances = xgb_model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # Top 10
                
                for idx in indices:
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        feature_importance[feature_name] = float(importances[idx])
            except (AttributeError, KeyError) as e:
                logger.warning(f"Could not extract feature importance: {e}")
        
        return SentimentResult(
            text=text,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            feature_importance=feature_importance
        )
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.model:
            raise ValueError("No trained model to save")
        
        from joblib import dump
        dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        from joblib import load
        self.model = load(path)
        logger.info(f"Model loaded from {path}")

    def build_enhanced_model(self):
        """Build an enhanced sentiment analysis model with improved components."""
        # 1. Define preprocessing for text data
        text_processor = Pipeline([
            ('count', CountVectorizer(
                max_features=3000, 
                ngram_range=(1, 2),
                stop_words='english'
            ))
        ])
        
        # 2. Use ColumnTransformer to apply different transformations
        preprocessor = ColumnTransformer([
            ('text_processor', text_processor, 0),  # Apply to first column (text)
            ('features', TextFeatureExtractor(), 0)  # Apply custom features to text
        ])
        
        # 3. Create the full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss'
            ))
        ])
        
        # 4. Use F1 score for model evaluation and optimization
        parameters = {
            'features__tfidf__max_features': [3000, 5000],
            'features__tfidf__ngram_range': [(1, 2), (1, 3)],
            'features__count__max_features': [2000, 3000],
            'feature_selection__threshold': ['mean', 'median', '1.25*median'],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.05, 0.1]
        }
        
        model = GridSearchCV(
            pipeline, 
            parameters, 
            cv=5, 
            scoring='f1',  # Optimize for F1 score
            n_jobs=-1
        )
        
        return model

    def optimize_with_f1_score(self, X_train: List[str], y_train: List[int]) -> Dict[str, Any]:
        """
        Optimize the model using GridSearchCV with F1 score.
        
        Args:
            X_train: Training text samples
            y_train: Training labels
            
        Returns:
            Dictionary with optimization results
        """
        # Prepare parameter grid
        param_grid = {
            'features__tfidf__max_features': [3000, 5000, 7000],
            'features__tfidf__ngram_range': [(1, 2), (1, 3), (2, 4)],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        # Use stratified k-fold with shuffling
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=cv,
            verbose=1,
            n_jobs=-1
        )
        
        # Find best parameters
        logger.info("Starting hyperparameter optimization using F1 score...")
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'all_results': grid_search.cv_results_
        }
    
    def evaluate_with_f1(self, X_test: List[str], y_test: List[int]) -> Dict[str, float]:
        """
        Evaluate the model using F1 score explicitly.
        
        Args:
            X_test: Test texts
            y_test: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics explicitly using imported functions
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision, recall, _, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Log detailed report
        logger.info("\nDetailed Classification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Create metrics dictionary
        metrics = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        return metrics

    def visualize_feature_importance(self, top_n=20):
        """Visualize top feature importances."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not self.model or not hasattr(self.model, 'named_steps'):
            return
        
        try:
            # Get classifier and feature names
            classifier = self.model.named_steps['classifier']
            feature_names = []
            
            if 'feature_selection' in self.model.named_steps:
                # Get selected features
                selector = self.model.named_steps['feature_selection']
                if hasattr(self.model.named_steps['features'], 'get_feature_names_out'):
                    all_features = self.model.named_steps['features'].get_feature_names_out()
                    feature_names = [f for f, selected in zip(all_features, selector.get_support()) if selected]
            else:
                # Get all features
                if hasattr(self.model.named_steps['features'], 'get_feature_names_out'):
                    feature_names = self.model.named_steps['features'].get_feature_names_out()
            
            # Get importances
            importances = classifier.feature_importances_
            
            # Show only top_n features
            if len(feature_names) > top_n:
                indices = np.argsort(importances)[::-1][:top_n]
                feature_names = [feature_names[i] for i in indices]
                importances = importances[indices]
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x=importances, y=feature_names)
            plt.title(f"Top {len(feature_names)} Features by Importance")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing feature importance: {e}")


def analyze_sentiment_batch(
    texts: List[str],
    analyzer: Optional[HybridSentimentAnalyzer] = None,
    model_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of texts.
    
    Args:
        texts: List of texts to analyze
        analyzer: Optional pre-initialized analyzer
        model_path: Optional path to a pre-trained model
        
    Returns:
        List of sentiment analysis results
    """
    # Initialize analyzer if not provided
    if not analyzer:
        analyzer = HybridSentimentAnalyzer()
    
    results = []
    for text in texts:
        sentiment_result = analyzer.predict(text)
        results.append(sentiment_result.to_dict())
    
    return results


if __name__ == "__main__":
    # Example usage with F1 optimization
    analyzer = HybridSentimentAnalyzer()

    # Train with hyperparameter optimization using F1 score
    # This is just an example - replace with real variable names
    example_texts = ["This is great!", "This is terrible."]
    example_labels = ["positive", "negative"]
    metrics, X_test, y_test, y_pred = analyzer.train(
        texts=example_texts,
        labels=example_labels,
        perform_cv=True,
        optimize=True  # Enable F1-based optimization
    )

    # Check the F1 score
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    # Evaluate on specific examples
    test_texts = [
        "This product is amazing! Highly recommend.",
        "Terrible experience, don't waste your money.",
        "It's okay, but not worth the price."
    ]

    for text in test_texts:
        result = analyzer.predict(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result.sentiment_label} ({result.sentiment_score:.2f})")
        print(f"Confidence: {result.confidence:.2f}")
        print("---")

def load_sentiment140_sample(file_path="sentiment140.csv", sample_size=10000, random_seed=42):
    """Load and sample data from Sentiment140 dataset."""
    # Format: target,id,date,flag,user,text
    try:
        df = pd.read_csv(file_path, encoding='latin-1', 
                          names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])
        print(f"Loaded {len(df)} examples from {file_path}")
        
        # Convert sentiment from 4 (positive) to 1 and 0 (negative) to 0
        df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
        
        # Sample data
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=random_seed)
            print(f"Sampled {sample_size} examples")
            
        return df['text'].tolist(), df['sentiment'].tolist()
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

def test_model(use_feature_selection=True, sample_size=10000):
    """Test the hybrid sentiment analyzer with detailed metrics."""
    print(f"\n{'='*50}")
    print(f"Testing HybridSentimentAnalyzer {'with' if use_feature_selection else 'without'} feature selection")
    print(f"{'='*50}")
    
    # Load data
    texts, labels = load_sentiment140_sample(sample_size=sample_size)
    if not texts:
        print("No data loaded, aborting test")
        return
    
    # Initialize analyzer
    start_time = time.time()
    analyzer = HybridSentimentAnalyzer()
    print(f"Initialized analyzer in {time.time() - start_time:.2f} seconds")
    
    # Train with cross-validation and optimization
    print("\nTraining model with optimization...")
    metrics, X_test, y_test, y_pred = analyzer.train(
        texts=texts,
        labels=labels,
        test_size=0.2,
        perform_cv=True,
        optimize=True  # Enable F1-based optimization
    )
    
    # Print results
    print("\nTraining Results:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # If feature selection was used, analyze selected features
    if use_feature_selection and hasattr(analyzer, 'analyze_selected_features'):
        feature_analysis = analyzer.analyze_selected_features()
        print("\nFeature Selection Analysis:")
        print(f"Selected {feature_analysis.get('feature_count', 'unknown')} out of {feature_analysis.get('total_features', 'unknown')} features")
        
        # Print top 10 features by importance
        if 'feature_importances' in feature_analysis:
            importances = feature_analysis['feature_importances']
            print("\nTop 10 Most Important Features:")
            for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feature}: {importance:.4f}")
    
    # Test on challenging examples
    print("\nTesting on Challenging Examples:")
    challenging_examples = [
        "Not as bad as I thought it would be, but still not worth the money.",
        "I can't say I hate it, but I definitely don't love it either.",
        "It's honestly surprising how something so expensive could be so average.",
        "This isn't terrible, but there are much better alternatives.",
        "I don't think I will be buying this again, although it does have some good points."
    ]
    
    # Uncomment and import AdScorePredictor when needed
    """
    print("\n=== Testing Ad Score Predictor ===")
    try:
        from app.models.ml.prediction.ad_score_predictor import AdScorePredictor
        ad_score_predictor = AdScorePredictor()
        for text in challenging_examples:
            ad_score_result = ad_score_predictor.predict(text)
            print(f"Text: {text}")
            print(f"Ad Score Prediction: {ad_score_result}")
    except ImportError:
        print("AdScorePredictor not available")
    """

    print("\n=== Testing Hybrid Analyzer ===")
    for text in challenging_examples:
        hybrid_result = analyzer.predict(text)
        print(f"Text: {text}")
        print(f"Hybrid Prediction: {hybrid_result.sentiment_label} ({hybrid_result.sentiment_score:.2f})")
    
    return analyzer, metrics

if __name__ == "__main__":
    # Run test with feature selection
    analyzer_with_fs, metrics_with_fs = test_model(use_feature_selection=True, sample_size=10000)
    
    # Run test without feature selection
    analyzer_without_fs, metrics_without_fs = test_model(use_feature_selection=False, sample_size=10000)
    
    # Compare results
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    print(f"With Feature Selection: F1={metrics_with_fs['f1_score']:.4f}, Accuracy={metrics_with_fs['accuracy']:.4f}")
    print(f"Without Feature Selection: F1={metrics_without_fs['f1_score']:.4f}, Accuracy={metrics_without_fs['accuracy']:.4f}")

# Example of using transformer-based models
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)