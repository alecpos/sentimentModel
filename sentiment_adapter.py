#!/usr/bin/env python
"""
Sentiment Adapter for AdScorePredictor

This module provides an adapter layer to use AdScorePredictor for sentiment analysis tasks.
It adds sentiment-specific preprocessing, calibration, and interpretation of scores.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm  # For progress bars

# Import AdScorePredictor and preprocessing utilities
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor
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

class SentimentAdapterForAdPredictor:
    """
    Adapter class to use AdScorePredictor for sentiment analysis tasks.
    
    This class wraps AdScorePredictor and adds sentiment-specific preprocessing,
    calibration, and output interpretation to make it suitable for sentiment analysis.
    """
    
    def __init__(
        self, 
        ad_predictor: Optional[AdScorePredictor] = None,
        threshold: float = 50.0,
        use_enhanced_preprocessing: bool = True,
        calibrate_scores: bool = True,
        fallback_to_internal_model: bool = True
    ):
        """
        Initialize the adapter with an AdScorePredictor instance.
        
        Args:
            ad_predictor: Optional AdScorePredictor instance (creates a new one if None)
            threshold: Score threshold for classifying positive vs negative sentiment
            use_enhanced_preprocessing: Whether to use enhanced text preprocessing
            calibrate_scores: Whether to calibrate scores based on training data
            fallback_to_internal_model: Whether to use internal sentiment model if AdScorePredictor is not fitted
        """
        self.ad_predictor = ad_predictor or AdScorePredictor()
        self.threshold = threshold
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.calibrate_scores = calibrate_scores
        self.calibration_model = None
        self.is_calibrated = False
        self.fallback_to_internal_model = fallback_to_internal_model
        self.internal_model = None
        self.using_fallback = False
        self.is_ad_predictor_fitted = True  # Assume fitted until proven otherwise
        
        # Track performance metrics
        self.metrics = {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'positive_accuracy': 0.0,
            'negative_accuracy': 0.0
        }
        
        # Track data statistics
        self.stats = {
            'training_samples': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0
        }
        
        logger.info("Initialized SentimentAdapterForAdPredictor")
        
    def _preprocess_for_sentiment(self, text: str) -> str:
        """
        Apply sentiment-specific preprocessing to the input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text optimized for sentiment analysis
        """
        if self.use_enhanced_preprocessing:
            # Use the enhanced preprocessing from sentiment_utils
            processed_text = enhanced_preprocess_text(text)
        else:
            # Use basic preprocessing
            processed_text = preprocess_text(text)
            
        # Add any adapter-specific preprocessing here
        # For example, add sentiment-specific tags or formatting
        
        return processed_text
    
    def _extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """
        Extract additional sentiment-specific features from text.
        
        Args:
            text: Input text (can be raw or preprocessed)
            
        Returns:
            Dictionary of sentiment-relevant features
        """
        # Extract basic sentiment features
        features = {}
        
        # Count positive and negative words
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
        
        # Count word occurrences
        text_lower = text.lower()
        words = text_lower.split()
        
        features['positive_word_count'] = sum(1 for word in positive_words if word in words)
        features['negative_word_count'] = sum(1 for word in negative_words if word in words)
        
        # Check for negation patterns
        features['contains_negation'] = any(word in ['not', "n't", 'no', 'never'] for word in words)
        
        # Add text length metrics
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        
        # Emoticon sentiment
        features['has_positive_emoticon'] = any(emoticon in text for emoticon in [':)', ':D', ':-)', '=)'])
        features['has_negative_emoticon'] = any(emoticon in text for emoticon in [':(', ':-(', ':/'])
        
        return features
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment prediction results
        """
        # Preprocess the text
        processed_text = self._preprocess_for_sentiment(text)
        
        # Extract sentiment-specific features
        sentiment_features = self._extract_sentiment_features(processed_text)
        
        # Check if we should use internal model (if AdScorePredictor is not fitted)
        if self.fallback_to_internal_model and (not self.is_ad_predictor_fitted or self.using_fallback):
            return self._predict_with_internal_model(processed_text, sentiment_features)
        
        # Create input for AdScorePredictor
        predictor_input = {
            'text': processed_text,
            'id': 'sentiment_analysis',  # Use a consistent ID
            'features': sentiment_features  # Add extracted features
        }
        
        # Get prediction from AdScorePredictor
        try:
            prediction = self.ad_predictor.predict(predictor_input)
            
            # Check for "Model not fitted" warning
            if 'warning' in prediction and 'not fitted' in prediction['warning']:
                self.is_ad_predictor_fitted = False
                logger.warning("AdScorePredictor not fitted, switching to internal model")
                if self.fallback_to_internal_model:
                    self.using_fallback = True
                    return self._predict_with_internal_model(processed_text, sentiment_features)
            
            # Get the raw score
            raw_score = prediction['score']
            
            # Apply calibration if available
            if self.is_calibrated and self.calibration_model is not None:
                # Convert score to sentiment probability using calibration
                # Scale the score to 0-1 range first
                normalized_score = raw_score / 100.0
                # Apply calibration
                calibrated_score = self.calibration_model.predict_proba([[normalized_score]])[0][1]
                # Scale back to the 0-100 range
                adjusted_score = calibrated_score * 100.0
            else:
                adjusted_score = raw_score
            
            # Determine sentiment label based on threshold
            sentiment = "positive" if adjusted_score > self.threshold else "negative"
            
            # Create result dictionary
            result = {
                'text': text,
                'sentiment': sentiment,
                'score': adjusted_score,
                'raw_score': raw_score,
                'confidence': prediction.get('confidence', 0.5),
                'features': sentiment_features,
                'model_used': 'ad_predictor'
            }
            
            return result
        except Exception as e:
            logger.warning(f"Error using AdScorePredictor: {e}, switching to internal model")
            self.is_ad_predictor_fitted = False
            self.using_fallback = True
            if self.fallback_to_internal_model:
                return self._predict_with_internal_model(processed_text, sentiment_features)
            else:
                # Return a default prediction if not using fallback
                return {
                    'text': text,
                    'sentiment': 'positive',  # Default
                    'score': 50.0,
                    'raw_score': 50.0,
                    'confidence': 0.5,
                    'features': sentiment_features,
                    'model_used': 'default',
                    'error': str(e)
                }
    
    def _predict_with_internal_model(self, processed_text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use internal ML model to predict sentiment when AdScorePredictor is not available/fitted.
        
        Args:
            processed_text: Preprocessed text
            features: Extracted features
            
        Returns:
            Sentiment prediction result
        """
        # Initialize internal model if needed
        if self.internal_model is None:
            self._initialize_internal_model()
        
        # Extract features for internal model
        polarity_score = features.get('simple_polarity', 0)
        
        # Calculate a sentiment score based on lexicon approach
        pos_word_count = features.get('positive_word_count', 0)
        neg_word_count = features.get('negative_word_count', 0)
        pos_emoji_count = features.get('positive_emoticon_count', 0)
        neg_emoji_count = features.get('negative_emoticon_count', 0)
        has_negation = features.get('contains_negation', False)
        
        # Simple rule-based scoring
        base_score = 50.0  # Neutral starting point
        
        # Adjust score based on features
        if pos_word_count > neg_word_count:
            base_score += 10 * (pos_word_count - neg_word_count)
        else:
            base_score -= 10 * (neg_word_count - pos_word_count)
        
        # Consider emoticons
        base_score += 5 * (pos_emoji_count - neg_emoji_count)
        
        # Adjust for negation - can flip sentiment
        if has_negation:
            # Simplistic negation handling - move score towards neutral, possibly flipping
            base_score = 50 - (base_score - 50) * 0.5
        
        # Ensure score is in range 0-100
        score = max(0, min(100, base_score))
        
        # Determine confidence based on distance from neutral
        confidence = abs(score - 50) / 50
        
        # Apply threshold
        sentiment = "positive" if score > self.threshold else "negative"
        
        # If we have a trained internal model, use it to refine prediction
        if self.internal_model is not None and hasattr(self.internal_model, 'predict_proba'):
            try:
                # Prepare features for the model
                model_features = np.array([
                    features.get('positive_word_count', 0),
                    features.get('negative_word_count', 0),
                    features.get('contains_negation', 0),
                    features.get('positive_emoticon_count', 0),
                    features.get('negative_emoticon_count', 0),
                    features.get('simple_polarity', 0),
                    features.get('text_length', 0) / 100,  # Normalize
                    features.get('word_count', 0) / 20     # Normalize
                ]).reshape(1, -1)
                
                # Get prediction probability
                proba = self.internal_model.predict_proba(model_features)[0][1]
                score = proba * 100
                sentiment = "positive" if score > self.threshold else "negative"
                confidence = max(abs(proba - 0.5) * 2, 0.5)  # Scale to 0.5-1.0
            except Exception as e:
                logger.warning(f"Error using internal model: {e}, falling back to rule-based")
        
        return {
            'text': processed_text,
            'sentiment': sentiment,
            'score': score,
            'raw_score': score,
            'confidence': confidence,
            'features': features,
            'model_used': 'internal_fallback'
        }
    
    def _initialize_internal_model(self):
        """Initialize a simple internal model for sentiment prediction"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            # Create a very basic model
            self.internal_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=5,
                random_state=42
            )
            logger.info("Initialized internal sentiment model")
        except Exception as e:
            logger.warning(f"Could not initialize internal model: {e}")
            self.internal_model = None
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment prediction results
        """
        results = []
        
        for text in texts:
            results.append(self.predict_sentiment(text))
            
        return results
    
    def train_calibration(self, texts: List[str], labels: List[int], test_size: float = 0.2) -> Dict[str, float]:
        """
        Train a calibration model for AdScorePredictor scores based on sentiment data.
        Also trains the internal model if the AdScorePredictor is not fitted.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0 for negative, 1 for positive)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with performance metrics
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must be the same")
        
        # Preprocess texts
        processed_texts = [self._preprocess_for_sentiment(text) for text in texts]
        
        # Extract features for all texts
        features_list = []
        for text in processed_texts:
            features_list.append(self._extract_sentiment_features(text))
        
        # Get raw predictions from AdScorePredictor
        raw_scores = []
        ad_predictor_fitted = True
        
        for text in processed_texts:
            try:
                prediction = self.ad_predictor.predict({'text': text, 'id': 'calibration'})
                
                # Check if model is not fitted
                if 'warning' in prediction and 'not fitted' in prediction['warning']:
                    ad_predictor_fitted = False
                    # Use a random value for calibration in this case
                    raw_scores.append(np.random.uniform(0.3, 0.7))
                else:
                    # Normalize to 0-1 range
                    raw_scores.append(prediction['score'] / 100.0)
            except Exception as e:
                logger.warning(f"Error getting prediction: {e}")
                ad_predictor_fitted = False
                # Use a default neutral value
                raw_scores.append(0.5)
        
        # Save fitted status
        self.is_ad_predictor_fitted = ad_predictor_fitted
        if not ad_predictor_fitted:
            logger.warning("AdScorePredictor not fitted. Will train internal model.")
            self.using_fallback = True
            
            # Train internal model if AdScorePredictor is not fitted
            if self.fallback_to_internal_model:
                self._train_internal_model(features_list, labels)
        
        # Create a numpy array for calibration
        raw_scores = np.array(raw_scores).reshape(-1, 1)
        labels_array = np.array(labels)
        
        # Split data for calibration
        X_train, X_test, y_train, y_test = train_test_split(
            raw_scores, labels_array, test_size=test_size, 
            random_state=42, stratify=labels_array
        )
        
        # Create and train calibration model
        from sklearn.linear_model import LogisticRegression
        base_estimator = LogisticRegression(C=1.0, solver='lbfgs')
        self.calibration_model = CalibratedClassifierCV(
            estimator=base_estimator,
            method='sigmoid',  # Use Platt scaling
            cv=5
        )
        
        # Train the calibration model
        logger.info(f"Training calibration model on {len(X_train)} samples...")
        self.calibration_model.fit(X_train, y_train)
        self.is_calibrated = True
        
        # Evaluate on test set
        y_pred_proba = self.calibration_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate class-specific accuracy
        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)
        
        pos_accuracy = accuracy_score(y_test[pos_mask], y_pred[pos_mask]) if any(pos_mask) else 0.0
        neg_accuracy = accuracy_score(y_test[neg_mask], y_pred[neg_mask]) if any(neg_mask) else 0.0
        
        # Update metrics and stats
        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'positive_accuracy': pos_accuracy,
            'negative_accuracy': neg_accuracy
        }
        
        self.stats = {
            'training_samples': len(X_train),
            'positive_ratio': np.mean(labels_array == 1),
            'negative_ratio': np.mean(labels_array == 0),
            'using_internal_model': not ad_predictor_fitted and self.fallback_to_internal_model
        }
        
        # Log results
        logger.info(f"Calibration model trained. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Positive class accuracy: {pos_accuracy:.4f}")
        logger.info(f"Negative class accuracy: {neg_accuracy:.4f}")
        
        return self.metrics
    
    def _train_internal_model(self, features_list: List[Dict[str, Any]], labels: List[int]) -> None:
        """
        Train the internal model using extracted features.
        
        Args:
            features_list: List of feature dictionaries
            labels: List of sentiment labels (0 for negative, 1 for positive)
        """
        if self.internal_model is None:
            self._initialize_internal_model()
            
        if self.internal_model is None:
            logger.warning("Could not initialize internal model, skipping training")
            return
            
        try:
            # Convert features to numpy array for model
            X = np.array([
                [
                    f.get('positive_word_count', 0),
                    f.get('negative_word_count', 0),
                    f.get('contains_negation', 0),
                    f.get('positive_emoticon_count', 0),
                    f.get('negative_emoticon_count', 0),
                    f.get('simple_polarity', 0),
                    f.get('text_length', 0) / 100,  # Normalize
                    f.get('word_count', 0) / 20     # Normalize
                ]
                for f in features_list
            ])
            
            y = np.array(labels)
            
            # Train the model
            logger.info(f"Training internal sentiment model on {len(X)} samples...")
            self.internal_model.fit(X, y)
            
            # Evaluate in-sample performance
            y_pred = self.internal_model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            logger.info(f"Internal model trained. In-sample accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.warning(f"Error training internal model: {e}")
    
    def find_optimal_threshold(self, texts: List[str], labels: List[int], 
                              thresholds: List[float] = None) -> float:
        """
        Find the optimal threshold for sentiment classification.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0 for negative, 1 for positive)
            thresholds: List of thresholds to try (defaults to range from 40 to 60)
            
        Returns:
            Optimal threshold value
        """
        if thresholds is None:
            # Default range of thresholds to try
            thresholds = np.linspace(40, 60, 21)
        
        # Get predictions for all texts
        predictions = self.predict_batch(texts)
        scores = [pred['score'] for pred in predictions]
        
        # Test each threshold
        results = []
        for threshold in thresholds:
            # Apply threshold
            predicted_labels = [1 if score > threshold else 0 for score in scores]
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predicted_labels)
            f1 = f1_score(labels, predicted_labels)
            
            results.append((threshold, accuracy, f1))
            
        # Find threshold with best F1 score
        best_threshold, best_accuracy, best_f1 = max(results, key=lambda x: x[2])
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.4f}, F1: {best_f1:.4f})")
        
        # Update threshold
        self.threshold = best_threshold
        
        return best_threshold
    
    def save(self, filepath: str) -> None:
        """
        Save the adapter to disk.
        
        Args:
            filepath: Path to save the adapter
        """
        import joblib
        
        # Create a dictionary with components to save
        save_dict = {
            'threshold': self.threshold,
            'calibration_model': self.calibration_model,
            'is_calibrated': self.is_calibrated,
            'metrics': self.metrics,
            'stats': self.stats,
            'use_enhanced_preprocessing': self.use_enhanced_preprocessing,
            'internal_model': self.internal_model,
            'using_fallback': self.using_fallback,
            'is_ad_predictor_fitted': self.is_ad_predictor_fitted,
            'fallback_to_internal_model': self.fallback_to_internal_model
        }
        
        # Save to disk
        joblib.dump(save_dict, filepath)
        logger.info(f"Adapter saved to {filepath}")
    
    def load(self, filepath: str) -> 'SentimentAdapterForAdPredictor':
        """
        Load the adapter from disk.
        
        Args:
            filepath: Path to the saved adapter
            
        Returns:
            The loaded adapter instance
        """
        import joblib
        
        # Load from disk
        save_dict = joblib.load(filepath)
        
        # Restore components
        self.threshold = save_dict['threshold']
        self.calibration_model = save_dict['calibration_model']
        self.is_calibrated = save_dict['is_calibrated']
        self.metrics = save_dict['metrics']
        self.stats = save_dict['stats']
        self.use_enhanced_preprocessing = save_dict['use_enhanced_preprocessing']
        
        # Load additional fields if available (for backward compatibility)
        self.internal_model = save_dict.get('internal_model', None)
        self.using_fallback = save_dict.get('using_fallback', False)
        self.is_ad_predictor_fitted = save_dict.get('is_ad_predictor_fitted', True)
        self.fallback_to_internal_model = save_dict.get('fallback_to_internal_model', True)
        
        logger.info(f"Adapter loaded from {filepath}")
        return self

    def predict_score(self, text: str) -> float:
        """
        Predict the raw sentiment score for the given text.
        
        This method extracts just the raw score without additional processing.
        It's mainly used for calibration.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Raw sentiment score (0-100 range)
        """
        # Preprocess the text
        processed_text = self._preprocess_for_sentiment(text)
        
        # Create input for AdScorePredictor
        predictor_input = {
            'text': processed_text,
            'id': 'sentiment_analysis',  # Use a consistent ID
        }
        
        # Get prediction from AdScorePredictor
        try:
            prediction = self.ad_predictor.predict(predictor_input)
            
            # Check for "Model not fitted" warning
            if 'warning' in prediction and 'not fitted' in prediction['warning']:
                self.is_ad_predictor_fitted = False
                logger.warning("AdScorePredictor not fitted, returning default score")
                
                # Extract sentiment features for fallback scoring
                features = self._extract_sentiment_features(processed_text)
                
                # Calculate a simple sentiment score based on features
                pos_word_count = features.get('positive_word_count', 0)
                neg_word_count = features.get('negative_word_count', 0)
                has_pos_emoticon = features.get('has_positive_emoticon', False)
                has_neg_emoticon = features.get('has_negative_emoticon', False)
                
                # Simple rule-based scoring for fallback
                score = 50.0  # Neutral base
                score += 5.0 * (pos_word_count - neg_word_count)
                score += 10.0 if has_pos_emoticon else 0.0
                score -= 10.0 if has_neg_emoticon else 0.0
                
                # Ensure in range 0-100
                return max(0.0, min(100.0, score))
            
            # Get the raw score
            return prediction['score']
            
        except Exception as e:
            logger.warning(f"Error in predict_score: {e}, returning default score")
            return 50.0  # Return neutral score as default

    def calibrate(self, scores: np.ndarray, labels: np.ndarray, find_threshold: bool = True) -> Dict[str, float]:
        """
        Calibrate the adapter scores using existing scores and labels.
        
        Args:
            scores: Array of score values (0-100 range)
            labels: Array of true labels (0 for negative, 1 for positive)
            find_threshold: Whether to find optimal threshold
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Calibrating adapter on {len(scores)} samples...")
        
        # Normalize scores to 0-1 range for calibration
        normalized_scores = scores / 100.0
        normalized_scores = normalized_scores.reshape(-1, 1)
        
        # Split data for calibration
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_scores, labels, test_size=0.2, 
            random_state=42, stratify=labels
        )
        
        # Check if AdScorePredictor is fitted
        if not self.is_ad_predictor_fitted and self.fallback_to_internal_model:
            logger.warning("AdScorePredictor not fitted - training internal model for fallback")
            self.using_fallback = True
            self._initialize_internal_model()
            
            # We need features for training internal model
            # This is a simplified version - in production you would
            # want to extract the same features used during prediction
            logger.info("Simulating features for internal model training")
            
            # Train internal model if initialized
            if self.internal_model is not None:
                # Create synthetic features based on scores
                # This is a simplified approach - in production, you would 
                # want to extract real features from the original texts
                features = np.column_stack([
                    normalized_scores,  # Use scores as a feature
                    np.random.normal(0.5, 0.1, size=len(normalized_scores)),  # Random noise
                    np.random.normal(0.5, 0.1, size=len(normalized_scores)),  # Random noise
                    np.random.normal(0.5, 0.1, size=len(normalized_scores)),  # Random noise
                    np.random.normal(0.5, 0.1, size=len(normalized_scores)),  # Random noise
                ])
                
                try:
                    self.internal_model.fit(features, labels)
                    logger.info("Internal model trained for fallback.")
                except Exception as e:
                    logger.warning(f"Error training internal model: {e}")
        
        # Create and train calibration model
        from sklearn.linear_model import LogisticRegression
        base_estimator = LogisticRegression(C=1.0, solver='lbfgs')
        self.calibration_model = CalibratedClassifierCV(
            estimator=base_estimator,
            method='sigmoid',  # Use Platt scaling
            cv=5
        )
        
        # Train the calibration model
        self.calibration_model.fit(X_train, y_train)
        self.is_calibrated = True
        
        # Evaluate on test set
        y_pred_proba = self.calibration_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate class-specific accuracy
        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)
        
        pos_accuracy = accuracy_score(y_test[pos_mask], y_pred[pos_mask]) if any(pos_mask) else 0.0
        neg_accuracy = accuracy_score(y_test[neg_mask], y_pred[neg_mask]) if any(neg_mask) else 0.0
        
        # Update metrics
        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'positive_accuracy': pos_accuracy,
            'negative_accuracy': neg_accuracy
        }
        
        # Find optimal threshold if requested
        if find_threshold:
            logger.info("Finding optimal threshold...")
            best_threshold = 50.0
            best_f1 = 0.0
            
            # Try different thresholds
            for threshold in np.linspace(40, 60, 21):
                # Apply threshold
                threshold_preds = (scores > threshold).astype(int)
                threshold_f1 = f1_score(labels, threshold_preds)
                
                if threshold_f1 > best_f1:
                    best_f1 = threshold_f1
                    best_threshold = threshold
            
            logger.info(f"Optimal threshold: {best_threshold:.1f} with F1: {best_f1:.4f}")
            self.threshold = best_threshold
        
        logger.info(f"Calibration complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return self.metrics


# Utility functions for loading and using the adapter

def load_sentiment140_sample(file_path="sentiment140.csv", sample_size=1000, random_state=42):
    """
    Load and sample data from the Sentiment140 dataset.
    
    Args:
        file_path (str): Path to the Sentiment140 dataset CSV file
        sample_size (int): Number of samples to use
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading data from {file_path}")
    
    # Column names for the Sentiment140 dataset
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    try:
        df = pd.read_csv(file_path, encoding='latin-1', names=column_names)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Convert target from 0/4 to 0/1 for negative/positive sentiment
        df['sentiment'] = df['target'].map({0: 0, 4: 1})
        
        # Keep only text and sentiment columns
        df = df[['text', 'sentiment']]
        
        # Sample data if needed
        if 0 < sample_size < len(df):
            df = df.sample(sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size} rows")
            
        return df['text'].tolist(), df['sentiment'].tolist()
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return [], []

def train_adapter_on_sentiment140(sample_size=1000, find_threshold=True, 
                            use_enhanced_preprocessing=True, fallback_to_internal_model=True):
    """
    Train a SentimentAdapterForAdPredictor on Sentiment140 dataset.
    
    Args:
        sample_size (int): Number of examples to sample from the dataset
        find_threshold (bool): Whether to find the optimal threshold
        use_enhanced_preprocessing (bool): Whether to use enhanced preprocessing
        fallback_to_internal_model (bool): Whether to use internal model if AdScorePredictor is not fitted
        
    Returns:
        SentimentAdapterForAdPredictor: Trained adapter
    """
    import pandas as pd
    import numpy as np
    import logging
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm  # For progress bars
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading Sentiment140 dataset and sampling {sample_size} examples...")
    
    # Read the CSV file - adjust path if necessary
    try:
        df = pd.read_csv('sentiment140.csv', 
                         encoding='latin-1', 
                         names=['target', 'id', 'date', 'flag', 'user', 'text'])
    except FileNotFoundError:
        logger.error("sentiment140.csv file not found. Please download it from https://www.kaggle.com/kazanova/sentiment140")
        raise
    
    # Convert target from Twitter format (0=negative, 4=positive) to binary (0=negative, 1=positive)
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Ensure we have a balanced dataset
    pos_samples = df[df['target'] == 1].sample(sample_size // 2, random_state=42)
    neg_samples = df[df['target'] == 0].sample(sample_size // 2, random_state=42)
    balanced_df = pd.concat([pos_samples, neg_samples])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Sampled {len(balanced_df)} examples with {len(pos_samples)} positive and {len(neg_samples)} negative samples")
    
    # Split data into training and testing sets
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['target'])
    
    logger.info(f"Split data into {len(train_df)} training and {len(test_df)} testing examples")
    
    # Initialize the adapter
    adapter = SentimentAdapterForAdPredictor(
        use_enhanced_preprocessing=use_enhanced_preprocessing,
        fallback_to_internal_model=fallback_to_internal_model
    )
    
    # Calibrate on training data
    logger.info("Calibrating adapter on training data...")
    X_train = train_df['text'].values
    y_train = train_df['target'].values
    
    # Use batch processing for larger datasets
    batch_size = 500
    num_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size > 0 else 0)
    
    scores = []
    labels = []
    
    for i in tqdm(range(num_batches), desc="Processing training batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        
        batch_scores = []
        for text in batch_X:
            pred = adapter.predict_score(text)
            batch_scores.append(pred)
        
        scores.extend(batch_scores)
        labels.extend(batch_y)
    
    adapter.calibrate(np.array(scores), np.array(labels), find_threshold=find_threshold)
    
    # Test the adapter
    logger.info("Testing adapter on test data...")
    X_test = test_df['text'].values
    y_test = test_df['target'].values
    
    predictions = []
    for text in tqdm(X_test, desc="Processing test examples"):
        pred = adapter.predict_sentiment(text)
        predictions.append(1 if pred['sentiment'] == 'positive' else 0)
    
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 score: {f1:.4f}")
    logger.info("\n" + classification_report(y_test, predictions))
    
    # Store metrics in the adapter
    adapter.metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'sample_size': sample_size,
        'threshold': adapter.threshold,
        'training_size': len(train_df),
        'test_size': len(test_df)
    }
    
    return adapter


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and test SentimentAdapterForAdPredictor')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of samples to use')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--save-path', type=str, default='sentiment_adapter.joblib', 
                        help='Path to save the trained adapter')
    parser.add_argument('--find-threshold', action='store_true', 
                        help='Find optimal threshold for sentiment classification')
    
    args = parser.parse_args()
    
    # Train adapter on Sentiment140 data
    adapter = train_adapter_on_sentiment140(
        sample_size=args.sample_size,
        find_threshold=args.find_threshold
    )
    
    if adapter is not None:
        # Save the trained adapter
        adapter.save(args.save_path)
        
        # Test on some examples
        test_texts = [
            "This is amazing! I love this product.",
            "Terrible experience, do not recommend.",
            "It's okay but not worth the price.",
            "The service was excellent, will definitely return!",
            "I'm disappointed with the quality."
        ]
        
        logger.info("Testing adapter on example texts:")
        for text in test_texts:
            result = adapter.predict_sentiment(text)
            logger.info(f"Text: {text}")
            logger.info(f"Sentiment: {result['sentiment']} (Score: {result['score']:.2f}, Confidence: {result['confidence']:.2f})")
            logger.info("-" * 50) 