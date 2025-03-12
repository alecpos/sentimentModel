"""
Ad Sentiment Analyzer for text content analysis in advertising.

This module provides sentiment analysis capabilities for ad content,
including text processing, sentiment scoring, and emotion detection.
It supports both rule-based and machine learning-based analysis approaches.
"""

import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    text: str
    sentiment_score: float  # -1 to 1 scale, negative to positive
    sentiment_label: str    # "negative", "neutral", "positive"
    confidence: float
    emotions: Dict[str, float]  # emotion -> intensity mapping
    highlighted_phrases: List[Dict[str, Any]]  # phrases with sentiment markers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "text": self.text,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "emotions": self.emotions,
            "highlighted_phrases": self.highlighted_phrases
        }

class AdSentimentAnalyzer:
    """
    Sentiment analysis for ad content.
    
    This class provides methods for analyzing the sentiment of ad content,
    detecting emotions, and identifying sentiment-charged phrases.
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_path: Optional path to a pre-trained model
        """
        self.model_path = model_path
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 3),
            min_df=5
        )
        self.emotions = {
            "joy": ["happy", "excited", "delighted", "pleased", "thrilled"],
            "trust": ["reliable", "dependable", "honest", "authentic", "secure"],
            "fear": ["worried", "scared", "anxious", "afraid", "concerned"],
            "surprise": ["amazing", "astonishing", "unexpected", "shocking", "surprising"],
            "sadness": ["sad", "unhappy", "disappointed", "depressed", "upset"],
            "disgust": ["gross", "disgusting", "unpleasant", "awful", "offensive"],
            "anger": ["angry", "frustrated", "annoyed", "irritated", "outraged"],
            "anticipation": ["eager", "looking forward", "anticipate", "expect", "awaiting"]
        }
        
        # Create sentiment lexicon
        self.positive_terms = set([
            "great", "excellent", "good", "amazing", "awesome", "wonderful", "fantastic", 
            "terrific", "outstanding", "superb", "best", "love", "perfect", "better",
            "superior", "remarkable", "exceptional", "incredible", "impressive", "extraordinary"
        ])
        
        self.negative_terms = set([
            "bad", "poor", "terrible", "awful", "horrible", "worst", "disappointing", 
            "mediocre", "ineffective", "useless", "inferior", "waste", "failure", "mess",
            "problem", "mistake", "defective", "inadequate", "substandard", "unacceptable"
        ])
        
        self.intensifiers = set([
            "very", "extremely", "incredibly", "absolutely", "completely", "totally",
            "highly", "especially", "particularly", "exceptionally", "remarkably"
        ])
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
        else:
            # Create a default logistic regression model
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', LogisticRegression(max_iter=1000, C=10.0))
            ])
    
    def train(self, texts: List[str], labels: List[int], 
             train_test_split: float = 0.2,
             save_path: Optional[Union[str, Path]] = None) -> Dict[str, float]:
        """
        Train the sentiment analysis model.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0 for negative, 1 for neutral, 2 for positive)
            train_test_split: Proportion of data to use for testing
            save_path: Optional path to save the trained model
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must be the same")
        
        # Convert labels to numpy array
        labels_array = np.array(labels)
        
        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split as tts
        X_train, X_test, y_train, y_test = tts(
            texts, labels_array, test_size=train_test_split, random_state=42
        )
        
        # Train the model
        logger.info(f"Training sentiment analysis model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save the model if a path is provided
        if save_path:
            self.save_model(save_path)
            
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
    
    def predict(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult object with analysis results
        """
        # Clean and normalize the text
        cleaned_text = self._preprocess_text(text)
        
        # Try ML-based prediction if model is available
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            # Get predicted probabilities
            probs = self.model.predict_proba([cleaned_text])[0]
            predicted_class = self.model.predict([cleaned_text])[0]
            confidence = probs[predicted_class]
            
            # Map predicted class to score and label
            if predicted_class == 0:  # Negative
                score = -0.5 - (0.5 * confidence)  # -0.5 to -1.0 depending on confidence
                label = "negative"
            elif predicted_class == 1:  # Neutral
                score = 0.0
                label = "neutral"
            else:  # Positive
                score = 0.5 + (0.5 * confidence)  # 0.5 to 1.0 depending on confidence
                label = "positive"
        else:
            # Fall back to rule-based analysis
            score, label, confidence = self._rule_based_sentiment(cleaned_text)
        
        # Detect emotions
        emotions = self._detect_emotions(cleaned_text)
        
        # Find sentiment-charged phrases
        highlighted_phrases = self._highlight_sentiment_phrases(text)
        
        return SentimentResult(
            text=text,
            sentiment_score=score,
            sentiment_label=label,
            confidence=confidence,
            emotions=emotions,
            highlighted_phrases=highlighted_phrases
        )
    
    def batch_predict(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        return [self.predict(text) for text in texts]
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _rule_based_sentiment(self, text: str) -> Tuple[float, str, float]:
        """
        Perform rule-based sentiment analysis.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Tuple of (sentiment_score, sentiment_label, confidence)
        """
        words = text.split()
        
        positive_count = sum(1 for word in words if word in self.positive_terms)
        negative_count = sum(1 for word in words if word in self.negative_terms)
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        
        # Apply intensity multiplier
        intensity_multiplier = 1.0 + (0.2 * intensifier_count)
        
        # Calculate base sentiment score
        total_sentiment_terms = positive_count + negative_count
        if total_sentiment_terms == 0:
            return 0.0, "neutral", 0.7  # Default to neutral with moderate confidence
        
        raw_score = (positive_count - negative_count) / total_sentiment_terms
        weighted_score = raw_score * intensity_multiplier
        
        # Clamp score to [-1, 1]
        score = max(-1.0, min(1.0, weighted_score))
        
        # Determine sentiment label and confidence
        if score > 0.2:
            label = "positive"
            confidence = 0.5 + (0.5 * abs(score))
        elif score < -0.2:
            label = "negative"
            confidence = 0.5 + (0.5 * abs(score))
        else:
            label = "neutral"
            confidence = 1.0 - (abs(score) * 2.5)  # Higher confidence near zero
        
        return score, label, confidence
    
    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in the text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary mapping emotions to intensity scores
        """
        results = {}
        words = set(text.split())
        
        for emotion, terms in self.emotions.items():
            # Count matching terms
            matches = sum(1 for term in terms if term in text)
            
            # Calculate intensity (0 to 1)
            if matches > 0:
                intensity = min(1.0, 0.3 + (matches * 0.15))
                results[emotion] = round(intensity, 2)
        
        # If no emotions detected, add neutral with high confidence
        if not results:
            results["neutral"] = 0.9
            
        return results
    
    def _highlight_sentiment_phrases(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify and highlight sentiment-charged phrases in the text.
        
        Args:
            text: Original text
            
        Returns:
            List of dictionaries with phrase information
        """
        highlighted = []
        
        # Look for phrases with sentiment terms
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            sentiment = None
            intensity = 0.5
            
            # Check if word is a sentiment term
            if word_lower in self.positive_terms:
                sentiment = "positive"
                intensity = 0.7
            elif word_lower in self.negative_terms:
                sentiment = "negative"
                intensity = 0.7
            
            # If sentiment term found, extract surrounding context
            if sentiment:
                # Check for intensifiers before the sentiment word
                if i > 0 and words[i-1].lower() in self.intensifiers:
                    start_idx = i - 1
                    intensity = 0.9  # Increase intensity for intensified terms
                else:
                    start_idx = i
                
                # Include a few words after the sentiment term
                end_idx = min(i + 2, len(words))
                
                # Extract the phrase
                phrase = " ".join(words[start_idx:end_idx+1])
                
                highlighted.append({
                    "phrase": phrase,
                    "sentiment": sentiment,
                    "intensity": intensity,
                    "position": {
                        "start": text.find(phrase),
                        "end": text.find(phrase) + len(phrase)
                    }
                })
        
        return highlighted
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        
        # Extract vectorizer from pipeline if available
        if hasattr(self.model, 'named_steps') and 'tfidf' in self.model.named_steps:
            self.vectorizer = self.model.named_steps['tfidf']
            
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0 for negative, 1 for neutral, 2 for positive)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model to evaluate")
            
        # Make predictions
        predictions = self.model.predict(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "test_samples": len(texts)
        } 