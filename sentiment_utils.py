#!/usr/bin/env python
"""
Sentiment Analysis Utilities

This module provides utility functions for sentiment analysis preprocessing,
feature extraction, and model evaluation.
"""

import re
import string
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common regex patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
NUMBER_PATTERN = re.compile(r'\b\d+\b')
EXTRA_SPACES_PATTERN = re.compile(r'\s+')
ELONGATED_PATTERN = re.compile(r'(.)\1{2,}')

# Define common emoticons and their sentiment mappings
EMOTICONS_DICT = {
    # Positive emoticons
    ':)': 'happy',
    ':-)': 'happy',
    ':D': 'laugh',
    ':-D': 'laugh',
    ':p': 'playful',
    ':-p': 'playful',
    ':P': 'playful',
    ':-P': 'playful',
    ';)': 'wink',
    ';-)': 'wink',
    '=)': 'happy',
    '=D': 'laugh',
    '<3': 'love',
    '(^_^)': 'happy',
    '(^-^)': 'happy',
    '(^.^)': 'happy',
    '^^': 'happy',
    '^_^': 'happy',
    '^-^': 'happy',
    '^.^': 'happy',
    '=]': 'happy',
    '=}': 'happy',
    ':}': 'happy',
    ':]': 'happy',
    
    # Negative emoticons
    ':(': 'sad',
    ':-(': 'sad',
    ':\'(': 'cry',
    ':"(': 'cry',
    ':-/': 'skeptical',
    ':/': 'skeptical',
    ':\\': 'skeptical',
    ':-\\': 'skeptical',
    ':|': 'indifferent',
    ':-|': 'indifferent',
    '>:(': 'angry',
    '>:-(': 'angry',
    ':@': 'angry',
    ':-@': 'angry',
    ':s': 'confused',
    ':-s': 'confused',
    ':S': 'confused',
    ':-S': 'confused',
    '=(': 'sad',
    '=\\': 'skeptical',
    '=/': 'skeptical',
    ':[': 'sad',
    ':{': 'sad',
    '>_<': 'frustrated',
    '>.<': 'frustrated',
    '-_-': 'annoyed',
}

# Define common social media slang and their standardized forms
SLANG_DICT = {
    'u': 'you',
    'ur': 'your',
    'r': 'are',
    'n': 'and',
    'y': 'why',
    'm': 'am',
    'b': 'be',
    '2': 'to',
    '4': 'for',
    'tbh': 'to be honest',
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
    'lol': 'laugh out loud',
    'rofl': 'rolling on the floor laughing',
    'lmao': 'laughing my ass off',
    'omg': 'oh my god',
    'wtf': 'what the fuck',
    'btw': 'by the way',
    'idk': 'i do not know',
    'bff': 'best friend forever',
    'brb': 'be right back',
    'afk': 'away from keyboard',
    'thx': 'thanks',
    'ty': 'thank you',
    'np': 'no problem',
    'bc': 'because',
    'b/c': 'because',
    'cuz': 'because',
    'cause': 'because',
    'pls': 'please',
    'plz': 'please',
    'rt': 'retweet',
    'fb': 'facebook',
    'dm': 'direct message',
    'gr8': 'great',
    'fav': 'favorite',
    'fwd': 'forward',
    'atm': 'at the moment',
    'abt': 'about',
    'convo': 'conversation',
    'w/': 'with',
    'w/o': 'without',
    'rly': 'really',
    'srs': 'serious',
    'tho': 'though',
    'probs': 'probably',
    'def': 'definitely',
    'ppl': 'people',
    'grt': 'great',
    'app': 'application',
    'luv': 'love',
    'haha': 'laugh',
    'hehe': 'laugh',
    'xd': 'laugh',
    'ya': 'yes',
    'yep': 'yes',
    'yea': 'yes',
    'nah': 'no',
    'nope': 'no',
}

# Define negation words
NEGATION_WORDS = [
    'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 
    'neither', 'nor', 'hardly', 'scarcely', 'barely', 'doesn\'t', 
    'isn\'t', 'wasn\'t', 'wouldn\'t', 'couldn\'t', 'won\'t', 'can\'t', 
    'don\'t', 'didnt', 'cant', 'wont', 'dont', 'doesnt', 'isnt', 'wasnt',
    'wouldnt', 'couldnt', 'havent', 'haven\'t', 'hasn\'t', 'hadn\'t',
    'shouldn\'t', 'shouldnt', 'werent', 'weren\'t', 'arent', 'aren\'t',
    'ain\'t', 'aint', 'without'
]

# Define stop words
STOP_WORDS = [
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
    'now', 'also', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that',
    'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'me', 'my', 'mine', 'myself',
    'you', 'yours', 'yourself', 'he', 'him', 'himself', 'she', 'hers', 'herself',
    'it', 'itself', 'we', 'us', 'ourselves', 'they', 'them', 'themselves',
    'what', 'which', 'who', 'whom', 'whose', 'whatever', 'whichever', 'whoever',
    'whomever', 'am'
]

def preprocess_text(text: str) -> str:
    """
    Basic preprocessing for text data.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        # Handle non-string inputs
        if text is None:
            return ""
        try:
            text = str(text)
        except:
            return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = URL_PATTERN.sub(' url ', text)
    
    # Remove mentions
    text = MENTION_PATTERN.sub(' mention ', text)
    
    # Replace hashtags with the word only
    text = HASHTAG_PATTERN.sub(lambda x: ' ' + x.group(0)[1:] + ' ', text)
    
    # Replace numbers with 'number' token
    text = NUMBER_PATTERN.sub(' number ', text)
    
    # Replace elongated characters (e.g., "sooooo" -> "so")
    text = ELONGATED_PATTERN.sub(r'\1\1', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    text = EXTRA_SPACES_PATTERN.sub(' ', text).strip()
    
    return text

def enhanced_preprocess_text(text: str) -> str:
    """
    Enhanced preprocessing for sentiment analysis.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text with sentiment-specific enhancements
    """
    # Apply basic preprocessing
    text = preprocess_text(text)
    
    # Replace emoticons with their sentiment words
    text = replace_emoticons(text)
    
    # Handle negation context
    text = handle_negation_context(text)
    
    # Standardize social media slang
    text = standardize_slang(text)
    
    return text

def replace_emoticons(text: str) -> str:
    """
    Replace emoticons with corresponding sentiment words.
    
    Args:
        text: Input text
        
    Returns:
        Text with emoticons replaced by sentiment words
    """
    for emoticon, sentiment in EMOTICONS_DICT.items():
        # Escape special regex characters in emoticons
        emoticon_escaped = re.escape(emoticon)
        text = re.sub(f'(^|\\s){emoticon_escaped}(\\s|$)', f' {sentiment} ', text)
    
    # Remove extra spaces
    text = EXTRA_SPACES_PATTERN.sub(' ', text).strip()
    
    return text

def handle_negation_context(text: str) -> str:
    """
    Handle negation context by appending 'NEG_' prefix to words following negation terms.
    
    Args:
        text: Input text
        
    Returns:
        Text with negation handling
    """
    # Tokenize text
    words = text.split()
    result = []
    
    # Initialize negation flag
    negation_active = False
    
    # Define punctuation that ends negation
    negation_enders = ['.', '!', '?', ',', ';', ':', ')', ']', '}']
    
    for word in words:
        # Check if this word starts negation
        if word in NEGATION_WORDS:
            negation_active = True
            result.append(word)
        # Check if this word ends negation
        elif any(ender in word for ender in negation_enders):
            negation_active = False
            result.append(word)
        # Apply negation prefix if active
        elif negation_active:
            result.append('NEG_' + word)
        else:
            result.append(word)
    
    return ' '.join(result)

def standardize_slang(text: str) -> str:
    """
    Standardize common social media slang to its formal equivalent.
    
    Args:
        text: Input text
        
    Returns:
        Text with standardized slang
    """
    # Tokenize
    words = text.split()
    result = []
    
    for word in words:
        # Check if the word is in our slang dictionary
        if word in SLANG_DICT:
            result.append(SLANG_DICT[word])
        else:
            result.append(word)
    
    return ' '.join(result)

def remove_stop_words(text: str, stop_words: Optional[List[str]] = None) -> str:
    """
    Remove stop words from text.
    
    Args:
        text: Input text
        stop_words: Optional list of stop words to use (uses default if None)
        
    Returns:
        Text with stop words removed
    """
    if stop_words is None:
        stop_words = STOP_WORDS
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    return ' '.join(filtered_words)

def extract_sentiment_features(text: str) -> Dict[str, Any]:
    """
    Extract features relevant for sentiment analysis.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of sentiment-relevant features
    """
    features = {}
    
    # Original text length
    features['text_length'] = len(text)
    
    # Word count
    words = text.split()
    features['word_count'] = len(words)
    
    # Count uppercase words (may indicate emphasis/shouting)
    features['uppercase_word_count'] = sum(1 for word in text.split() if word.isupper())
    
    # Count punctuation (may indicate emphasis)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Count emoticons
    features['positive_emoticon_count'] = sum(
        1 for emoticon, sentiment in EMOTICONS_DICT.items() 
        if emoticon in text and sentiment in ['happy', 'laugh', 'love', 'wink', 'playful']
    )
    features['negative_emoticon_count'] = sum(
        1 for emoticon, sentiment in EMOTICONS_DICT.items() 
        if emoticon in text and sentiment in ['sad', 'cry', 'angry', 'skeptical', 'confused', 'frustrated', 'annoyed']
    )
    
    # Check for negation
    features['contains_negation'] = any(word in text.lower() for word in NEGATION_WORDS)
    
    # Sentiment words
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
    features['positive_word_count'] = sum(1 for word in positive_words if word in text_lower)
    features['negative_word_count'] = sum(1 for word in negative_words if word in text_lower)
    
    # Simple sentiment polarity score
    features['simple_polarity'] = (features['positive_word_count'] - features['negative_word_count']) / (
        features['positive_word_count'] + features['negative_word_count'] + 1
    )
    
    return features

def evaluate_sentiment_model(y_true: List[int], y_pred: List[int], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Evaluate a sentiment analysis model and return detailed metrics.
    
    Args:
        y_true: True sentiment labels (0 for negative, 1 for positive)
        y_pred: Predicted sentiment labels
        labels: Optional label names for display
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    if labels is None:
        labels = ['negative', 'positive']
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }
    
    # Detailed report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    metrics['detailed_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Class-specific accuracy
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    metrics['positive_accuracy'] = accuracy_score(y_true[pos_mask], y_pred[pos_mask]) if any(pos_mask) else 0.0
    metrics['negative_accuracy'] = accuracy_score(y_true[neg_mask], y_pred[neg_mask]) if any(neg_mask) else 0.0
    
    return metrics

def plot_confusion_matrix(confusion_matrix, class_names, filepath=None):
    """
    Plot a confusion matrix with labels.
    
    Args:
        confusion_matrix: The confusion matrix to plot
        class_names: Names of the classes
        filepath: Path to save the plot (if None, displays it)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

def main():
    """
    Example usage of the utility functions.
    """
    example_texts = [
        "I love this product! It's amazing :)",
        "This is the worst experience ever :(",
        "Not bad, but not great either :/",
        "I can't believe how terrible this is",
        "The service was fantastic, highly recommend!"
    ]
    
    print("Basic Preprocessing Examples:")
    for text in example_texts:
        print(f"Original: {text}")
        print(f"Preprocessed: {preprocess_text(text)}")
        print("-" * 50)
    
    print("\nEnhanced Preprocessing Examples:")
    for text in example_texts:
        print(f"Original: {text}")
        print(f"Enhanced: {enhanced_preprocess_text(text)}")
        print("-" * 50)
    
    print("\nFeature Extraction Example:")
    for text in example_texts:
        features = extract_sentiment_features(text)
        print(f"Text: {text}")
        for key, value in features.items():
            print(f"  {key}: {value}")
        print("-" * 50)

if __name__ == "__main__":
    main() 