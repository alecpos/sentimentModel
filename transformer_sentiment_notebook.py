#!/usr/bin/env python
"""
Notebook-optimized version of the transformer-based sentiment analysis.
This version is specifically designed for running in Jupyter notebooks or Google Colab.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import time
import joblib
import re
import emoji
import contractions
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)

# PyTorch and HuggingFace Transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup,
    Trainer, TrainingArguments
)
from accelerate import Accelerator, DataLoaderConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment optimizations for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Copy all the utility functions from transformer_sentiment_analysis.py
# (preprocess_tweet, load_sentiment140, etc.)
# ... [Copy all the utility functions from the original file]

# Set parameters for notebook environment
dataset_path = None  # Will be set after download
output_dir = 'transformer_models/sentiment140'
max_samples = 100000  # Smaller for initial testing on MPS
model_type = 'distilbert'
batch_size = 16  # Optimized for MPS
learning_rate = 2e-5
epochs = 3
max_seq_length = 128

# Create output directories
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

models_dir = output_dir / "models"
fairness_dir = output_dir / "fairness"
plots_dir = output_dir / "plots"

for directory in [models_dir, fairness_dir, plots_dir]:
    directory.mkdir(exist_ok=True)

logger.info(f"Starting transformer-based sentiment analysis with {model_type}")
logger.info(f"Output directory: {output_dir}")
logger.info(f"Model configuration: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")

# Download dataset if needed
if dataset_path is None or not os.path.exists(dataset_path):
    logger.info("Dataset not found, downloading...")
    dataset_path = download_sentiment140()

# Load and process data
df = load_sentiment140(dataset_path, max_samples)
logger.info(f"Loaded dataset with {len(df)} samples")

# Split data
train_val_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['target']
)

train_df, val_df = train_test_split(
    train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['target']
)

logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")

# Initialize and train model
analyzer = TransformerSentimentAnalyzer(model_type=model_type)

# Train with optimized settings for MPS
train_metrics = analyzer.train(
    train_texts=train_df['cleaned_text'].tolist(),
    train_labels=train_df['target'].tolist(),
    val_texts=val_df['cleaned_text'].tolist(),
    val_labels=val_df['target'].tolist(),
    batch_size=batch_size,
    learning_rate=learning_rate,
    epochs=epochs,
    max_seq_length=max_seq_length,
    output_dir=str(models_dir)
)

# Save training metrics
with open(models_dir / "training_metrics.json", 'w') as f:
    json.dump(train_metrics, f, indent=2)

# Evaluate the model
logger.info("Evaluating the model on test data...")
eval_metrics, all_preds, all_labels = analyzer.evaluate(
    test_texts=test_df['cleaned_text'].tolist(),
    test_labels=test_df['target'].tolist(),
    batch_size=batch_size,
    max_seq_length=max_seq_length
)

# Save evaluation metrics
with open(models_dir / "evaluation_metrics.json", 'w') as f:
    json.dump(eval_metrics, f, indent=2)

logger.info(f"Transformer-based sentiment analysis completed successfully!")
logger.info(f"Model saved to {models_dir}")
logger.info(f"Accuracy: {eval_metrics['accuracy']:.4f}, F1 Score: {eval_metrics['f1_score']:.4f}")

# Test on example texts
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

results = []
for example in test_examples:
    result = analyzer.predict(example)
    results.append(result)
    logger.info(f"Text: {example}")
    logger.info(f"  Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.2f})")
    logger.info(f"  Confidence: {result['confidence']:.2f}")
    logger.info(f"  Probabilities: Negative={result['probabilities']['negative']:.4f}, Positive={result['probabilities']['positive']:.4f}")
    logger.info("")

# Save example results
with open(output_dir / "example_predictions.json", 'w') as f:
    json.dump(results, f, indent=2) 