#!/usr/bin/env python
"""
Transformer-Based Sentiment Analysis for Apple Silicon (MPS)

This script implements sentiment analysis using transformer models specifically
optimized for Apple Silicon's MPS (Metal Performance Shaders) environment.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import requests
from tqdm import tqdm
from datasets import load_dataset
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for MPS
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "./transformer_cache"

def preprocess_tweet(text: str) -> str:
    """Preprocess tweet text for sentiment analysis."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

def add_synthetic_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic demographic features for fairness analysis."""
    # Generate synthetic demographic features
    np.random.seed(42)
    
    # Age groups (18-24, 25-34, 35-44, 45-54, 55+)
    df['age_group'] = np.random.choice(
        ['18-24', '25-34', '35-44', '45-54', '55+'],
        size=len(df),
        p=[0.2, 0.3, 0.25, 0.15, 0.1]
    )
    
    # Gender (binary for simplicity)
    df['gender'] = np.random.choice(['M', 'F'], size=len(df), p=[0.48, 0.52])
    
    # Race/ethnicity (simplified categories)
    df['race'] = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
        size=len(df),
        p=[0.6, 0.13, 0.18, 0.06, 0.03]
    )
    
    # Education level
    df['education'] = np.random.choice(
        ['High School', 'Some College', 'Bachelor', 'Graduate'],
        size=len(df),
        p=[0.3, 0.25, 0.3, 0.15]
    )
    
    return df

def load_sentiment140(max_samples=None):
    """Load and preprocess the Sentiment140 dataset in MPS environment."""
    try:
        # Use local CSV file
        csv_path = "/Users/alecposner/WITHIN/sentiment140.csv"
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
        # Load CSV with proper encoding
        column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(csv_path, encoding='latin1', header=None, names=column_names)
        
        # Convert target from 0/4 to 0/1
        df['target'] = df['target'].map({0: 0, 4: 1})
        
        # Get class distribution
        class_dist = df['target'].value_counts()
        logger.info(f"Original class distribution: {class_dist.to_dict()}")
        
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
        
        # Apply preprocessing
        logger.info("Applying text preprocessing...")
        start_time = time.time()
        df['cleaned_text'] = df['text'].apply(preprocess_tweet)
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        
        # Add synthetic demographics
        logger.info("Adding synthetic demographic features...")
        df = add_synthetic_demographics(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

class SentimentDataset(Dataset):
    """PyTorch dataset for Sentiment140 data with MPS optimizations"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with MPS optimizations
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerSentimentAnalyzer:
    """Transformer-based sentiment analysis model optimized for MPS."""
    
    def __init__(self, model_type='distilbert', num_labels=2):
        self.model_type = model_type
        self.num_labels = num_labels
        self.model_path = None
        
        # Map model type to HuggingFace model identifier
        self.model_mapping = {
            'bert': 'bert-base-uncased',
            'distilbert': 'distilbert-base-uncased',
            'roberta': 'roberta-base',
            'xlnet': 'xlnet-base-cased'
        }
        
        # Set device to MPS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple MPS (Metal Performance Shaders) device")
        else:
            raise RuntimeError("MPS is not available on this system")
        
        # Initialize tokenizer and model with MPS optimizations
        self.model_name = self.model_mapping.get(model_type, 'distilbert-base-uncased')
        logger.info(f"Initializing tokenizer and model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True if os.path.exists(f"./models/{self.model_name}") else False,
            trust_remote_code=True
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            local_files_only=True if os.path.exists(f"./models/{self.model_name}") else False,
            trust_remote_code=True
        )
        
        # Move model to MPS and optimize
        self.model.to(self.device)
        self.optimize_for_mps()
        
        logger.info(f"Model initialized and moved to {self.device}")
    
    def optimize_for_mps(self):
        """Apply MPS-specific optimizations."""
        # Ensure model is using float32
        self.model = self.model.to(torch.float32)
        
        # Disable features that might cause problems on MPS
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Limit sequence length
        self.model.config.max_position_embeddings = min(
            self.model.config.max_position_embeddings, 
            512
        )
        
        logger.info("Applied MPS-specific optimizations")
    
    def train(self, train_dataset, val_dataset, args):
        """Train the model with MPS optimizations."""
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=args.output_dir,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            # MPS-specific settings
            fp16=False,  # Disable mixed precision for MPS
            gradient_accumulation_steps=2,  # Smaller accumulation for MPS
            dataloader_pin_memory=False,  # Disable pin memory for MPS
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        
        # Save the model
        self.model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def evaluate(self, eval_dataset):
        """Evaluate the model."""
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./tmp"),
        )
        
        eval_results = trainer.evaluate(eval_dataset)
        return eval_results
    
    def predict(self, texts):
        """Make predictions on new texts."""
        self.model.eval()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.cpu().numpy()

def evaluate_fairness(model, test_dataset, demographic_features):
    """Evaluate model fairness across demographic groups."""
    fairness_metrics = {}
    
    for feature in demographic_features:
        # Get unique values for the demographic feature
        unique_values = test_dataset[feature].unique()
        
        # Calculate metrics for each group
        group_metrics = {}
        for value in unique_values:
            # Filter dataset for this group
            group_mask = test_dataset[feature] == value
            group_dataset = test_dataset[group_mask]
            
            # Get predictions
            predictions = model.predict(group_dataset['cleaned_text'].tolist())
            true_labels = group_dataset['target'].values
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary'
            )
            
            group_metrics[value] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        fairness_metrics[feature] = group_metrics
    
    return fairness_metrics

def plot_fairness_metrics(fairness_metrics, output_dir):
    """Plot fairness metrics across demographic groups."""
    for feature, metrics in fairness_metrics.items():
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        groups = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        
        # Plot each metric
        x = np.arange(len(groups))
        width = 0.2
        
        for i, metric in enumerate(metric_names):
            values = [metrics[group][metric] for group in groups]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel(feature)
        plt.ylabel('Score')
        plt.title(f'Fairness Metrics by {feature}')
        plt.xticks(x + width*1.5, groups)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'fairness_{feature}.png'))
        plt.close()

def main():
    """Main function for MPS environment."""
    try:
        # Set default arguments for MPS
        class Args:
            def __init__(self):
                self.output_dir = './transformer_models/sentiment140'
                self.max_samples = 400000
                self.model_type = 'distilbert'
                self.test_size = 0.2
                self.batch_size = 16  # Smaller batch size for MPS
                self.learning_rate = 2e-5
                self.epochs = 3
                self.max_seq_length = 128
                self.use_full_dataset = False
                self.fairness_evaluation = False
                self.bias_mitigation = False
        
        args = Args()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        models_dir = output_dir / "models"
        fairness_dir = output_dir / "fairness"
        plots_dir = output_dir / "plots"
        
        for directory in [models_dir, fairness_dir, plots_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"Starting transformer-based sentiment analysis with {args.model_type}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model configuration: batch_size={args.batch_size}, lr={args.learning_rate}, epochs={args.epochs}")
        
        # Load and preprocess data
        if args.use_full_dataset:
            max_samples = None
            logger.info("Using full dataset as requested")
        else:
            max_samples = args.max_samples
            logger.info(f"Using max_samples={max_samples}")
        
        # Load dataset
        df = load_sentiment140(max_samples=max_samples)
        
        # Validate dataset size
        if len(df) < 1000:
            raise ValueError(f"Dataset too small: {len(df)} samples. Please check the data loading process.")
        
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Split dataset
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=42, stratify=df['target']
        )
        
        # Initialize model
        model = TransformerSentimentAnalyzer(
            model_type=args.model_type,
            num_labels=2
        )
        
        # Create datasets
        train_dataset = SentimentDataset(
            train_df['cleaned_text'].tolist(),
            train_df['target'].tolist(),
            model.tokenizer,
            max_length=args.max_seq_length
        )
        
        test_dataset = SentimentDataset(
            test_df['cleaned_text'].tolist(),
            test_df['target'].tolist(),
            model.tokenizer,
            max_length=args.max_seq_length
        )
        
        # Train model
        logger.info("Starting model training...")
        model.train(train_dataset, test_dataset, args)
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_results = model.evaluate(test_dataset)
        logger.info(f"Evaluation results: {eval_results}")
        
        # Fairness evaluation if requested
        if args.fairness_evaluation:
            logger.info("Performing fairness evaluation...")
            demographic_features = ['age_group', 'gender', 'race', 'education']
            fairness_metrics = evaluate_fairness(model, test_df, demographic_features)
            
            # Save fairness metrics
            with open(os.path.join(fairness_dir, 'fairness_metrics.json'), 'w') as f:
                json.dump(fairness_metrics, f, indent=4)
            
            # Plot fairness metrics
            plot_fairness_metrics(fairness_metrics, plots_dir)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 