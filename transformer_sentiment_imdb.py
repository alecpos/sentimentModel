#!/usr/bin/env python
"""
Transformer-Based IMDB Sentiment Analysis

This script implements sentiment analysis on the IMDB movie reviews dataset using
transformer models, with a focus on cross-domain adaptation from Twitter-trained models.
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
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# PyTorch and HuggingFace Transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup,
    Trainer, TrainingArguments
)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data with proper error handling
def download_nltk_data():
    """Download required NLTK data with proper error handling."""
    required_data = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    for data in required_data:
        try:
            if data == 'punkt':
                nltk.data.find(f'tokenizers/{data}')
            else:
                nltk.data.find(f'corpora/{data}')
        except LookupError:
            logger.info(f"Downloading NLTK data: {data}")
            try:
                nltk.download(data, quiet=True)
                logger.info(f"Successfully downloaded {data}")
            except Exception as e:
                logger.error(f"Failed to download {data}: {str(e)}")
                raise

# Download NLTK data before any other operations
download_nltk_data()

def preprocess_text(text: str, is_twitter=False) -> str:
    """Preprocess text for sentiment analysis."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    if is_twitter:
        # Twitter-specific preprocessing
        text = re.sub(r'@\w+', '@user', text)
        text = re.sub(r'#\w+', '#hashtag', text)
    else:
        # Non-Twitter preprocessing - keep more original content
        # Remove excessive punctuation but keep sentence structure
        text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Remove numbers but keep them as a special token for BERTweet
    text = re.sub(r'\d+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add special tokens for BERTweet
    text = f" {text} "
    
    return text

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

def load_imdb_dataset(max_samples=None):
    """Load and preprocess the IMDB dataset."""
    try:
        logger.info("Loading IMDB dataset from Hugging Face...")
        
        # Load dataset from Hugging Face
        dataset = load_dataset("imdb", trust_remote_code=True)
        
        # Convert to pandas DataFrame
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Combine train and test sets
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'label': 'target',
            'text': 'text'
        })
        
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
        df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, is_twitter=False))
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        
        # Add synthetic demographics
        logger.info("Adding synthetic demographic features...")
        df = add_synthetic_demographics(df)
        
        # Final validation
        if len(df) < 1000:
            raise ValueError(f"Dataset too small after cleaning: {len(df)} samples")
        
        logger.info(f"Final dataset size: {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

class Args:
    def __init__(self):
        self.output_dir = './transformer_models/imdb'
        self.max_samples = None  # Use full dataset
        self.model_type = 'bertweet'  # Using BERTweet for cross-domain testing
        self.test_size = 0.1  # Reduced test size for more training data
        self.batch_size = 16  # Reduced batch size for better memory handling
        self.learning_rate = 2e-5
        self.epochs = 10
        self.max_seq_length = 128  # Standard BERTweet max length
        self.use_full_dataset = True
        self.fairness_evaluation = False
        self.bias_mitigation = False

class SentimentDataset(Dataset):
    """PyTorch dataset for IMDB sentiment data."""
    
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
        
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Ensure all tensors are properly shaped
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            # Validate tensor shapes
            if input_ids.shape[0] > self.max_length:
                raise ValueError(f"Input sequence length {input_ids.shape[0]} exceeds max_length {self.max_length}")
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error processing text at index {idx}: {str(e)}")
            # Return a safe fallback item
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.tensor(0, dtype=torch.long)
            }

class TransformerSentimentAnalyzer:
    """Transformer-based sentiment analysis model."""
    
    def __init__(self, model_type='bertweet', num_labels=2):
        self.model_type = model_type
        self.num_labels = num_labels
        self.model_path = None
        
        # Map model type to HuggingFace model identifier
        self.model_mapping = {
            'bertweet': 'vinai/bertweet-base',  # BERTweet model for cross-domain testing
            'twitter-roberta': 'cardiffnlp/twitter-roberta-base',  # Twitter-RoBERTa model
            'bert': 'bert-base-uncased',
            'distilbert': 'distilbert-base-uncased',
            'roberta': 'roberta-base',
            'xlnet': 'xlnet-base-cased'
        }
        
        # Set device to CUDA if available and clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")
        
        # Initialize tokenizer and model
        self.model_name = self.model_mapping.get(model_type, 'vinai/bertweet-base')
        logger.info(f"Initializing tokenizer and model: {self.model_name}")
        
        # Initialize tokenizer with proper configuration
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True if os.path.exists(f"./models/{self.model_name}") else False,
            trust_remote_code=True,
            use_fast=True  # Use fast tokenizer for better performance
        )
        
        # Initialize model with proper configuration
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            local_files_only=True if os.path.exists(f"./models/{self.model_name}") else False,
            trust_remote_code=True,
            ignore_mismatched_sizes=True  # Ignore size mismatches in classification head
        )
        
        # Move model to device and enable gradient checkpointing
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        
        # Log model architecture and parameters
        logger.info(f"Model initialized and moved to {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def train(self, train_dataset, val_dataset, args):
        """Train the model."""
        logger.info(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Steps per epoch: {len(train_dataset) // args.batch_size}")
        
        # Define compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary'
            )
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        # Calculate total training steps
        total_steps = len(train_dataset) // args.batch_size * args.epochs
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,  # Larger batch size for evaluation
            warmup_steps=warmup_steps,  # Dynamic warmup based on total steps
            weight_decay=0.01,
            logging_dir=args.output_dir,
            logging_steps=10,  # Log every 10 steps
            evaluation_strategy="steps",  # Ensure evaluation is done during training
            eval_steps=500,  # Evaluate every 500 steps
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            # Optimizations
            fp16=True,  # Enable mixed precision training
            gradient_checkpointing=True,
            optim="adamw_torch",
            # Memory optimizations
            max_grad_norm=1.0,  # Gradient clipping
            dataloader_num_workers=4,  # Enable multiprocessing
            remove_unused_columns=True,
            # Add detailed logging
            logging_first_step=True,
            logging_nan_inf_filter=False,
            # Add progress bar
            disable_tqdm=False,
            # Additional optimizations
            gradient_accumulation_steps=1,
            dataloader_pin_memory=True,
            # Learning rate scheduling
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",  # Use cosine annealing
            # Disable unnecessary features
            report_to="none",
            # Evaluation settings
            eval_accumulation_steps=1,  # No accumulation for evaluation
            eval_delay=0  # Start evaluation immediately
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Log initial model state
        logger.info("Initial model state:")
        logger.info(f"Model device: {self.model.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Training configuration:")
        logger.info(f"- Total steps: {total_steps}")
        logger.info(f"- Warmup steps: {warmup_steps}")
        logger.info(f"- Learning rate: {args.learning_rate}")
        logger.info(f"- Batch size: {args.batch_size}")
        
        # Train the model
        logger.info("Starting training...")
        try:
            train_result = trainer.train()
            logger.info("Training completed successfully!")
            logger.info(f"Training metrics: {train_result.metrics}")
            
            # Log final metrics
            final_eval = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {final_eval}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
        
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
            max_length=130,
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

class EnsembleSentimentAnalyzer:
    """Ensemble model combining Transformer and XGBoost predictions."""
    
    def __init__(self, transformer_model, num_labels=2):
        self.transformer_model = transformer_model
        self.num_labels = num_labels
        
        # Initialize XGBoost model with optimal parameters
        self.xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            objective='binary:logistic',
            booster='gbtree',
            tree_method='gpu_hist',  # Use GPU acceleration
            eval_metric=['logloss', 'error'],
            use_label_encoder=False,
            early_stopping_rounds=20,
            gamma=0.1,  # Minimum loss reduction for partition
            min_child_weight=5,  # Minimum sum of instance weight in child
            subsample=0.8,  # Subsample ratio of training instances
            colsample_bytree=0.8,  # Subsample ratio of columns for each tree
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
        )
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            analyzer='word',
            min_df=5,
            max_df=0.7,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Apply sublinear scaling to tf
        )
        
        # Ensemble weights (will be optimized during training)
        self.transformer_weight = 0.7
        self.xgb_weight = 0.3
    
    def train(self, train_dataset, val_dataset, args):
        """Train both models and optimize ensemble weights."""
        logger.info("Training Transformer model...")
        self.transformer_model.train(train_dataset, val_dataset, args)
        
        # Extract text and labels from datasets
        train_texts = [str(item['text']) for item in train_dataset]
        train_labels = [int(item['labels']) for item in train_dataset]
        val_texts = [str(item['text']) for item in val_dataset]
        val_labels = [int(item['labels']) for item in val_dataset]
        
        # Train XGBoost model
        logger.info("Training XGBoost model...")
        X_train = self.tfidf.fit_transform(train_texts)
        X_val = self.tfidf.transform(val_texts)
        
        # Train XGBoost with early stopping
        self.xgb_model.fit(
            X_train, train_labels,
            eval_set=[(X_val, val_labels)],
            verbose=100
        )
        
        # Optimize ensemble weights
        logger.info("Optimizing ensemble weights...")
        self._optimize_weights(val_texts, val_labels)
    
    def _optimize_weights(self, val_texts, val_labels):
        """Optimize ensemble weights using validation set."""
        # Get predictions from both models
        transformer_preds = self.transformer_model.predict(val_texts)
        X_val = self.tfidf.transform(val_texts)
        xgb_preds = self.xgb_model.predict_proba(X_val)
        
        # Grid search for optimal weights
        best_acc = 0
        best_weights = (0.7, 0.3)
        
        for w1 in np.arange(0.5, 1.0, 0.1):
            w2 = 1 - w1
            ensemble_preds = w1 * transformer_preds + w2 * xgb_preds[:, 1]
            ensemble_labels = (ensemble_preds > 0.5).astype(int)
            acc = accuracy_score(val_labels, ensemble_labels)
            
            if acc > best_acc:
                best_acc = acc
                best_weights = (w1, w2)
        
        self.transformer_weight, self.xgb_weight = best_weights
        logger.info(f"Optimal weights - Transformer: {self.transformer_weight:.2f}, XGBoost: {self.xgb_weight:.2f}")
    
    def predict(self, texts):
        """Make predictions using the ensemble."""
        # Get transformer predictions
        transformer_preds = self.transformer_model.predict(texts)
        
        # Get XGBoost predictions
        X = self.tfidf.transform(texts)
        xgb_preds = self.xgb_model.predict_proba(X)
        
        # Combine predictions using optimized weights
        ensemble_preds = (
            self.transformer_weight * transformer_preds +
            self.xgb_weight * xgb_preds[:, 1]
        )
        
        return (ensemble_preds > 0.5).astype(int)
    
    def evaluate(self, test_dataset):
        """Evaluate the ensemble model."""
        test_texts = [str(item['text']) for item in test_dataset]
        test_labels = [int(item['labels']) for item in test_dataset]
        
        # Get predictions
        predictions = self.predict(test_texts)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='binary'
        )
        
        return {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1
        }

def main():
    """Main function for IMDB sentiment analysis."""
    try:
        # Set default arguments
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
        
        logger.info(f"Starting ensemble-based IMDB sentiment analysis")
        logger.info(f"Output directory: {output_dir}")
        
        # Load and preprocess data
        df = load_imdb_dataset(max_samples=args.max_samples)
        
        # Split dataset
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=42, stratify=df['target']
        )
        
        # Initialize transformer model
        transformer_model = TransformerSentimentAnalyzer(
            model_type='bertweet',
            num_labels=2
        )
        
        # Create datasets
        train_dataset = SentimentDataset(
            train_df['cleaned_text'].tolist(),
            train_df['target'].tolist(),
            transformer_model.tokenizer,
            max_length=args.max_seq_length
        )
        
        test_dataset = SentimentDataset(
            test_df['cleaned_text'].tolist(),
            test_df['target'].tolist(),
            transformer_model.tokenizer,
            max_length=args.max_seq_length
        )
        
        # Initialize and train ensemble model
        ensemble_model = EnsembleSentimentAnalyzer(transformer_model)
        ensemble_model.train(train_dataset, test_dataset, args)
        
        # Evaluate ensemble model
        logger.info("Evaluating ensemble model...")
        eval_results = ensemble_model.evaluate(test_dataset)
        logger.info(f"Ensemble evaluation results: {eval_results}")
        
        # Save final evaluation metrics
        with open(os.path.join(output_dir, "ensemble_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 