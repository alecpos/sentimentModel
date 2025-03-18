#!/usr/bin/env python
"""
Transformer-Based Sentiment Analysis for Google Colab Environment

This script implements sentiment analysis using transformer models specifically
optimized for Google Colab's GPU environment.
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
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import shap

# PyTorch and HuggingFace Transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup,
    Trainer, TrainingArguments,
    EarlyStoppingCallback, ProgressCallback
)
import logging
def logger():
    return logging.getLogger(__name__)

# TensorBoard imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Some logging features will be disabled.")

# Custom TensorBoard callback
class CustomTensorBoardCallback:
    """Custom TensorBoard callback implementation."""
    
    def __init__(self, writer=None, log_dir=None):
        self.writer = writer if writer else (SummaryWriter(log_dir) if TENSORBOARD_AVAILABLE and log_dir else None)
        if not self.writer and TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard writer not initialized. Logging will be disabled.")
    
    def on_train_begin(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.add_text("hyperparameters", str(args))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.writer:
            return
        
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.close()

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
        'averaged_perceptron_tagger',
        'punkt_tab'  # Added punkt_tab
    ]
    
    for data in required_data:
        try:
            if data == 'punkt_tab':
                nltk.data.find(f'tokenizers/{data}/english/')
            else:
                nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
        except LookupError:
            logger.info(f"Downloading NLTK data: {data}")
            try:
                nltk.download(data, quiet=True)
                logger.info(f"Successfully downloaded {data}")
            except Exception as e:
                logger.error(f"Failed to download {data}: {str(e)}")
                # For punkt_tab, try alternative download method
                if data == 'punkt_tab':
                    try:
                        import urllib.request
                        import os
                        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
                        urllib.request.urlretrieve(url, "punkt_tab.zip")
                        import zipfile
                        with zipfile.ZipFile("punkt_tab.zip", "r") as zip_ref:
                            zip_ref.extractall(nltk.data.path[0])
                        os.remove("punkt_tab.zip")
                        logger.info("Successfully downloaded punkt_tab using alternative method")
                    except Exception as e:
                        logger.error(f"Failed to download punkt_tab using alternative method: {str(e)}")
                        raise

# Download NLTK data before any other operations
download_nltk_data()

# Set environment variables for Colab
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/content/transformer_cache"

def preprocess_tweet(text: str) -> str:
    """Preprocess tweet text for sentiment analysis."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions but keep the @ symbol for BERTweet
    text = re.sub(r'@\w+', '@user', text)
    
    # Remove hashtags but keep the # symbol for BERTweet
    text = re.sub(r'#\w+', '#hashtag', text)
    
    # Remove numbers but keep them as a special token for BERTweet
    text = re.sub(r'\d+', '<number>', text)
    
    # Remove special characters but keep emojis and basic punctuation
    text = re.sub(r'[^\w\s@#\U0001F300-\U0001F9FF]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add special tokens for BERTweet
    text = f"<s> {text} </s>"
    
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

def download_sentiment140():
    """Download the Sentiment140 dataset in Colab environment."""
    logger.info("Downloading Sentiment140 dataset...")
    
    # Create directories
    os.makedirs("/content/sentiment140_data", exist_ok=True)
    csv_path = "/content/sentiment140_data/sentiment140.csv"
    
    # Download using gdown
    import gdown
    url = "https://docs.google.com/uc?export=download&id=0B04GJPshIjmPRnZManQwWEdTZjg"
    gdown.download(url, csv_path, quiet=False)
    
    return csv_path

def load_sentiment140(max_samples=None):
    """Load and preprocess the Sentiment140 dataset in Colab."""
    try:
        logger.info("Loading Sentiment140 dataset from Hugging Face...")
        
        # Load dataset from Hugging Face
        dataset = load_dataset("stanfordnlp/sentiment140", trust_remote_code=True)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'sentiment': 'target',
            'text': 'text'
        })
        
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
        
        # Final validation
        if len(df) < 1000:
            raise ValueError(f"Dataset too small after cleaning: {len(df)} samples")
        
        logger.info(f"Final dataset size: {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

class SentimentDataset(Dataset):
    """PyTorch dataset for Sentiment140 data."""
    
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
    """Transformer-based sentiment analysis model."""
    
    def __init__(self, model_type='bertweet', num_labels=2):
        self.model_type = model_type
        self.num_labels = num_labels
        self.model_path = None
        
        # Map model type to HuggingFace model identifier
        self.model_mapping = {
            'bertweet': 'vinai/bertweet-base',  # BERTweet model
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
        
        # Initialize the classification head with proper weights based on model architecture
        try:
            if hasattr(self.model, 'classifier'):
                # Try different initialization patterns based on model architecture
                try:
                    # Pattern 1: Direct classifier with weight attribute (e.g., DistilBERT)
                    if hasattr(self.model.classifier, 'weight'):
                        torch.nn.init.kaiming_normal_(self.model.classifier.weight, mode='fan_out', nonlinearity='relu')
                        if self.model.classifier.bias is not None:
                            torch.nn.init.zeros_(self.model.classifier.bias)
                        logger.info("Initialized classifier weights (Pattern 1)")
                    
                    # Pattern 2: RoBERTa-style classifier with dense and out_proj layers
                    elif hasattr(self.model.classifier, 'dense') and hasattr(self.model.classifier, 'out_proj'):
                        torch.nn.init.kaiming_normal_(self.model.classifier.dense.weight, mode='fan_out', nonlinearity='relu')
                        torch.nn.init.zeros_(self.model.classifier.dense.bias)
                        torch.nn.init.kaiming_normal_(self.model.classifier.out_proj.weight, mode='fan_out', nonlinearity='relu')
                        torch.nn.init.zeros_(self.model.classifier.out_proj.bias)
                        logger.info("Initialized classifier weights (Pattern 2)")
                    
                    # Pattern 3: BERT-style classifier with pre_classifier
                    elif hasattr(self.model.classifier, 'pre_classifier'):
                        torch.nn.init.kaiming_normal_(self.model.classifier.pre_classifier.weight, mode='fan_out', nonlinearity='relu')
                        torch.nn.init.zeros_(self.Zmodel.classifier.pre_classifier.bias)
                        torch.nn.init.kaiming_normal_(self.model.classifier.weight, mode='fan_out', nonlinearity='relu')
                        torch.nn.init.zeros_(self.model.classifier.bias)
                        logger.info("Initialized classifier weights (Pattern 3)")
                    
                    else:
                        logger.warning("Could not identify classifier architecture pattern, using default initialization")
                
                except Exception as e:
                    logger.warning(f"Error during classifier initialization: {str(e)}")
                    logger.info("Using default initialization for classifier weights")
            
            else:
                logger.warning("Model does not have a classifier attribute, using default initialization")
        
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise
        
        # Move model to device and enable gradient checkpointing
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        
        # Log model architecture and parameters
        logger.info(f"Model initialized and moved to {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def train(self, train_dataset, val_dataset, args):
        """Train the model with advanced optimizations."""
        logger.info(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        
        # Check if we're using A100
        is_a100 = torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)
        logger.info(f"Using A100 GPU: {is_a100}")
        
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
        
        # Create output directories
        tensorboard_dir = os.path.join(args.output_dir, 'runs')
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Base training arguments that are always supported
        base_args = {
            'output_dir': args.output_dir,
            'num_train_epochs': args.epochs,
            'per_device_train_batch_size': args.batch_size,
            'per_device_eval_batch_size': args.batch_size * 2,
            'warmup_steps': warmup_steps,
            'weight_decay': 0.01,
            'logging_dir': tensorboard_dir,
            'logging_steps': 10,
            'evaluation_strategy': "steps",
            'eval_steps': 500,
            'save_strategy': "steps",
            'save_steps': 500,
            'load_best_model_at_end': True,
            'metric_for_best_model': "accuracy",
            'greater_is_better': True,
            'optim': "adamw_torch",
            'max_grad_norm': 1.0,
            'dataloader_num_workers': 4,
            'remove_unused_columns': True,
            'dataloader_pin_memory': True,
            'learning_rate': args.learning_rate,
            'lr_scheduler_type': "cosine_with_restarts",
            'logging_first_step': True,
            'logging_nan_inf_filter': False,
            'report_to': ["tensorboard"] if TENSORBOARD_AVAILABLE else [],
            'eval_accumulation_steps': 1,
            'eval_delay': 0,
            'push_to_hub': False,
            'hub_strategy': "every_save"
        }
        
        # Advanced optimizations that might not be supported in all versions
        advanced_args = {
            'gradient_checkpointing': True,
            'gradient_accumulation_steps': 4,
            'ddp_find_unused_parameters': True,
            'torch_compile': is_a100,
            'tf32': is_a100
        }
        
        # Set mixed precision based on GPU type
        if is_a100:
            advanced_args['bf16'] = True
            advanced_args['fp16'] = False
        else:
            advanced_args['bf16'] = False
            advanced_args['fp16'] = True
        
        try:
            # Try with all optimizations
            training_args = TrainingArguments(**base_args, **advanced_args)
            logger.info("Successfully initialized training arguments with all optimizations")
        except TypeError as e:
            logger.warning(f"Error with advanced training arguments: {e}")
            logger.info("Falling back to basic configuration")
            
            # Remove potentially problematic parameters
            fallback_args = base_args.copy()
            fallback_args.update({
                'fp16': True,  # Keep basic mixed precision
                'gradient_checkpointing': True,  # Keep basic memory optimization
                'gradient_accumulation_steps': 4  # Keep basic gradient accumulation
            })
            
            training_args = TrainingArguments(**fallback_args)
            logger.info("Successfully initialized training arguments with fallback configuration")
        
        # Initialize trainer with enhanced configuration and custom TensorBoard callback
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=3),
            ProgressCallback()
        ]
        
        # Add TensorBoard callback if available
        if TENSORBOARD_AVAILABLE:
            callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks
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
        logger.info(f"- Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        logger.info(f"- Mixed precision: {'bf16' if is_a100 else 'fp16'}")
        
        # Train the model with error handling and progress tracking
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

class EnsembleSentimentAnalyzer:
    """Enhanced ensemble model combining Transformer, XGBoost, and advanced ensemble methods."""
    
    def __init__(self, transformer_model, num_labels=2):
        self.transformer_model = transformer_model
        self.num_labels = num_labels
        self.version = "2.1.0"  # Updated version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize base models for bagging with enhanced regularization
        self.bagging_models = []
        self.n_bagging_models = 5  # Number of bagging models
        
        # Initialize XGBoost models for bagging with enhanced regularization
        tree_method = 'gpu_hist' if torch.cuda.is_available() else 'hist'
        for _ in range(self.n_bagging_models):
            self.bagging_models.append(xgb.XGBClassifier(
                max_depth=4,                  # Reduced from 6
                learning_rate=0.05,           # Reduced from 0.1
                n_estimators=200,
                objective='binary:logistic',
                booster='gbtree',
                tree_method=tree_method,
                eval_metric=['logloss', 'error', 'auc'],
                use_label_encoder=False,
                early_stopping_rounds=30,     # Increased from 20
                gamma=0.2,                    # Increased from 0.1
                min_child_weight=7,           # Increased from 5
                subsample=0.7,                # Decreased from 0.8
                colsample_bytree=0.7,         # Decreased from 0.8
                reg_alpha=0.3,                # Increased from 0.1
                reg_lambda=2.0,               # Increased from 1.0
                scale_pos_weight=1,           # Added for class balance
                random_state=42,
                n_jobs=-1,
                enable_categorical=True,
                max_cat_to_onehot=10
            ))
        
        # Initialize additional diverse base models
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from lightgbm import LGBMClassifier
        
        self.diverse_models = [
            ("lightgbm", LGBMClassifier(
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.7,
                random_state=42
            )),
            ("svm", SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                random_state=42
            )),
            ("logistic", LogisticRegression(
                C=0.1,
                max_iter=500,
                random_state=42
            ))
        ]
        
        # Initialize stacking meta-learner with enhanced regularization
        self.meta_learner = xgb.XGBClassifier(
            max_depth=3,                      # Reduced from 4
            learning_rate=0.03,               # Reduced from 0.05
            n_estimators=100,
            objective='binary:logistic',
            tree_method=tree_method,
            use_label_encoder=False,
            early_stopping_rounds=30,
            gamma=0.2,
            min_child_weight=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=2.0,
            scale_pos_weight=1,
            random_state=42
        )
        
        # Enhanced TF-IDF vectorizer with better feature engineering
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            analyzer='word',
            min_df=5,
            max_df=0.7,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            binary=False,
            norm='l2'
        )
        
        # Initialize SHAP explainer (will be set during training)
        self.explainer = None
        
        # Model calibration parameters
        self.calibration = {
            'method': 'isotonic',
            'cv_folds': 5
        }
        
        # Fairness thresholds
        self.fairness_thresholds = {
            'demographic_parity_diff': 0.1,
            'equal_opportunity_diff': 0.1
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'inference_time': [],
            'prediction_distribution': [],
            'feature_importance': None,
            'bagging_metrics': [],
            'stacking_metrics': [],
            'cv_scores': []
        }
        
        # Initialize ensemble weights (will be optimized)
        self.weights = None
        
        # Shadow deployment mode
        self.shadow_mode = False
        self.shadow_predictions = []
        
        # Enhanced SHAP configuration
        self.shap_config = {
            'background_size': 100,
            'batch_size': 32,
            'max_display': 15,
            'cache_size': 1000
        }
        
        # Initialize explanation cache
        self.explanation_cache = {}
        
        # Initialize background dataset
        self.background_dataset = None
        
        # Initialize visualization settings
        self.viz_config = {
            'emotion_colors': {
                'joy': '#FFD700',
                'sadness': '#4169E1',
                'anger': '#DC143C',
                'fear': '#800080',
                'neutral': '#808080'
            },
            'plot_style': 'dark_background'
        }
        
        logger.info(f"Initialized enhanced ensemble model version {self.version} with bagging and stacking")

    def _optimize_weights(self, base_models, X_val, y_val):
        """Optimize ensemble weights using validation performance."""
        from scipy.optimize import minimize
        
        # Get predictions from each model
        predictions = []
        for model in base_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val)[:, 1]
            else:
                pred = model.predict(X_val)
            predictions.append(pred)
        
        # Function to minimize (negative AUC)
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Weighted prediction
            weighted_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_pred += weights[i] * pred
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            return -roc_auc_score(y_val, weighted_pred)
        
        # Initial weights (equal)
        initial_weights = np.ones(len(base_models)) / len(base_models)
        
        # Optimize
        bounds = [(0, 1)] * len(base_models)
        result = minimize(objective, initial_weights, bounds=bounds)
        
        # Return optimized weights
        return result.x / np.sum(result.x)

    def _train_bagging_models(self, X_train, y_train, X_val, y_val):
        """Train bagging models with cross-validation."""
        logger.info("Training bagging models with cross-validation...")
        
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, model in enumerate(self.bagging_models):
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold = X_train[train_idx]
                y_train_fold = y_train[train_idx]
                X_val_fold = X_train[val_idx]
                y_val_fold = y_train[val_idx]
                
                # Train model
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=50,
                    early_stopping_rounds=30
                )
                
                # Store metrics
                val_pred = model.predict_proba(X_val_fold)[:, 1]
                val_auc = roc_auc_score(y_val_fold, val_pred)
                cv_scores.append(val_auc)
            
            # Store cross-validation metrics
            self.performance_metrics['cv_scores'].append({
                'model_idx': i,
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores)
            })
            
            # Final training on full training set
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50,
                early_stopping_rounds=30
            )
            
            logger.info(f"Bagging model {i+1}/{self.n_bagging_models} trained with mean CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    def _prepare_stacking_features(self, X, y=None):
        """Prepare features for stacking meta-learner using cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize meta-features array
        n_samples = X.shape[0]
        n_base_models = len(self.bagging_models) + len(self.diverse_models) + 1  # +1 for transformer
        meta_features = np.zeros((n_samples, n_base_models * 2))  # 2 for probability predictions
        
        # Get transformer predictions
        transformer_preds = self.transformer_model.predict(X)
        meta_features[:, 0:2] = np.column_stack([1 - transformer_preds, transformer_preds])
        
        # Get bagging predictions
        for i, model in enumerate(self.bagging_models):
            preds = model.predict_proba(X)
            meta_features[:, (i+1)*2:(i+2)*2] = preds
        
        # Get diverse model predictions
        for i, (name, model) in enumerate(self.diverse_models):
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
            else:
                preds = np.column_stack([1 - model.predict(X), model.predict(X)])
            meta_features[:, (i+len(self.bagging_models)+1)*2:(i+len(self.bagging_models)+2)*2] = preds
        
        return meta_features, y

    def _calculate_feature_importance(self, X):
        """Calculate and store feature importance using SHAP values."""
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.meta_learner)
        shap_values = self.explainer.shap_values(X)
        self.performance_metrics['feature_importance'] = {
            'values': np.abs(shap_values).mean(0),
            'features': self.tfidf.get_feature_names_out()
        }
    
    def _check_fairness(self, predictions, labels, protected_attributes):
        """Check model fairness across protected attributes."""
        fairness_metrics = {}
        for attr in protected_attributes:
            # Calculate demographic parity
            pos_rate_0 = np.mean(predictions[protected_attributes[attr] == 0])
            pos_rate_1 = np.mean(predictions[protected_attributes[attr] == 1])
            demo_parity_diff = abs(pos_rate_0 - pos_rate_1)
            
            # Calculate equal opportunity
            tpr_0 = np.mean(predictions[(protected_attributes[attr] == 0) & (labels == 1)])
            tpr_1 = np.mean(predictions[(protected_attributes[attr] == 1) & (labels == 1)])
            eq_opp_diff = abs(tpr_0 - tpr_1)
            
            fairness_metrics[attr] = {
                'demographic_parity_diff': demo_parity_diff,
                'equal_opportunity_diff': eq_opp_diff
            }
        
        return fairness_metrics
    
    def _prepare_background_dataset(self, train_dataset, n_samples=100):
        """Prepare background dataset for SHAP using k-means clustering."""
        from sklearn.cluster import MiniBatchKMeans
        import time
        import torch
        import numpy as np
        
        logger.info(f"Starting background dataset preparation with {n_samples} samples...")
        start_time = time.time()
        
        try:
            # Extract features from training data in batches
            logger.info("Extracting features from training data...")
            logger.info(f"Total samples to process: {len(train_dataset)}")
            
            batch_size = 1000
            all_texts = []
            
            for i in range(0, len(train_dataset), batch_size):
                try:
                    batch = [train_dataset[j] for j in range(i, min(i + batch_size, len(train_dataset)))]
                    # Convert input_ids to text representation
                    batch_texts = []
                    for item in batch:
                        try:
                            # Get input_ids and ensure it's a tensor
                            if isinstance(item, dict) and 'input_ids' in item:
                                input_ids = item['input_ids']
                                if isinstance(input_ids, (list, tuple)):
                                    input_ids = torch.tensor(input_ids)
                                elif not isinstance(input_ids, torch.Tensor):
                                    logger.warning(f"Unexpected input_ids type: {type(input_ids)}")
                                    continue
                                
                                # Move tensor to CPU and convert to numpy
                                if input_ids.device != torch.device('cpu'):
                                    input_ids = input_ids.cpu()
                                
                                # Ensure input_ids is 1D
                                if len(input_ids.shape) > 1:
                                    input_ids = input_ids.flatten()
                                
                                # Convert to numpy and decode
                                input_ids_np = input_ids.numpy()
                                text = self.transformer_model.tokenizer.decode(input_ids_np, skip_special_tokens=True)
                                if text.strip():  # Only add non-empty texts
                                    batch_texts.append(text)
                            else:
                                logger.warning("Item missing input_ids or not in expected format")
                                continue
                            
                        except Exception as e:
                            logger.error(f"Error processing item in batch: {str(e)}")
                            continue
                    
                    all_texts.extend(batch_texts)
                    if (i + batch_size) % 10000 == 0:
                        logger.info(f"Processed {i + batch_size}/{len(train_dataset)} samples...")
                        logger.info(f"Current texts collected: {len(all_texts)}")
                except Exception as e:
                    logger.error(f"Error processing batch starting at {i}: {str(e)}")
                    continue
            
            if not all_texts:
                raise ValueError("No valid texts extracted from the dataset")
            
            logger.info(f"Successfully extracted {len(all_texts)} texts")
            
            # Fit TF-IDF in batches
            logger.info("Fitting TF-IDF vectorizer...")
            try:
                X = self.tfidf.fit_transform(all_texts)
                logger.info(f"TF-IDF features shape: {X.shape}")
            except Exception as e:
                logger.error(f"Error during TF-IDF transformation: {str(e)}")
                raise
            
            # Use MiniBatchKMeans for better memory efficiency
            logger.info("Performing clustering for sample selection...")
            kmeans = MiniBatchKMeans(
                n_clusters=min(n_samples, len(all_texts)),
                random_state=42,
                batch_size=1000,
                init='k-means++',
                max_iter=100
            )
            
            # Fit in batches to avoid memory issues
            batch_size = min(10000, X.shape[0])
            for i in range(0, X.shape[0], batch_size):
                end_idx = min(i + batch_size, X.shape[0])
                kmeans.partial_fit(X[i:end_idx].toarray())  # Convert sparse to dense
                if (i + batch_size) % 50000 == 0:
                    logger.info(f"Clustering progress: {i + batch_size}/{X.shape[0]} samples...")
            
            # Get cluster centers and find nearest samples
            logger.info("Finding representative samples...")
            distances = kmeans.transform(X.toarray())  # Convert sparse to dense
            background_indices = []
            
            for cluster_idx in range(kmeans.n_clusters):
                cluster_samples = np.where(kmeans.labels_ == cluster_idx)[0]
                if len(cluster_samples) > 0:
                    # Find sample closest to cluster center
                    closest_idx = cluster_samples[np.argmin(distances[cluster_samples, cluster_idx])]
                    background_indices.append(closest_idx)
                    
                if (cluster_idx + 1) % 20 == 0:
                    logger.info(f"Selected {cluster_idx + 1}/{kmeans.n_clusters} representative samples...")
            
            # Store background dataset
            self.background_dataset = [train_dataset[i] for i in background_indices]
            
            end_time = time.time()
            logger.info(f"Background dataset preparation completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Final background dataset size: {len(self.background_dataset)} samples")
            
            # Validate background dataset
            if len(self.background_dataset) < n_samples * 0.9:  # Allow for some missing clusters
                logger.warning(f"Background dataset smaller than expected: {len(self.background_dataset)} < {n_samples}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in background dataset preparation: {str(e)}")
            logger.info("Falling back to random sampling...")
            
            # Fallback to random sampling
            try:
                indices = np.random.choice(len(train_dataset), size=min(n_samples, len(train_dataset)), replace=False)
                self.background_dataset = [train_dataset[i] for i in indices]
                logger.info(f"Successfully created background dataset using random sampling with {len(self.background_dataset)} samples")
                return True
            except Exception as e:
                logger.error(f"Failed to create background dataset: {str(e)}")
                return False
    
    def explain_prediction(self, text: str, cache: bool = True) -> Dict[str, Any]:
        """Generate comprehensive SHAP explanation for a prediction."""
        # Check cache first
        if cache and text in self.explanation_cache:
            return self.explanation_cache[text]
        
        # Get predictions and prepare input
        transformer_encoding = self.transformer_model.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        
        tfidf_features = self.tfidf.transform([text])
        
        # Get SHAP values for transformer
        if not hasattr(self, 'transformer_explainer'):
            self.transformer_explainer = shap.DeepExplainer(
                self.transformer_model.model,
                [item['input_ids'] for item in self.background_dataset]
            )
        
        transformer_shap = self.transformer_explainer.shap_values(
            transformer_encoding['input_ids']
        )
        
        # Get SHAP values for XGBoost
        if not hasattr(self, 'xgb_explainer'):
            self.xgb_explainer = shap.TreeExplainer(self.meta_learner)
        
        xgb_shap = self.xgb_explainer.shap_values(tfidf_features)
        
        # Combine explanations
        explanation = {
            'transformer': {
                'contribution': transformer_shap[0] * self.weights[0],
                'tokens': self.transformer_model.tokenizer.convert_ids_to_tokens(
                    transformer_encoding['input_ids'][0]
                )
            },
            'xgb_shap': {
                'contribution': xgb_shap * self.weights[1],
                'features': self.tfidf.get_feature_names_out()
            },
            'prediction': self.predict([text])[0],
            'prediction_proba': self.predict([text], return_proba=True)[0]
        }
        
        # Cache explanation if requested
        if cache:
            if len(self.explanation_cache) >= self.shap_config['cache_size']:
                # Remove oldest entry
                self.explanation_cache.pop(next(iter(self.explanation_cache)))
            self.explanation_cache[text] = explanation
        
        return explanation
    
    def visualize_explanation(self, explanation: Dict[str, Any], output_dir: str = None):
        """Create comprehensive visualization of SHAP explanation."""
        plt.style.use(self.viz_config['plot_style'])
        
        fig = plt.figure(figsize=(15, 20))
        gs = plt.GridSpec(4, 2)
        
        # 1. Token-level contributions (Transformer)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_token_contributions(explanation['transformer'], ax1)
        
        # 2. Feature importance (XGBoost)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_feature_importance(explanation['xgb_shap'], ax2)
        
        # 3. Sentiment progression
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_sentiment_progression(explanation, ax3)
        
        # 4. Emotion distribution
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_emotion_distribution(explanation, ax4)
        
        # 5. Model confidence
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_model_confidence(explanation, ax5)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'shap_explanation.png'))
            plt.close()
        else:
            return fig
    
    def analyze_fairness_shap(self, test_dataset, demographic_features: List[str]):
        """Analyze fairness using SHAP values across demographic groups."""
        fairness_analysis = {}
        
        for feature in demographic_features:
            # Get unique demographic values
            unique_values = test_dataset[feature].unique()
            
            # Calculate SHAP values for each group
            group_shap = {}
            for value in unique_values:
                group_mask = test_dataset[feature] == value
                group_texts = test_dataset[group_mask]['cleaned_text'].tolist()
                group_explanations = [
                    self.explain_prediction(text, cache=True)
                    for text in group_texts
                ]
                
                # Aggregate SHAP values
                group_shap[value] = {
                    'transformer_shap': np.mean([
                        exp['transformer']['contribution']
                        for exp in group_explanations
                    ], axis=0),
                    'xgb_shap': np.mean([
                        exp['xgb_shap']['contribution']
                        for exp in group_explanations
                    ], axis=0)
                }
            
            # Calculate fairness metrics based on SHAP
            fairness_analysis[feature] = self._calculate_shap_fairness_metrics(group_shap)
        
        return fairness_analysis
    
    def _calculate_shap_fairness_metrics(self, group_shap: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fairness metrics based on SHAP values."""
        metrics = {}
        
        # Calculate SHAP value differences between groups
        groups = list(group_shap.keys())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                
                # Calculate transformer SHAP difference
                transformer_diff = np.mean(np.abs(
                    group_shap[group1]['transformer_shap'] -
                    group_shap[group2]['transformer_shap']
                ))
                
                # Calculate XGBoost SHAP difference
                xgb_diff = np.mean(np.abs(
                    group_shap[group1]['xgb_shap'] -
                    group_shap[group2]['xgb_shap']
                ))
                
                metrics[f'{group1}_vs_{group2}'] = {
                    'transformer_shap_diff': float(transformer_diff),
                    'xgb_shap_diff': float(xgb_diff),
                    'total_shap_diff': float(
                        transformer_diff * self.weights[0] +
                        xgb_diff * self.weights[1]
                    )
                }
        
        return metrics
    
    def train(self, train_dataset, val_dataset, args):
        """Enhanced training with bagging, stacking, and dynamic weight optimization."""
        start_time = time.time()
        logger.info("Starting enhanced ensemble training...")
        
        # Prepare background dataset for SHAP
        logger.info("Preparing background dataset for SHAP...")
        self._prepare_background_dataset(train_dataset, n_samples=self.shap_config['background_size'])
        
        # Train transformer model
        logger.info("Training Transformer model...")
        self.transformer_model.train(train_dataset, val_dataset, args)
        
        # Extract features and prepare data for ensemble
        train_texts = [str(item['input_ids'].numpy()) for item in train_dataset]
        train_labels = [int(item['labels'].numpy()) for item in train_dataset]
        val_texts = [str(item['input_ids'].numpy()) for item in val_dataset]
        val_labels = [int(item['labels'].numpy()) for item in val_dataset]
        
        # Transform texts to TF-IDF features
        X_train = self.tfidf.fit_transform(train_texts)
        X_val = self.tfidf.transform(val_texts)
        
        # Train bagging models with cross-validation
        self._train_bagging_models(X_train, train_labels, X_val, val_labels)
        
        # Train diverse models
        logger.info("Training diverse models...")
        for name, model in self.diverse_models:
            logger.info(f"Training {name}...")
            model.fit(X_train, train_labels)
        
        # Prepare stacking features
        logger.info("Preparing stacking features...")
        X_stack_train, y_stack_train = self._prepare_stacking_features(X_train, train_labels)
        X_stack_val, y_stack_val = self._prepare_stacking_features(X_val, val_labels)
        
        # Train meta-learner
        logger.info("Training meta-learner...")
        self.meta_learner.fit(
            X_stack_train, y_stack_train,
            eval_set=[(X_stack_val, y_stack_val)],
            verbose=50,
            early_stopping_rounds=30
        )
        
        # Optimize ensemble weights
        logger.info("Optimizing ensemble weights...")
        base_models = [self.transformer_model] + self.bagging_models + [model for _, model in self.diverse_models]
        self.weights = self._optimize_weights(base_models, X_val, val_labels)
        
        # Calculate feature importance
        logger.info("Calculating feature importance...")
        self._calculate_feature_importance(X_train)
        
        # Log training summary
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final ensemble weights: {dict(zip(['transformer'] + [f'bagging_{i}' for i in range(len(self.bagging_models))] + [name for name, _ in self.diverse_models], self.weights))}")
        
        # Log cross-validation scores
        for cv_score in self.performance_metrics['cv_scores']:
            logger.info(f"Model {cv_score['model_idx']} CV AUC: {cv_score['mean_cv_score']:.4f} ± {cv_score['std_cv_score']:.4f}")

    def predict(self, texts, return_proba=False):
        """Enhanced prediction with optimized ensemble weights."""
        start_time = time.time()
        
        # Convert texts to proper format if they're from a dataset
        if isinstance(texts[0], dict):
            texts = [str(item['input_ids'].numpy()) for item in texts]
        
        # Get transformer predictions
        transformer_preds = self.transformer_model.predict(texts)
        
        # Get bagging predictions
        X = self.tfidf.transform(texts)
        bagging_preds = np.mean([model.predict_proba(X)[:, 1] for model in self.bagging_models], axis=0)
        
        # Get diverse model predictions
        diverse_preds = []
        for _, model in self.diverse_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            diverse_preds.append(pred)
        
        # Get stacking predictions
        X_stack, _ = self._prepare_stacking_features(X)
        stacking_preds = self.meta_learner.predict_proba(X_stack)[:, 1]
        
        # Combine predictions using optimized weights
        ensemble_preds = (
            self.weights[0] * transformer_preds +
            np.mean(self.weights[1:self.n_bagging_models+1]) * bagging_preds +
            np.mean(self.weights[self.n_bagging_models+1:-1]) * np.mean(diverse_preds, axis=0) +
            self.weights[-1] * stacking_preds
        )
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self.performance_metrics['inference_time'].append(inference_time)
        self.performance_metrics['prediction_distribution'].extend(ensemble_preds.tolist())
        
        # Log prediction statistics
        logger.debug(f"Inference time: {inference_time:.4f} seconds")
        logger.debug(f"Prediction distribution - Mean: {np.mean(ensemble_preds):.4f}, Std: {np.std(ensemble_preds):.4f}")
        
        if return_proba:
            return ensemble_preds
        return (ensemble_preds > 0.5).astype(int)
    
    def evaluate(self, test_dataset, protected_attributes=None):
        """Enhanced evaluation with fairness metrics and detailed performance analysis."""
        # Convert test data to proper format
        test_texts = [str(item['input_ids'].numpy()) for item in test_dataset]
        test_labels = [int(item['labels'].numpy()) for item in test_dataset]
        
        # Get predictions with probabilities
        predictions_proba = self.predict(test_texts, return_proba=True)
        predictions = (predictions_proba > 0.5).astype(int)
        
        # Calculate standard metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='binary'
        )
        
        # Calculate additional metrics
        roc_auc = roc_auc_score(test_labels, predictions_proba)
        
        # Prepare evaluation results
        eval_results = {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
            'eval_roc_auc': roc_auc,
            'eval_inference_time_mean': np.mean(self.performance_metrics['inference_time']),
            'eval_inference_time_std': np.std(self.performance_metrics['inference_time'])
        }
        
        # Add fairness metrics if protected attributes provided
        if protected_attributes is not None:
            fairness_metrics = self._check_fairness(predictions, test_labels, protected_attributes)
            eval_results['fairness_metrics'] = fairness_metrics
        
        return eval_results
    
    def _plot_token_contributions(self, transformer_explanation: Dict[str, Any], ax):
        """Plot token-level contributions from transformer model."""
        contributions = transformer_explanation['contribution']
        tokens = transformer_explanation['tokens']
        
        # Remove padding tokens
        mask = [t not in ['<pad>', '</s>', '<s>'] for t in tokens]
        contributions = contributions[mask]
        tokens = [t for i, t in enumerate(tokens) if mask[i]]
        
        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(contributions))
        top_indices = sorted_indices[-self.shap_config['max_display']:]
        
        # Create horizontal bar plot
        colors = ['red' if c < 0 else 'blue' for c in contributions[top_indices]]
        y_pos = np.arange(len(top_indices))
        
        ax.barh(y_pos, contributions[top_indices], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([tokens[i] for i in top_indices])
        ax.set_title('Top Token Contributions to Sentiment')
        ax.set_xlabel('SHAP value (impact on model output)')
    
    def _plot_feature_importance(self, xgb_explanation: Dict[str, Any], ax):
        """Plot feature importance from XGBoost model."""
        contributions = xgb_explanation['contribution']
        features = xgb_explanation['features']
        
        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(contributions))
        top_indices = sorted_indices[-self.shap_config['max_display']:]
        
        # Create horizontal bar plot
        colors = ['red' if c < 0 else 'blue' for c in contributions[top_indices]]
        y_pos = np.arange(len(top_indices))
        
        ax.barh(y_pos, contributions[top_indices], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([features[i] for i in top_indices])
        ax.set_title('Top TF-IDF Feature Contributions')
        ax.set_xlabel('SHAP value (impact on model output)')
    
    def _plot_sentiment_progression(self, explanation: Dict[str, Any], ax):
        """Plot how sentiment evolves through the text."""
        transformer_contrib = explanation['transformer']['contribution']
        tokens = explanation['transformer']['tokens']
        
        # Remove padding tokens
        mask = [t not in ['<pad>', '</s>', '<s>'] for t in tokens]
        contributions = transformer_contrib[mask]
        tokens = [t for i, t in enumerate(tokens) if mask[i]]
        
        # Calculate cumulative sentiment
        cumulative = np.cumsum(contributions)
        
        # Plot progression
        ax.plot(cumulative, marker='o')
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_title('Sentiment Progression Through Text')
        ax.set_ylabel('Cumulative Sentiment')
        ax.grid(True, alpha=0.3)
    
    def _plot_emotion_distribution(self, explanation: Dict[str, Any], ax):
        """Plot distribution of emotional content based on SHAP values."""
        from textblob import TextBlob
        
        # Get tokens and their contributions
        tokens = explanation['transformer']['tokens']
        contributions = explanation['transformer']['contribution']
        
        # Remove padding tokens
        mask = [t not in ['<pad>', '</s>', '<s>'] for t in tokens]
        contributions = contributions[mask]
        tokens = [t for i, t in enumerate(tokens) if mask[i]]
        
        # Analyze emotion for each token
        emotions = []
        for token in tokens:
            blob = TextBlob(token)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.5:
                emotions.append('joy')
            elif polarity < -0.5:
                emotions.append('sadness')
            elif polarity < -0.2:
                emotions.append('anger')
            elif abs(polarity) < 0.1:
                emotions.append('neutral')
            else:
                emotions.append('fear')
        
        # Count emotions
        emotion_counts = {
            emotion: sum(1 for e in emotions if e == emotion)
            for emotion in self.viz_config['emotion_colors']
        }
        
        # Create pie chart
        colors = [self.viz_config['emotion_colors'][e] for e in emotion_counts.keys()]
        ax.pie(
            emotion_counts.values(),
            labels=emotion_counts.keys(),
            colors=colors,
            autopct='%1.1f%%'
        )
        ax.set_title('Distribution of Emotional Content')
    
    def _plot_model_confidence(self, explanation: Dict[str, Any], ax):
        """Plot model confidence and contribution breakdown."""
        # Get prediction probability
        prob = explanation['prediction_proba']
        
        # Calculate contribution sources
        transformer_contrib = np.sum(np.abs(explanation['transformer']['contribution']))
        xgb_contrib = np.sum(np.abs(explanation['xgb_shap']['contribution']))
        total_contrib = transformer_contrib + xgb_contrib
        
        # Create stacked bar for contribution sources
        contrib_data = [
            (transformer_contrib / total_contrib) * 100,
            (xgb_contrib / total_contrib) * 100
        ]
        
        # Plot confidence bar
        ax.barh(0, prob * 100, height=0.3, color='green', alpha=0.6)
        ax.barh(0, 100, height=0.3, color='gray', alpha=0.2)
        
        # Plot contribution breakdown
        ax.barh(1, contrib_data[0], height=0.3, color='blue', alpha=0.6, label='Transformer')
        ax.barh(1, contrib_data[1], height=0.3, left=contrib_data[0],
                color='red', alpha=0.6, label='XGBoost')
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Confidence', 'Contribution'])
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage')
        ax.set_title('Model Confidence and Contribution Sources')
        ax.legend()
        
        # Add confidence value annotation
        ax.text(prob * 100 + 1, 0, f'{prob*100:.1f}%', va='center')

def main():
    """Main function for Colab environment."""
    try:
        # Set default arguments for Colab
        class Args:
            def __init__(self):
                self.output_dir = '/content/transformer_models/sentiment140'
                self.max_samples = None  # Use full dataset
                self.model_type = 'bertweet'  # Changed to BERTweet
                self.test_size = 0.1  # Reduced test size for more training data
                self.batch_size = 128  # Increased batch size for A100
                self.learning_rate = 2e-5
                self.epochs = 3  # Increased epochs for better learning
                self.max_seq_length = 128
                self.use_full_dataset = True
                self.fairness_evaluation = False
                self.bias_mitigation = False
                self.use_ensemble = True  # Added ensemble flag
        
        args = Args()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        models_dir = output_dir / "models"
        fairness_dir = output_dir / "fairness"
        plots_dir = output_dir / "plots"
        ensemble_dir = output_dir / "ensemble"  # New directory for ensemble model
        
        for directory in [models_dir, fairness_dir, plots_dir, ensemble_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"Starting ensemble-based sentiment analysis with {args.model_type}")
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
        logger.info(f"Class distribution: {df['target'].value_counts().to_dict()}")
        
        # Split dataset
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=42, stratify=df['target']
        )
        
        logger.info(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")
        
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
        logger.info("Initializing ensemble model...")
        ensemble_model = EnsembleSentimentAnalyzer(transformer_model)
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        ensemble_model.train(train_dataset, test_dataset, args)
        
        # Evaluate ensemble model
        logger.info("Evaluating ensemble model...")
        eval_results = ensemble_model.evaluate(test_dataset)
        logger.info(f"Ensemble evaluation results: {eval_results}")
        
        # Save final evaluation metrics
        metrics_file = os.path.join(ensemble_dir, "ensemble_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(eval_results, f, indent=4)
        logger.info(f"Saved ensemble metrics to {metrics_file}")
        
        # Optional: Evaluate fairness if requested
        if args.fairness_evaluation:
            logger.info("Evaluating model fairness...")
            demographic_features = ['age_group', 'gender', 'race', 'education']
            fairness_metrics = evaluate_fairness(ensemble_model, test_df, demographic_features)
            
            # Plot and save fairness metrics
            plot_fairness_metrics(fairness_metrics, fairness_dir)
            
            # Save fairness metrics
            fairness_file = os.path.join(fairness_dir, "fairness_metrics.json")
            with open(fairness_file, "w") as f:
                json.dump(fairness_metrics, f, indent=4)
            logger.info(f"Saved fairness metrics to {fairness_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 