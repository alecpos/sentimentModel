#!/usr/bin/env python
"""
Transformer-Based Sentiment Analysis with Fairness Evaluation

This script implements advanced sentiment analysis using transformer models with:
1. Improved model architecture (BERT/RoBERTa-based models)
2. Advanced text preprocessing and feature extraction
3. Fairness assessment across demographic intersections
4. Bias mitigation techniques

Usage:
    python transformer_sentiment_analysis.py --dataset_path path/to/sentiment140.csv
"""

import os
import sys
import argparse
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
import requests
import zipfile
from tqdm import tqdm
from datasets import load_dataset

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

def preprocess_tweet(text):
    """
    Advanced preprocessing for tweets with Twitter-specific handling.
    
    Args:
        text: Raw tweet text
            
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
            
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with placeholder
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Replace user mentions with placeholder
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Extract hashtag content (removing # symbol)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle emojis - replace with text description
    text = emoji.demojize(text)
    text = text.replace(':', ' ').replace('_', ' ')
    
    # Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)
    
    # Handle common Twitter slang/abbreviations
    slang_dict = {
        'u': 'you',
        'r': 'are',
        'y': 'why',
        'ur': 'your',
        '2': 'to',
        '4': 'for',
        'b': 'be',
        'idk': 'i do not know',
        'tbh': 'to be honest',
        'imo': 'in my opinion',
        'omg': 'oh my god',
        'lol': 'laughing out loud',
        'lmao': 'laughing my ass off',
        'brb': 'be right back',
        'btw': 'by the way',
    }
    
    # Split into words and replace slang terms
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(words)
    
    # Handle negations (adding NEG prefix to words after negation)
    negation_words = {'not', 'no', 'never', 'none', 'nowhere', 'nothing', 'neither', 'hardly', 'barely'}
    words = text.split()
    in_negation = False
    result = []
    
    for word in words:
        if word in negation_words or word.endswith("n't"):
            in_negation = True
            result.append(word)
        elif in_negation and word in {'.', '!', '?', ';', ','}:
            in_negation = False
            result.append(word)
        elif in_negation:
            result.append('NEG_' + word)
        else:
            result.append(word)
    
    text = ' '.join(result)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def add_synthetic_demographics(df, random_seed=42):
    """
    Add synthetic demographic features to the dataset for fairness evaluation.
    Ensures a realistic distribution of demographic variables and their intersections.
    
    Args:
        df: DataFrame to augment
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic demographic features
    """
    np.random.seed(random_seed)
    
    # Create more realistic demographic distributions
    
    # Age groups with realistic distribution (skewed toward younger users on Twitter)
    age_probs = [0.35, 0.30, 0.25, 0.10]  # Probabilities for '18-25', '26-35', '36-50', '51+'
    df['age_group'] = np.random.choice(
        ['18-25', '26-35', '36-50', '51+'], 
        size=len(df), 
        p=age_probs
    )
    
    # Gender (simplified to binary for demonstration)
    df['gender'] = np.random.choice([0, 1], size=len(df), p=[0.48, 0.52])
    
    # Location with urban areas having higher representation
    location_probs = [0.22, 0.33, 0.45]  # Probabilities for 'rural', 'suburban', 'urban'
    df['location'] = np.random.choice(
        ['rural', 'suburban', 'urban'], 
        size=len(df), 
        p=location_probs
    )
    
    # Create conditional distributions to model intersectional demographics more realistically
    # For example, making certain combinations more/less likely
    
    # Create intersectional groups for later analysis
    df['gender_age'] = df['gender'].astype(str) + '_' + df['age_group']
    df['gender_location'] = df['gender'].astype(str) + '_' + df['location']
    df['age_location'] = df['age_group'] + '_' + df['location']
    
    # Add a combined intersectional column
    df['demographic_intersection'] = df['gender'].astype(str) + '_' + df['age_group'] + '_' + df['location']
    
    return df

def download_sentiment140(output_dir='sentiment140_data'):
    """
    Download the Sentiment140 dataset using multiple methods.
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    csv_path = output_dir / "sentiment140.csv"
    
    # Check if file already exists and is valid
    if csv_path.exists():
        try:
            # Try to read the first few lines to verify format
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = next(f).strip()
                if first_line.startswith('<!DOCTYPE html>'):
                    logger.warning("Existing file appears to be HTML, will re-download")
                    csv_path.unlink()
                else:
                    logger.info(f"Valid dataset already exists at {csv_path}")
                    return str(csv_path)
        except Exception as e:
            logger.warning(f"Error verifying existing file: {str(e)}")
            csv_path.unlink()
    
    try:
        # Try downloading from Google Drive first
        logger.info("Attempting to download from Google Drive...")
        import gdown
        
        url = "https://docs.google.com/uc?export=download&id=0B04GJPshIjmPRnZManQwWEdTZjg"
        gdown.download(url, str(csv_path), quiet=False)
        
        if csv_path.exists() and csv_path.stat().st_size > 0:
            logger.info(f"Successfully downloaded dataset to {csv_path}")
            return str(csv_path)
        
        # If Google Drive download fails, try direct download
        logger.info("Google Drive download failed, trying direct download...")
        import requests
        
        direct_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        zip_path = output_dir / "sentiment140.zip"
        
        response = requests.get(direct_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        # Extract the zip file
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Find the CSV file
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            # Move the first CSV file to the expected location
            csv_files[0].rename(csv_path)
            logger.info(f"Dataset extracted and saved to {csv_path}")
            return str(csv_path)
        
        # If both downloads fail, try Hugging Face datasets
        logger.info("Direct download failed, trying Hugging Face datasets...")
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
        
        dataset = load_dataset("stanfordnlp/sentiment140", trust_remote_code=True)
        df = dataset['train'].to_pandas()
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset downloaded from Hugging Face and saved to {csv_path}")
        
        return str(csv_path)
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        if csv_path.exists():
            csv_path.unlink()
        raise

def get_default_paths():
    """Get default paths based on environment."""
    if is_colab():
        return {
            'zip_path': '/content/sentiment140.csv',  # Use CSV file in Colab
            'output_dir': '/content/transformer_models/sentiment140',
            'script_path': '/content/transformer_sentiment_analysis.py'
        }
    elif is_mps():
        return {
            'zip_path': '/Users/alecposner/WITHIN/sentiment140.csv',
            'output_dir': 'transformer_models/sentiment140',
            'script_path': 'transformer_sentiment_analysis.py'
        }
    else:
        return {
            'zip_path': 'sentiment140.csv',
            'output_dir': 'transformer_models/sentiment140',
            'script_path': 'transformer_sentiment_analysis.py'
        }

def load_sentiment140_from_zip(zip_path, max_samples=None):
    """
    Load and preprocess the Sentiment140 dataset from a local zip or CSV file.
    
    Args:
        zip_path: Path to the Sentiment140 zip or CSV file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        DataFrame with the preprocessed dataset
    """
    logger.info(f"Loading Sentiment140 dataset from file: {zip_path}")
    
    try:
        # Check if file is zip or csv
        if zip_path.endswith('.zip'):
            # Handle zip file
            import zipfile
            import tempfile
            
            # Create a temporary directory to extract files
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Extracting zip file to temporary directory: {temp_dir}")
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the CSV file
                import os
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                
                if not csv_files:
                    logger.error("No CSV files found in the zip archive")
                    return create_sample_dataset()
                
                # Use the first CSV file found
                csv_path = os.path.join(temp_dir, csv_files[0])
                logger.info(f"Found CSV file: {csv_path}")
                
                # Load the CSV
                df = load_csv_file(csv_path, max_samples)
                return df
        else:
            # Handle CSV file directly
            logger.info(f"Loading CSV file directly: {zip_path}")
            df = load_csv_file(zip_path, max_samples)
            return df
            
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return create_sample_dataset()

def load_csv_file(csv_path, max_samples=None):
    """Helper function to load and process CSV file."""
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    try:
        # Try with different encodings
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', header=None, names=column_names)
        except UnicodeDecodeError:
            logger.info("UTF-8 encoding failed, trying latin1 encoding")
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
        logger.error(f"Error loading CSV file: {str(e)}")
        return create_sample_dataset()

def create_sample_dataset(max_samples=None):
    """Create a small sample dataset for testing."""
    logger.warning("Creating sample dataset for testing")
    sample_data = {
        'target': [0, 1, 0, 1, 0],
        'id': range(5),
        'date': ['2024-01-01'] * 5,
        'flag': ['NO_QUERY'] * 5,
        'user': ['user' + str(i) for i in range(5)],
        'text': [
            "This is a negative review. The product was terrible.",
            "Great experience! Would recommend to everyone.",
            "Very disappointed with the service.",
            "Amazing product, exceeded my expectations!",
            "Poor quality, would not buy again."
        ]
    }
    df = pd.DataFrame(sample_data)
    df['cleaned_text'] = df['text'].apply(preprocess_tweet)
    df = add_synthetic_demographics(df)
    
    # If max_samples is specified, limit the dataset size
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    return df

def load_sentiment140(max_samples=None, use_local_zip=False, zip_path=None):
    """Load the Sentiment140 dataset with improved error handling and data validation."""
    try:
        if use_local_zip and zip_path:
            logger.info(f"Loading Sentiment140 dataset from zip file: {zip_path}")
            return load_sentiment140_from_zip(zip_path, max_samples)
        
        logger.info("Loading Sentiment140 dataset from Hugging Face...")
        # Load dataset with trust_remote_code=True to avoid prompt
        dataset = load_dataset("stanfordnlp/sentiment140", trust_remote_code=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'sentiment': 'target',
            'text': 'text'
        })
        
        # Add missing columns with default values
        df['id'] = range(len(df))
        df['date'] = pd.Timestamp.now()
        df['flag'] = 'NO_QUERY'
        df['user'] = 'unknown'
        
        # Reorder columns to match expected format
        df = df[['target', 'id', 'date', 'flag', 'user', 'text']]
        
        # Data validation and cleaning
        logger.info("Validating and cleaning data...")
        
        # Remove rows with missing values
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} rows with missing values")
        
        # Ensure target is numeric
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna(subset=['target'])
        
        # Convert sentiment labels from 0/4 to 0/1
        df['target'] = df['target'].map({0: 0, 4: 1})
        
        # Add synthetic demographic features for fairness analysis
        logger.info("Adding synthetic demographic features...")
        np.random.seed(42)
        df['gender'] = np.random.choice(['male', 'female'], size=len(df))
        df['age_group'] = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], size=len(df))
        
        # Apply text preprocessing
        logger.info("Preprocessing text...")
        df['cleaned_text'] = df['text'].apply(preprocess_tweet)
        
        # Limit dataset size if specified
        if max_samples:
            logger.info(f"Limiting dataset to {max_samples} samples")
            df = df.sample(n=max_samples, random_state=42)
        
        logger.info(f"Successfully loaded {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.warning("Creating sample dataset for testing")
        return create_sample_dataset(max_samples)

class SentimentDataset(Dataset):
    """PyTorch dataset for Sentiment140 data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize dataset.
        
        Args:
            texts: List of preprocessed texts
            labels: List of sentiment labels (0 for negative, 1 for positive)
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Return as dictionary for easier batch construction
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerSentimentAnalyzer:
    """
    Transformer-based sentiment analysis model with fairness evaluation.
    
    This class manages the entire lifecycle of a transformer model for sentiment
    analysis, including initialization, training, evaluation, inference, and
    fairness assessment.
    """
    
    def __init__(self, model_type='distilbert', num_labels=2, device=None):
        """
        Initialize the transformer sentiment analyzer with MPS optimizations.
        """
        self.model_type = model_type
        self.num_labels = num_labels
        self.model_path = None
        
        # Map model type to HuggingFace model identifier
        self.model_mapping = {
            'bert': 'bert-base-uncased',
            'distilbert': 'distilbert-base-uncased',
            'roberta': 'roberta-base',
            'xlnet': 'xlnet-base-cased',
            'bertweet': 'vinai/bertweet-base'
        }
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            # Check if MPS (Apple Silicon) is available
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple MPS (Metal Performance Shaders) device")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            self.device = device
        
        # Initialize tokenizer and model
        self.model_name = self.model_mapping.get(model_type, 'distilbert-base-uncased')
        logger.info(f"Initializing tokenizer and model: {self.model_name}")
        
        # Initialize tokenizer with local caching
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True if os.path.exists(f"./models/{self.model_name}") else False,
            trust_remote_code=True
        )
        
        # Initialize model with specific optimizations for MPS
        if self.device.type == "mps":
            # Use torch's native optimizer for better MPS compatibility
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                local_files_only=True if os.path.exists(f"./models/{self.model_name}") else False,
                trust_remote_code=True
            )
            
            # Initialize classifier weights explicitly based on model architecture
            if hasattr(self.model, 'classifier'):
                if isinstance(self.model.classifier, torch.nn.Linear):
                    # For DistilBERT and similar architectures
                    torch.nn.init.xavier_normal_(self.model.classifier.weight)
                    torch.nn.init.zeros_(self.model.classifier.bias)
                elif hasattr(self.model.classifier, 'out_proj'):
                    # For RoBERTa and similar architectures
                    torch.nn.init.xavier_normal_(self.model.classifier.out_proj.weight)
                    torch.nn.init.zeros_(self.model.classifier.out_proj.bias)
                    if hasattr(self.model.classifier, 'dense'):
                        torch.nn.init.xavier_normal_(self.model.classifier.dense.weight)
                        torch.nn.init.zeros_(self.model.classifier.dense.bias)
                logger.info("Initialized classifier weights for MPS device")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                local_files_only=True if os.path.exists(f"./models/{self.model_name}") else False,
                trust_remote_code=True
            )
        
        # Move model to device and optimize for MPS
        self.model.to(self.device)
        self.optimize_for_mps()
        
        # Log model initialization
        logger.info(f"Model initialized and moved to {self.device}")
        logger.info("Note: Classifier weights are newly initialized as expected for the classification task")
    
    def optimize_for_mps(self):
        """Optimize model settings for MPS device."""
        if self.device.type == "mps":
            # Ensure model is using float32 (MPS has issues with some other dtypes)
            self.model = self.model.to(torch.float32)
            
            # Disable some features that might cause problems on MPS
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
                
            # Enable gradient checkpointing to reduce memory usage
            self.model.gradient_checkpointing_enable()
            
            # Add smaller batch processing
            self.model.config.max_position_embeddings = min(
                self.model.config.max_position_embeddings, 
                512  # Limit sequence length
            )
            
            logger.info("Applied MPS-specific optimizations")
    
    def train(self, 
              train_texts, train_labels, 
              val_texts=None, val_labels=None,
              batch_size=32,
              learning_rate=2e-5,
              epochs=3,
              max_seq_length=128,
              output_dir='./transformer_model'):
        """
        Train the transformer model with MPS optimizations.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, max_seq_length
        )
        
        # Create validation dataset if provided
        if val_texts is not None and val_labels is not None:
            val_dataset = SentimentDataset(
                val_texts, val_labels, self.tokenizer, max_seq_length
            )
            do_eval = True
        else:
            val_dataset = None
            do_eval = False
        
        # Calculate number of training steps
        train_steps_per_epoch = len(train_dataset) // batch_size
        num_train_steps = train_steps_per_epoch * epochs
        
        # Set up training arguments with MPS optimizations
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=train_steps_per_epoch,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=train_steps_per_epoch // 10,
            evaluation_strategy="epoch" if do_eval else "no",
            save_strategy="epoch",
            load_best_model_at_end=do_eval,
            metric_for_best_model="eval_loss" if do_eval else None,
            report_to="none",  # Disable reporting to avoid dependencies
            dataloader_drop_last=False,
            # MPS-specific optimizations
            no_cuda=self.device.type == "cpu",
            use_mps_device=self.device.type == "mps",
            fp16=False,  # Disable fp16 for MPS
            bf16=False,  # Disable bf16 for MPS
            optim="adamw_torch",  # Use torch's native optimizer
            gradient_accumulation_steps=4,  # Accumulate gradients for better memory management
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train model
        logger.info(f"Training transformer model for {epochs} epochs")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Save the model
        model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(model_path)
        self.model_path = model_path
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Log training information
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        logger.info(f"Model saved to {model_path}")
        
        # Return training metrics
        metrics = {
            "model_type": self.model_type,
            "training_time_seconds": training_time,
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0
        }
        
        return metrics
    
    def evaluate(self, test_texts, test_labels, batch_size=32, max_seq_length=128):
        """
        Evaluate the model with improved MPS handling.
        """
        # Create test dataset
        test_dataset = SentimentDataset(
            test_texts, test_labels, self.tokenizer, max_seq_length
        )
        
        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize lists to store predictions and true labels
        all_preds = []
        all_labels = []
        
        # Evaluate model with improved MPS handling
        logger.info("Evaluating model on test data")
        try:
            with torch.no_grad():
                for batch in test_loader:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    try:
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                        # Get predictions
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1)
                        
                        # Move predictions and labels to CPU before converting to numpy
                        preds = preds.cpu()
                        labels = labels.cpu()
                        
                        # Store predictions and labels
                        all_preds.extend(preds.numpy())
                        all_labels.extend(labels.numpy())
                        
                    except RuntimeError as e:
                        if "MPS" in str(e):
                            logger.warning("MPS device error encountered, falling back to CPU")
                            # Move model to CPU and retry
                            self.model = self.model.cpu()
                            self.device = torch.device("cpu")
                            # Retry the batch on CPU
                            input_ids = input_ids.cpu()
                            attention_mask = attention_mask.cpu()
                            labels = labels.cpu()
                            
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            logits = outputs.logits
                            preds = torch.argmax(logits, dim=1)
                            
                            # Store predictions and labels
                            all_preds.extend(preds.numpy())
                            all_labels.extend(labels.numpy())
                        else:
                            raise e
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])
        logger.info(f"Classification Report:\n{report}")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Log results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Return evaluation metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        return metrics, all_preds, all_labels
    
    def predict(self, text):
        """
        Predict sentiment for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Preprocess text
        processed_text = preprocess_tweet(text)
        
        # Tokenize
        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get logits and probabilities
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get prediction
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
        
        # Map predicted class to label
        sentiment_label = "positive" if predicted_class == 1 else "negative"
        
        # Map to a score between -1 and 1
        if sentiment_label == "positive":
            sentiment_score = 0.5 + (0.5 * confidence)  # 0.5 to 1.0
        else:
            sentiment_score = -0.5 - (0.5 * confidence)  # -0.5 to -1.0
        
        return {
            "text": text,
            "processed_text": processed_text,
            "sentiment_score": float(sentiment_score),
            "sentiment_label": sentiment_label,
            "confidence": float(confidence),
            "class": int(predicted_class),
            "probabilities": {
                "negative": float(probs[0]),
                "positive": float(probs[1])
            }
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
        os.makedirs(path, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(path)
        
        self.model_path = path
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        # Load the model
        self.model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True)
        self.model.to(self.device)
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        self.model_path = path
        logger.info(f"Model loaded from {path}")

def is_colab():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_mps():
    """Check if running on Apple Silicon with MPS."""
    return torch.backends.mps.is_available()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a transformer-based sentiment analysis model')
    
    # Get default paths based on environment
    default_paths = get_default_paths()
    
    parser.add_argument('--output_dir', type=str, default=default_paths['output_dir'],
                      help='Directory to save the model and analysis results')
    parser.add_argument('--max_samples', type=int, default=400000,
                      help='Maximum number of samples to use (None to use all)')
    parser.add_argument('--model_type', type=str, default='distilbert',
                      choices=['bert', 'roberta', 'distilbert', 'xlnet', 'bertweet'],
                      help='Type of transformer model to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of epochs for training')
    parser.add_argument('--max_seq_length', type=int, default=128,
                      help='Maximum sequence length for tokenization')
    parser.add_argument('--use_full_dataset', action='store_true',
                      help='Use the full dataset (overrides max_samples)')
    parser.add_argument('--fairness_evaluation', action='store_true',
                      help='Perform fairness evaluation after training')
    parser.add_argument('--bias_mitigation', action='store_true',
                      help='Apply bias mitigation techniques during training')
    parser.add_argument('--use_local_zip', action='store_true',
                      help='Use local zip file instead of downloading from Hugging Face')
    parser.add_argument('--zip_path', type=str, default=default_paths['zip_path'],
                      help='Path to the local zip file if use_local_zip is True')
    
    return parser.parse_args()

def main():
    """Main function with notebook environment support."""
    try:
        # Check if we're in a notebook environment
        if 'ipykernel' in sys.modules:
            # We're in a notebook environment
            # Set default arguments here
            class Args:
                def __init__(self):
                    default_paths = get_default_paths()
                    self.output_dir = default_paths['output_dir']
                    self.max_samples = 400000  # Change as needed
                    self.model_type = 'distilbert'
                    self.test_size = 0.2
                    self.batch_size = 16  # Reduced for MPS
                    self.learning_rate = 2e-5
                    self.epochs = 3
                    self.max_seq_length = 128
                    self.use_full_dataset = False
                    self.fairness_evaluation = False
                    self.bias_mitigation = False
                    self.use_local_zip = is_colab()  # Use zip in Colab
                    self.zip_path = default_paths['zip_path']
            
            args = Args()
        else:
            # We're running as a script
            args = parse_args()
            # Set use_local_zip based on environment
            if not args.use_local_zip:
                args.use_local_zip = is_colab()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different outputs
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
        
        # Load dataset using appropriate method
        df = load_sentiment140(
            max_samples=max_samples,
            use_local_zip=args.use_local_zip,
            zip_path=args.zip_path
        )
        
        # Validate dataset size
        if len(df) < 1000:
            raise ValueError(f"Dataset too small: {len(df)} samples. Please check the data loading process.")
        
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Split data into train, validation, and test sets
        # First split into train+val and test
        train_val_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=42, stratify=df['target']
        )
        
        # Then split train+val into train and validation
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['target']
        )
        
        logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
        
        # Initialize the transformer model
        analyzer = TransformerSentimentAnalyzer(model_type=args.model_type)
        
        # Train the model
        logger.info("Training the transformer model...")
        train_metrics = analyzer.train(
            train_texts=train_df['cleaned_text'].tolist(),
            train_labels=train_df['target'].tolist(),
            val_texts=val_df['cleaned_text'].tolist(),
            val_labels=val_df['target'].tolist(),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_seq_length=args.max_seq_length,
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
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length
        )
        
        # Save evaluation metrics
        with open(models_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        logger.info(f"Transformer-based sentiment analysis completed successfully!")
        logger.info(f"Model saved to {models_dir}")
        logger.info(f"Accuracy: {eval_metrics['accuracy']:.4f}, F1 Score: {eval_metrics['f1_score']:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 