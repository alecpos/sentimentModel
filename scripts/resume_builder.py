#!/usr/bin/env python
"""Resume Builder using Transformer-based Model."""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, ProgressCallback
)
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import json
import re
from tqdm import tqdm
import argparse
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.ml.prediction.resume_builder import ResumeBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ResumeSection:
    """Data class for resume sections."""
    title: str
    content: str
    order: int

class ResumeDataset(Dataset):
    """Custom dataset for resume generation."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        is_training: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input text (job experience)
        input_text = self._prepare_input_text(item)
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare target text (resume)
        if self.is_training:
            target_text = self._prepare_target_text(item)
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': input_encoding['input_ids'].flatten(),
                'attention_mask': input_encoding['attention_mask'].flatten(),
                'labels': target_encoding['input_ids'].flatten()
            }
        else:
            return {
                'input_ids': input_encoding['input_ids'].flatten(),
                'attention_mask': input_encoding['attention_mask'].flatten()
            }
    
    def _prepare_input_text(self, item: Dict[str, Any]) -> str:
        """Prepare input text from job experience."""
        sections = []
        
        # Add work experience
        if 'work_experience' in item:
            sections.append("Work Experience:")
            for exp in item['work_experience']:
                sections.append(f"- {exp['title']} at {exp['company']}")
                sections.append(f"  Duration: {exp['start_date']} to {exp['end_date']}")
                sections.append(f"  Description: {exp['description']}")
        
        # Add education
        if 'education' in item:
            sections.append("\nEducation:")
            for edu in item['education']:
                sections.append(f"- {edu['degree']} from {edu['institution']}")
                sections.append(f"  Duration: {edu['start_date']} to {edu['end_date']}")
        
        # Add skills
        if 'skills' in item:
            sections.append("\nSkills:")
            sections.append(", ".join(item['skills']))
        
        return "\n".join(sections)
    
    def _prepare_target_text(self, item: Dict[str, Any]) -> str:
        """Prepare target text (resume) from structured data."""
        sections = []
        
        # Add personal info
        if 'personal_info' in item:
            sections.append("Personal Information:")
            for key, value in item['personal_info'].items():
                sections.append(f"{key}: {value}")
        
        # Add professional summary
        if 'summary' in item:
            sections.append("\nProfessional Summary:")
            sections.append(item['summary'])
        
        # Add work experience
        if 'work_experience' in item:
            sections.append("\nProfessional Experience:")
            for exp in item['work_experience']:
                sections.append(f"{exp['title']} at {exp['company']}")
                sections.append(f"{exp['start_date']} - {exp['end_date']}")
                sections.append(exp['description'])
        
        # Add education
        if 'education' in item:
            sections.append("\nEducation:")
            for edu in item['education']:
                sections.append(f"{edu['degree']}")
                sections.append(f"{edu['institution']}")
                sections.append(f"{edu['start_date']} - {edu['end_date']}")
        
        # Add skills
        if 'skills' in item:
            sections.append("\nSkills:")
            sections.append(", ".join(item['skills']))
        
        return "\n".join(sections)

def load_resume_dataset():
    """Load and preprocess the resume dataset."""
    logger.info("Loading resume dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("brackozi/Resume")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Preprocess the data
    processed_data = []
    for _, row in df.iterrows():
        processed_item = {
            'personal_info': {
                'name': row.get('name', ''),
                'email': row.get('email', ''),
                'phone': row.get('phone', ''),
                'location': row.get('location', '')
            },
            'summary': row.get('summary', ''),
            'work_experience': row.get('work_experience', []),
            'education': row.get('education', []),
            'skills': row.get('skills', [])
        }
        processed_data.append(processed_item)
    
    logger.info(f"Loaded {len(processed_data)} resume samples")
    return processed_data

def train_resume_builder(
    model_name: str = "t5-base",
    output_dir: str = "models/resume_builder",
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    max_length: int = 512
):
    """Train the resume builder model."""
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        data = load_resume_dataset()
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create datasets
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        train_dataset = ResumeDataset(
            train_data,
            tokenizer,
            max_length=max_length,
            is_training=True
        )
        
        val_dataset = ResumeDataset(
            val_data,
            tokenizer,
            max_length=max_length,
            is_training=True
        )
        
        # Initialize model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine_with_restarts",
            report_to=["tensorboard"]
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                ProgressCallback()
            ]
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model and tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False

def generate_resume(
    model_path: str,
    job_experience: Dict[str, Any],
    output_file: Optional[str] = None
) -> str:
    """Generate a resume from job experience."""
    try:
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare input
        input_text = ResumeDataset._prepare_input_text(None, job_experience)
        input_encoding = tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate resume
        outputs = model.generate(
            input_encoding['input_ids'],
            max_length=1024,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            temperature=0.7
        )
        
        # Decode output
        resume_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(resume_text)
            logger.info(f"Resume saved to {output_file}")
        
        return resume_text
        
    except Exception as e:
        logger.error(f"Error generating resume: {str(e)}")
        return None

def main():
    """Main function for resume builder."""
    parser = argparse.ArgumentParser(description='AI Resume Builder')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate'],
                      help='Mode to run the script in: train or generate')
    parser.add_argument('--model_path', type=str, default='models/resume_builder',
                      help='Path to save/load the model')
    parser.add_argument('--output_file', type=str, default='output/resume.json',
                      help='Path to save the generated resume')
    parser.add_argument('--job_experience', type=str, default='data/job_experience.json',
                      help='Path to job experience JSON file')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use for training')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    try:
        if args.mode == 'train':
            logger.info("Starting resume builder training...")
            train_resume_builder(
                model_path=args.model_path,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                max_samples=args.max_samples
            )
        elif args.mode == 'generate':
            logger.info("Generating resume...")
            generate_resume(
                model_path=args.model_path,
                job_experience=args.job_experience,
                output_file=args.output_file
            )
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 