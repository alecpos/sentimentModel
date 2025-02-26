from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.models.ml.prediction.ad_score_predictor import AdPredictorNN
from app.models.ml.prediction.base import BaseMLModel
from app.models.ml.prediction.training import ModelTrainer

def prepare_data() -> Tuple[DataLoader, DataLoader]:
    """Prepare example data loaders"""
    # Create synthetic data for example
    X_train = torch.randn(1000, 256)
    y_train = torch.randint(0, 2, (1000, 1)).float()
    X_val = torch.randn(200, 256)
    y_val = torch.randint(0, 2, (200, 1)).float()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    
    return train_loader, val_loader

def train_model():
    """Example of training a model with project standards"""
    # Prepare data
    train_loader, val_loader = prepare_data()
    
    # Initialize model
    model = AdPredictorNN(
        input_dim=256,
        enable_quantum_noise=True
    )
    
    # Create trainer with project standards
    trainer = ModelTrainer(
        model,
        config={
            'learning_rate': 0.001,
            'batch_size': 32,
            'gradient_accumulation_steps': 4
        }
    )
    
    # Train model
    metrics = trainer.train(train_loader, val_loader)
    print(f"Training metrics: {metrics}")

if __name__ == "__main__":
    train_model() 