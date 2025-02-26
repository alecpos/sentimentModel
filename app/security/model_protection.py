from typing import Any
from pathlib import Path
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class ModelProtection:
    """Implements security standards from config"""
    
    def __init__(self, key_path: Path):
        self.key_path = key_path
        self.key_path.mkdir(parents=True, exist_ok=True)
        self._init_key()
        
    def _init_key(self) -> None:
        """Initialize or load encryption key"""
        key_file = self.key_path / "model.key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.key)
        self.cipher = Fernet(self.key)
        
    def save_model(self, model: nn.Module, path: Path) -> None:
        """Save encrypted model weights"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state to buffer
        state_dict = model.state_dict()
        buffer = torch.save(state_dict, path)
        
        # Encrypt and save
        try:
            encrypted_data = self.cipher.encrypt(buffer)
            with open(path, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Error encrypting model: {str(e)}")
            raise
            
    def load_model(self, model: nn.Module, path: Path) -> None:
        """Load encrypted model weights"""
        try:
            with open(path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt and load
            buffer = self.cipher.decrypt(encrypted_data)
            state_dict = torch.load(buffer)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Error decrypting model: {str(e)}")
            raise 