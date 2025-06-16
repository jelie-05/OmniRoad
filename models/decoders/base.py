"""Base decoder class definition."""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from config.base import DecoderConfig

class BaseDecoder(nn.Module):
    """Base class for all decoder models."""
    
    def __init__(self, config: DecoderConfig):
        """
        Initialize the decoder.
        
        Args:
            config: Decoder configuration
        """
        super().__init__()
        self.config = config
        
        # if isinstance(config.input_dim, list):
        #     if len(config.input_dim) == 0:
        #         raise ValueError(f"Invalid input dimension: {config.input_dim}")
        # else:
        #     if config.input_dim <= 0:
        #         raise ValueError(f"Invalid input dimension: {config.input_dim}")
        
        if config.num_classes <= 0:
            raise ValueError(f"Invalid output dimension: {config.num_classes}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            features: Dictionary with encoder features and metadata
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward")