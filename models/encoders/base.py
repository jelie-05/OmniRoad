import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from config.base import EncoderConfig

class BaseEncoder(nn.Module):
    """Base class for all encoder models."""
    
    def __init__(self, config: EncoderConfig):
        """
        Initialize the encoder.
        
        Args:
            config: Encoder configuration
        """
        super().__init__()
        self.config = config
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary with encoded features and metadata
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        
        Returns:
            Output dimension
        """
        return self.config.output_dim