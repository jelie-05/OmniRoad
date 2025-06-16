"""Complete end-to-end model with encoder and decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from config.base import ModelConfig
from .encoders import BaseEncoder, create_encoder
from .decoders import BaseDecoder, create_decoder

class EncoderDecoderModel(nn.Module):
    """Full model combining an encoder and decoder."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the complete model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Create encoder
        self.encoder = create_encoder(config.encoder)
        
        # Ensure decoder input_dim matches encoder output_dim
        
        # assert config.decoder.input_dim == self.encoder.get_output_dim(), f"Input dimenstion of the decoder ({config.decoder.input_dim}) doesn't match the output dimension of the encoder ({self.encoder.get_output_dim()})."
        
        # Create decoder
        self.decoder = create_decoder(config.decoder)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor, shape depends on the task
        """
        # Encode features
        features = self.encoder(x)
        
        # Decode features
        logits = self.decoder(features)
        
        return logits
    
    def get_encoder_output(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get raw encoder output.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoder output dictionary
        """
        return self.encoder(x)