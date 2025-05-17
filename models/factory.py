from typing import Dict, Type, Optional

import torch.nn as nn

from config.base import Config, ModelConfig
from .full_model import EncoderDecoderModel

def create_model(config: Config) -> EncoderDecoderModel:
    """
    Create a complete model from configuration.
    
    Args:
        config: Complete configuration
        
    Returns:
        Initialized model
    """
    # Ensure decoder output_dim is set correctly based on task
    if hasattr(config.data, 'num_classes') and config.model.decoder.num_classes <= 0:
        config.model.decoder.num_classes = config.data.num_classes
    
    # Create the model
    return EncoderDecoderModel(config.model)