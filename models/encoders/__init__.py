"""Encoder models package."""
from typing import Dict, Type

import torch.nn as nn

from config.base import EncoderConfig
from .base import BaseEncoder
from .dinov1_wrapper import DinoViTWrapper
from .dinov2_wrapper import Dinov2ViTWrapper
from .clip_wrapper import CLIPViTWrapper


# Registry of encoder implementations
ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {
    'dino_vits8': DinoViTWrapper,
    'lora_dino_vits8': DinoViTWrapper,
    'dinov2_vits14': Dinov2ViTWrapper,
    'clip_vitb16': CLIPViTWrapper
}

def create_encoder(config: EncoderConfig) -> BaseEncoder:
    """
    Create an encoder model from configuration.
    
    Args:
        config: Encoder configuration
        
    Returns:
        Initialized encoder model
    """
    encoder_class = ENCODER_REGISTRY.get(config.name)
    if encoder_class is None:
        raise ValueError(f"Unknown encoder: {config.name}")
    
    return encoder_class(config)

__all__ = ['BaseEncoder', 'DinoViTWrapper', 'Dinov2ViTWrapper', 'CLIPViTWrapper', 'create_encoder']