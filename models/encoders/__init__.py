"""Encoder models package."""
from typing import Dict, Type

import torch.nn as nn

from config.base import EncoderConfig
from .base import BaseEncoder
from .dinov1_wrapper import DinoViTWrapper
from .dinov2_wrapper import Dinov2ViTWrapper
from .clip_wrapper import CLIPViTWrapper
from .swinv2_wrapper import SwinTransformerWrapper
from .vit_adapter import ViTAdapter
# Registry of encoder implementations
ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {
    'dino_vits8': DinoViTWrapper,
    'lora_dino_vits8': DinoViTWrapper,
    'dinov2_vits14': Dinov2ViTWrapper,
    'lora_dinov2_vits14': Dinov2ViTWrapper,
    'clip_vitb16': CLIPViTWrapper,
    'swinv2_tiny_window8_256': SwinTransformerWrapper,
    'swinv2_tiny_window16_256': SwinTransformerWrapper,
    'swinv2_small_window8_256': SwinTransformerWrapper,
    'swinv2_small_window16_256': SwinTransformerWrapper,
    'swinv2_base_window8_256': SwinTransformerWrapper,
    'swinv2_base_window16_256': SwinTransformerWrapper,
    'lora_swinv2_tiny_window8_256': SwinTransformerWrapper,
    'lora_swinv2_small_window8_256': SwinTransformerWrapper,
    'lora_swinv2_base_window8_256': SwinTransformerWrapper,
    'vit_adapter_dinov2_vits14': ViTAdapter,
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

__all__ = ['BaseEncoder', 'DinoViTWrapper', 'Dinov2ViTWrapper', 'CLIPViTWrapper', 'SwinTransformerWrapper', 'ViTAdapter', 'create_encoder']