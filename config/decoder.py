from dataclasses import dataclass, field
from typing import List, Optional

from .base import DecoderConfig
from .registry import ConfigRegistry

# Create a registry for probing head configurations
DecoderRegistry = ConfigRegistry[DecoderConfig]("DecoderRegistry")

@DecoderRegistry.register("mask2former_head")
@dataclass
class Mask2FormerHeadConfig(DecoderConfig):
    """mask2former head."""
    name: str = 'mask2former_head'
    input_dim: int = -1
    num_classes: int = -1
    in_channels: List[int] = field(default_factory=lambda: [384, 384, 384, 384])
    embed_dim: int = 512
    encoder_name: str = None
    dropout: float = 0.1

@DecoderRegistry.register("segformer_head")
@dataclass
class SegFormerHeadConfig(DecoderConfig):
    """SegFormer head."""
    name: str = 'segformer_head'
    input_dim: int = -1
    num_classes: int = -1
    in_channels: List[int] = field(default_factory=lambda: [384, 384, 384, 384])
    embed_dim: int = 512
    encoder_name: str = None
    dropout: float = 0.1

@DecoderRegistry.register("linear_probing")
@dataclass
class LinearProbingConfig(DecoderConfig):
    """Linear probing head."""
    name: str = 'linear_probing'
    input_dim: int = -1
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    activation: str = "relu"
    num_classes: int = -1
    # spatial_size: int = -1