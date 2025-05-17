# src/config/model_configs.py
from dataclasses import dataclass, field

from .base import EncoderConfig
from .registry import ConfigRegistry

# Create a registry specifically for model configurations
EncoderRegistry = ConfigRegistry[EncoderConfig]("EncoderRegistry")

@EncoderRegistry.register("dino_vits8")
@dataclass
class DinoViTS8Config(EncoderConfig):
    """Configuration for DINO ViT-S/8 model."""
    name: str = "dino_vits8"
    freeze: bool = True
    output_dim: int = 384
    patch_size: int = 8
    spatial_size: int = 28
    attention_heads: int = 6
    
    def __post_init__(self):
        # Ensure name is correct
        self.name = "dino_vits8"
        self.output_dim = 384
        self.patch_size = 8
        self.attention_heads = 6
        self.spatial_size = 28
