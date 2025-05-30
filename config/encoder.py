# src/config/model_configs.py
from dataclasses import dataclass, field
from typing import List, Optional

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
    attention_heads: int = 6
    
    def __post_init__(self):
        self.name = "dino_vits8"
        self.output_dim = 384
        self.patch_size = 8
        self.attention_heads = 6

@EncoderRegistry.register("lora_dino_vits8")
@dataclass
class LoRADinoViTS8Config(EncoderConfig):
    """Configuration for LoRa DINO ViT-S/8 model."""
    name: str = "lora_dino_vits8"
    freeze: bool = True
    output_dim: int = 384
    patch_size: int = 8
    attention_heads: int = 6
    lora: bool = True
    r: int = 16
    lora_alpha: int = 32
    enable_lora: List[bool] = field(default_factory=lambda: [True, False, True])
    
    def __post_init__(self):
        self.name = "lora_dino_vits8"
        self.output_dim = 384
        self.patch_size = 8
        self.attention_heads = 6

@EncoderRegistry.register("dinov2_vits14")
@dataclass
class Dinov2ViTS14Config(EncoderConfig):
    """Configuration for DINOv2 ViT-S/14 model."""
    name: str = "dinov2_vits14"
    freeze: bool = True
    output_dim: int = 384
    patch_size: int = 14
    attention_heads: int = 6

    def __post_init__(self):
        # Ensure name is correct
        self.name = "dinov2_vits14"
        self.output_dim = 384
        self.patch_size = 14
        self.attention_heads = 6

@EncoderRegistry.register("clip_vitb16")
@dataclass
class CLIPViTB16Config(EncoderConfig):
    """Configuration for CLIP ViT-B/16 model."""
    name: str = "clip_vitb16"
    freeze: bool = True
    output_dim: int = 768
    patch_size: int = 16
    attention_heads: int = 12

    def __post_init__(self):
        # Ensure name is correct
        self.name = "clip_vitb16"
        self.output_dim = 768
        self.patch_size = 16
        self.attention_heads = 12

