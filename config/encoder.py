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

@EncoderRegistry.register("lora_dino_vits8")
@dataclass
class LoRADinoViTS8Config(EncoderConfig):
    """Configuration for LoRa DINO ViT-S/8 model."""
    name: str = "lora_dino_vits8"
    freeze: bool = True
    output_dim: int = 384
    patch_size: int = 8
    attention_heads: int = 6
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target: List[str] = field(default_factory=lambda: ['qkv', 'proj'])
    lora_qkv_enable: List[bool] = field(default_factory=lambda: [True, False, True])

@EncoderRegistry.register("dinov2_vits14")
@dataclass
class Dinov2ViTS14Config(EncoderConfig):
    """Configuration for DINOv2 ViT-S/14 model."""
    name: str = "dinov2_vits14"
    freeze: bool = True
    output_dim: int = 384
    patch_size: int = 14
    attention_heads: int = 6
    
@EncoderRegistry.register("lora_dinov2_vits14")
@dataclass
class LoRADinov2ViTS14Config(EncoderConfig):
    """Configuration for LoRA DINOv2 ViT-S/14 model."""
    name: str = "lora_dinov2_vits14"
    freeze: bool = True
    output_dim: int = 384
    patch_size: int = 14
    attention_heads: int = 6
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target: List[str] = field(default_factory=lambda: ['qkv', 'proj'])
    lora_qkv_enable: List[bool] = field(default_factory=lambda: [True, False, True])
    
@EncoderRegistry.register("clip_vitb16")
@dataclass
class CLIPViTB16Config(EncoderConfig):
    """Configuration for CLIP ViT-B/16 model."""
    name: str = "clip_vitb16"
    freeze: bool = True
    output_dim: int = 768
    patch_size: int = 16
    attention_heads: int = 12

@EncoderRegistry.register("swinv2_tiny_window8_256")
@dataclass
class SwinV2TinyWindow8Config(EncoderConfig):
    """Configuration for SwinV2 Tiny Window 8 model."""
    name: str = "swinv2_tiny_window8_256"
    freeze: bool = True
    output_dim: List[int] = field(default_factory=lambda: [96, 192, 384, 768])

@EncoderRegistry.register("lora_swinv2_tiny_window8_256")
@dataclass
class LoRASwinV2TinyWindow8Config(EncoderConfig):
    """Configuration for LoRA SwinV2 Tiny Window 8 model."""
    name: str = "lora_swinv2_tiny_window8_256"
    freeze: bool = True
    output_dim: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target: List[str] = field(default_factory=lambda: ['qkv', 'proj'])
    lora_qkv_enable: List[bool] = field(default_factory=lambda: [True, False, True])

@EncoderRegistry.register("swinv2_small_window8_256")
@dataclass
class SwinV2SmallWindow8Config(EncoderConfig):
    """Configuration for SwinV2 Small Window 8 model."""
    name: str = "swinv2_small_window8_256"
    freeze: bool = True
    output_dim: List[int] = field(default_factory=lambda: [96, 192, 384, 768])

@EncoderRegistry.register("lora_swinv2_small_window8_256")
@dataclass
class LoRASwinV2SmallWindow8Config(EncoderConfig):
    """Configuration for LoRA SwinV2 Small Window 8 model."""
    name: str = "lora_swinv2_small_window8_256"
    freeze: bool = True
    output_dim: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target: List[str] = field(default_factory=lambda: ['qkv', 'proj'])
    lora_qkv_enable: List[bool] = field(default_factory=lambda: [True, False, True])

@EncoderRegistry.register("swinv2_base_window8_256")
@dataclass
class SwinV2BaseWindow8Config(EncoderConfig):
    """Configuration for SwinV2 Base Window 8 model."""
    name: str = "swinv2_base_window8_256"
    freeze: bool = True
    output_dim: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])

@EncoderRegistry.register("lora_swinv2_base_window8_256")
@dataclass
class LoRASwinV2BaseWindow8Config(EncoderConfig):
    """Configuration for LoRA SwinV2 Base Window 8 model."""
    name: str = "lora_swinv2_base_window8_256"
    freeze: bool = True
    output_dim: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target: List[str] = field(default_factory=lambda: ['qkv', 'proj'])
    lora_qkv_enable: List[bool] = field(default_factory=lambda: [True, False, True])


@EncoderRegistry.register("vit_adapter_dinov2_vits14")
@dataclass
class ViTAdapterDinov2ViTS14Config(EncoderConfig):
    """Configuration for ViT-Adapter with DinoV2 ViTS14 model."""
    name: str = "vit_adapter_dinov2_vits14"
    dinov2_type: str = "dinov2_vits14"
    lora_backbone: bool = False
    freeze: bool = True

    # lora_r: int = 16
    # lora_alpha: int = 32
    # lora_target: List[str] = field(default_factory=lambda: ['qkv', 'proj'])
    # lora_qkv_enable: List[bool] = field(default_factory=lambda: [True, False, True])
    output_dim: List[int] = field(default_factory=lambda: [384, 384, 384, 384])
    conv_inplane: int = 64
    n_points: int = 4
    deform_num_heads: int = 6
    init_values: float = 0.0
    interaction_indexes: List[List[int]] = field(default_factory=lambda: [[0, 2], [3, 5], [6, 8], [9, 11]])
    with_cffn: bool = True
    cffn_ratio: float = 0.25
    add_vit_feature: bool = True
    use_extra_extractor: bool = True
    use_cls: bool = True
    with_cp: bool = False
    drop_path_rate: float = 0.0
    
