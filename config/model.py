from dataclasses import dataclass, field
from typing import Optional, Tuple

from .base import ModelConfig, EncoderConfig, DecoderConfig
from .registry import ConfigRegistry
from .encoder import EncoderRegistry
from .decoder import DecoderRegistry

# Create registry for full model configurations
ModelRegistry = ConfigRegistry[ModelConfig]("ModelRegistry")

@ModelRegistry.register("dino_vits8-linear_probing")
@dataclass
class DinoViTS8LinearProbingConfig(ModelConfig):
    name: str = 'dino_vits8-linear_probing'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dino_vits8")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("lora_dino_vits8-linear_probing")
@dataclass
class LoRADinoViTS8LinearProbingConfig(ModelConfig):
    name: str = 'lora_dino_vits8-linear_probing'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("lora_dino_vits8")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("dinov2_vits14-linear_probing")
@dataclass
class Dinov2ViTS14LinearProbingConfig(ModelConfig):
    name: str = 'dinov2_vits14-linear_probing'
    input_size: Tuple[int, int] = (392, 392) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("dinov2_vits14_px224-linear_probing")
@dataclass
class Dinov2ViTS14Px224LinearProbingConfig(ModelConfig):
    name: str = 'dinov2_vits14_px224-linear_probing'
    input_size: Tuple[int, int] = (224, 224) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("clip_vitb16-linear_probing")
@dataclass
class CLIPViTB16LinearProbingConfig(ModelConfig):
    name: str = 'clip_vitb16-linear_probing'
    input_size: Tuple[int, int] = (224, 224) ## So that the output spatial size from the encoder is still (28, 28)

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("clip_vitb16")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim

@ModelRegistry.register("dino_vits8-segformer_head")
@dataclass
class DinoViTS8SegFormerHeadConfig(ModelConfig):
    name: str = 'dino_vits8-segformer_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dino_vits8")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("dinov2_vits14-segformer_head")
@dataclass
class Dinov2ViTS14SegFormerHeadConfig(ModelConfig):
    name: str = 'dinov2_vits14-segformer_head'
    input_size: Tuple[int, int] = (392, 392)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("dinov2_vits14_px224-segformer_head")
@dataclass
class Dinov2ViTS14Px224SegFormerHeadConfig(ModelConfig):
    name: str = 'dinov2_vits14_px224-segformer_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dinov2_vits14")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("clip_vitb16-segformer_head")
@dataclass
class CLIPViTB16SegFormerHeadConfig(ModelConfig):
    name: str = 'clip_vitb16-segformer_head'
    input_size: Tuple[int, int] = (224, 224)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("clip_vitb16")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.in_channels = [self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim, self.encoder.output_dim] 
        self.decoder.encoder_name = self.encoder.name

@ModelRegistry.register("swinv2_tiny_window8_256-segformer_head")
@dataclass
class SwinV2TinyWindow8SegFormerHeadConfig(ModelConfig):
    name: str = 'swinv2_tiny_window8_256-segformer_head'
    input_size: Tuple[int, int] = (256, 256)
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("swinv2_tiny_window8_256")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("segformer_head")()
    )
    
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.in_channels = self.encoder.output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.encoder_name = self.encoder.name