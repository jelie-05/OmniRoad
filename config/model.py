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