from dataclasses import dataclass, field
from typing import Optional

from .base import ModelConfig, EncoderConfig, DecoderConfig
from .registry import ConfigRegistry
from .encoder import EncoderRegistry
from .decoder import DecoderRegistry

# Create registry for full model configurations
ModelRegistry = ConfigRegistry[ModelConfig]("ModelRegistry")

@ModelRegistry.register("dino_segmentation")
@dataclass
class DinoSegmentationConfig(ModelConfig):
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderRegistry.get("dino_vits8")()
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderRegistry.get("linear_probing")()
    )
    def __post_init__(self):
        # Ensure decoder input_dim matches encoder output_dim
        self.decoder.input_dim = self.encoder.output_dim
        self.decoder.spatial_size = self.encoder.spatial_size