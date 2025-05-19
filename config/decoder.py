from dataclasses import dataclass, field
from typing import List, Optional

from .base import DecoderConfig
from .registry import ConfigRegistry

# Create a registry for probing head configurations
DecoderRegistry = ConfigRegistry[DecoderConfig]("DecoderRegistry")

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