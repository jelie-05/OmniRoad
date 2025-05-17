from dataclasses import dataclass, field
from typing import Optional

from .base import Config, TrainingConfig, OptimizerConfig, SchedulerConfig, ModelConfig, DataConfig
from .registry import ConfigRegistry
from .model import ModelRegistry
from .data import DataRegistry

# Create registry for experiment configurations
ExperimentRegistry = ConfigRegistry[Config]("ExperimentRegistry")

@ExperimentRegistry.register("dino_segmentation_r2s100k")
@dataclass
class DinoSegmentationR2S100k(Config):
    experiment_name: str = "dino_segmentation_r2s100k"
    model: ModelConfig = field(default_factory=lambda: ModelRegistry.get("dino_segmentation")())
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            batch_size=16,
            epochs=100,
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=0.001,
                weight_decay=0.01
            ),
            scheduler=SchedulerConfig(
                name="cosine",
                warmup_epochs=5
            )
        )
    )
    data: DataConfig = field(default_factory=lambda: DataRegistry.get("r2s100k")())
    seed: int = 0

    def __pose_init__(self):
        self.model.decoder.num_classes = self.data.num_classes