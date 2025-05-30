from dataclasses import dataclass, field
from typing import Optional

from .base import Config, TrainingConfig, OptimizerConfig, SchedulerConfig, ModelConfig, DataConfig
from .registry import ConfigRegistry
from .model import ModelRegistry
from .data import DataRegistry

# Create registry for experiment configurations
ExperimentRegistry = ConfigRegistry[Config]("ExperimentRegistry")

@ExperimentRegistry.register("segmentation_r2s100k")
@dataclass
class SegmentationR2S100k(Config):
    experiment_name: str = "segmentation_r2s100k"
    # run_id: Optional[str] = None
    model_name: str = "custom_model"
    model: ModelConfig = field(init=False)
    # model: ModelConfig = None
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

    def __post_init__(self):
        self.model = ModelRegistry.get(self.model_name)()
        self.model.decoder.num_classes = self.data.num_classes
        self.data.input_size = self.model.input_size
        if 'clip' in self.model.name:
            print("CLIP is used, changing to its default mean and std for image_transform")
            self.data.mean = [0.48145466, 0.4578275, 0.40821073]
            self.data.std = [0.26862954, 0.26130258, 0.27577711]
        