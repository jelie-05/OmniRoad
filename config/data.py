"""Dataset configurations."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .base import DataConfig
from .registry import ConfigRegistry
import torchvision.transforms as T

DataRegistry = ConfigRegistry[DataConfig]("DataRegistry")

@DataRegistry.register("r2s100k")
@dataclass
class R2S100KConfig(DataConfig):
    dataset_name: str = "r2s100k"
    dataset_path: str = "/home/phd_li/dataset/r2s100k"
    task_type: str = "segmentation"
    # image_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    
    num_workers: int = 8
    
    ## For r2s100k
    num_classes = 15
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ignore_index = 255
    label_colors_list = [
            (0, 0, 0), # BG
            (2, 79, 59), # Wet_Road_Region
            (17, 163, 74), # Road_region
            (112, 84, 62), # Mud
            (225, 148, 79), # Earthen_Patch
            (120, 114, 104), # Mountain-stones
            (166, 130, 95), # Dirt
            (128, 222, 91), # Vegitation_Misc
            (119, 61, 128), # Distressed_Patch
            (93, 86, 176), # Drainage_Grate
            (140, 160, 222), # Water_puddle
            (234, 133, 5), #Speed_Breaker
            (156, 28, 39), # Misc 
            (99, 122, 130), # Gravel_Patch 
            (123, 43, 31), # Concrete_Material
        ]

    # all the classes that are present in the dataset
    class_names = ['bg', 'wet_road_region', 'road_region', 'mud', 'earthen_patch', 'mountain-stones', 'dirt', 'vegitation_misc', 'distressed_patch', 'drainage_grate', 'water_puddle', 'speed_breaker', 'misc', 'gravel_patch', 'concrete_material']
        
    def get_transforms(self):
        """Get transforms based on configuration."""
        image_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
        
        mask_transform = T.Compose([
            T.Resize((224, 224))
        ])
        
        return image_transform, mask_transform