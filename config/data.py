"""Dataset configurations."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from .base import DataConfig
from .registry import ConfigRegistry
import torchvision.transforms as T

DataRegistry = ConfigRegistry[DataConfig]("DataRegistry")

@DataRegistry.register("r2s100k")
@dataclass
class R2S100KConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "r2s100k"
    dataset_path: str = "/home/phd_li/dataset/r2s100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For r2s100k
    # Raw dataset information - this represents the complete ground truth
    _raw_num_classes: int = 15
    _raw_class_names: List[str] = field(default_factory=lambda: [
        'bg', 'wet_road_region', 'road_region', 'mud', 'earthen_patch', 
        'mountain-stones', 'dirt', 'vegitation_misc', 'distressed_patch', 
        'drainage_grate', 'water_puddle', 'speed_breaker', 'misc', 
        'gravel_patch', 'concrete_material'
    ])
    _raw_label_colors_list: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 0),          # BG
        (2, 79, 59),        # Wet_Road_Region
        (17, 163, 74),      # Road_region
        (112, 84, 62),      # Mud
        (225, 148, 79),     # Earthen_Patch
        (120, 114, 104),    # Mountain-stones
        (166, 130, 95),     # Dirt
        (128, 222, 91),     # Vegitation_Misc
        (119, 61, 128),     # Distressed_Patch
        (93, 86, 176),      # Drainage_Grate
        (140, 160, 222),    # Water_puddle
        (234, 133, 5),      # Speed_Breaker
        (156, 28, 39),      # Misc 
        (99, 122, 130),     # Gravel_Patch 
        (123, 43, 31),      # Concrete_Material
    ])

    # num_classes = 15
    # ignore_index = 0
    # label_colors_list = [
    #         (0, 0, 0), # BG
    #         (2, 79, 59), # Wet_Road_Region
    #         (17, 163, 74), # Road_region
    #         (112, 84, 62), # Mud
    #         (225, 148, 79), # Earthen_Patch
    #         (120, 114, 104), # Mountain-stones
    #         (166, 130, 95), # Dirt
    #         (128, 222, 91), # Vegitation_Misc
    #         (119, 61, 128), # Distressed_Patch
    #         (93, 86, 176), # Drainage_Grate
    #         (140, 160, 222), # Water_puddle
    #         (234, 133, 5), #Speed_Breaker
    #         (156, 28, 39), # Misc 
    #         (99, 122, 130), # Gravel_Patch 
    #         (123, 43, 31), # Concrete_Material
    #     ]
    # all the classes that are present in the dataset
    # class_names = ['bg', 'wet_road_region', 'road_region', 'mud', 'earthen_patch', 'mountain-stones', 'dirt', 'vegitation_misc', 'distressed_patch', 'drainage_grate', 'water_puddle', 'speed_breaker', 'misc', 'gravel_patch', 'concrete_material']
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return self._raw_num_classes  # All 15 classes
        
        elif self.task_type == "instance_segmentation":
            # Only classes that can form instances (exclude background)
            # return len([cls_name for cls_name in self._raw_class_names if cls_name != 'bg'])  # 14 classes
            return self._raw_num_classes  # All 15 classes

        elif self.task_type == "panoptic_segmentation":
            # For panoptic, we typically count only the "thing" classes for instance detection
            # "Stuff" classes are handled separately through semantic segmentation
            # return len(self._thing_classes)  # Number of thing classes
            raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    @property
    def class_names(self) -> List[str]:
        """Return the class names appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return self._raw_class_names
        
        elif self.task_type == "instance_segmentation":
            # Remove background, return detectable classes
            # return [cls for cls in self._raw_class_names if cls != 'bg']
            return self._raw_class_names

        elif self.task_type == "panoptic_segmentation":
            # Return thing classes for instance detection
            raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    @property
    def ignore_index(self) -> Optional[int]:
        """Return the ignore index appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return 0  # Ignore background in loss computation
        
        elif self.task_type == "instance_segmentation":
            # return 255  # No ignore index needed - background simply doesn't generate instances
            return 0  
        
        elif self.task_type == "panoptic_segmentation":
           raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    @property
    def label_colors_list(self) -> List[Tuple[int, int, int]]:
        """Return the label colors appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return self._raw_label_colors_list
        
        elif self.task_type == "instance_segmentation":
            # Return colors for detectable classes (excluding background)
            # return [color for i, color in enumerate(self._raw_label_colors_list) if i != 0]
            return self._raw_label_colors_list
        
        elif self.task_type == "panoptic_segmentation":
            # Return colors for thing classes
            raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def get_transforms(self):
        """Get transforms based on configuration."""
        try:
            print("Input size: ", self.input_size)
            print(f"Mean: {self.mean}, Std: {self.std}")
            image_transform = T.Compose([ 
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize(self.input_size)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224))
            ])
        
        return image_transform, mask_transform