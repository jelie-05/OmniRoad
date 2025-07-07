"""Dataset configurations."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from .base import DataConfig
from .registry import ConfigRegistry
import torchvision.transforms as T
from collections import namedtuple
from torchvision.transforms import InterpolationMode

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


# Copied from https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py
# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate). Do not modify
        # these IDs, since exactly these IDs are expected by the evaluation
        # server.
        "trainId",
        # Feel free to modify these IDs as suitable for your method. Then
        # create ground truth images with train IDs, using the tools provided
        # in the 'preparation' folder. However, make sure to validate or submit
        # results to our evaluation server using the regular IDs above! For
        # trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the
        # inverse mapping, we use the label that is defined first in the list
        # below. For example, mapping all void-type classes to the same ID in
        # training, might make sense for some approaches. Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",
        # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",
        # Whether this label distinguishes between single instances or not
        "ignoreInEval",
        # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)

# Our extended list of label types. Our train id is compatible with Cityscapes
bdd100k_labels = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ground", 3, 255, "void", 0, False, True, (81, 0, 81)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("parking", 5, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("building", 10, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("fence", 11, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("garage", 12, 255, "construction", 2, False, True, (180, 100, 180)),
    Label(
        "guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180)
    ),
    Label("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("wall", 15, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("banner", 16, 255, "object", 3, False, True, (250, 170, 100)),
    Label("billboard", 17, 255, "object", 3, False, True, (220, 220, 250)),
    Label("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0)),
    Label("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60)),
    Label("pole", 20, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153)),
    Label("street light", 22, 255, "object", 3, False, True, (220, 220, 100)),
    Label("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0)),
    Label(
        "traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)
    ),
    Label("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0)),
    Label(
        "traffic sign frame",
        27,
        255,
        "object",
        3,
        False,
        True,
        (250, 170, 250),
    ),
    Label("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("sky", 30, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]

bdd100k_drivables = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("direct", 0, 0, "drivable", 0, False, False, (219, 94, 86)),
    Label("alternative", 1, 1, "drivable", 0, False, False, (86, 211, 219)),
    Label("background", 2, 2, "drivable", 0, False, False, (0, 0, 0)),
]

bdd100k_lane_directions = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("parallel", 0, 0, "lane_mark", 0, False, False, (0, 0, 0)),
    Label("vertical", 1, 1, "lane_mark", 0, False, False, (0, 0, 0)),
]

bdd100k_lane_styles = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("solid", 0, 0, "lane_mark", 0, False, False, (0, 0, 0)),
    Label("dashed", 1, 1, "lane_mark", 0, False, False, (0, 0, 0)),
]

bdd100k_lane_categories = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("crosswalk", 0, 0, "lane_mark", 0, False, False, (219, 94, 86)),
    Label("double other", 1, 1, "lane_mark", 0, False, False, (219, 194, 86)),
    Label("double white", 2, 2, "lane_mark", 0, False, False, (145, 219, 86)),
    Label("double yellow", 3, 3, "lane_mark", 0, False, False, (86, 219, 127)),
    Label("road curb", 4, 4, "lane_mark", 0, False, False, (86, 211, 219)),
    Label("single other", 5, 5, "lane_mark", 0, False, False, (86, 111, 219)),
    Label("single white", 6, 6, "lane_mark", 0, False, False, (160, 86, 219)),
    Label("single yellow", 7, 7, "lane_mark", 0, False, False, (219, 86, 178)),
]

BDD100K_ANNOTATIONs = {
    'sem_seg': bdd100k_labels,
    'lane': bdd100k_lane_categories,
    'drivable': bdd100k_drivables,
}

@DataRegistry.register("semseg_bdd100k")
@dataclass
class SemsegBDD100kConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "semseg_bdd100k"
    dataset_path: str = "/home/phd_li/dataset/bdd100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation    
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For bdd100k
    ignore_index = 255
    num_classes = None
    class_names = None
    label_colors_list = None
    
    def parse_color_and_names(self):
        trainid_colors = []
        labels = BDD100K_ANNOTATIONs['sem_seg']
        for idx, label in enumerate(labels):
            if label.trainId != 255:
                trainid_colors.append({'trainId': label.trainId, 'color': label.color, 'name': label.name})
        label_colors_list = [None] * len(trainid_colors)
        class_names = [None] * len(trainid_colors)

        for label in trainid_colors:
            label_colors_list[label['trainId']] = label['color']
            class_names[label['trainId']] = label['name']
        
        return label_colors_list, class_names
        
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
                T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            ])
        
        return image_transform, mask_transform

@DataRegistry.register("bdd100k")
@dataclass
class BDD100kConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "bdd100k"
    dataset_path: str = "/home/phd_li/dataset/bdd100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation
    
    tasks: List[str] = field(default_factory=lambda: ['sem_seg', 'lane', 'drivable'])
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For bdd100k
    ignore_index = 255

    num_classes = None
    class_names = None
    label_colors_list = None
    

    def parse_color_and_names(self, task):
        trainid_colors = []
        labels = BDD100K_ANNOTATIONs[task]
        for idx, label in enumerate(labels):
            if label.trainId != 255:
                trainid_colors.append({'trainId': label.trainId, 'color': label.color, 'name': label.name})
        label_colors_list = [None] * len(trainid_colors)
        class_names = [None] * len(trainid_colors)

        for label in trainid_colors:
            label_colors_list[label['trainId']] = label['color']
            class_names[label['trainId']] = label['name']
        
        return label_colors_list, class_names
        
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
                T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            ])
        
        return image_transform, mask_transform