from typing import Dict, Type, Any, Optional
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from pathlib import Path

from config.base import DataConfig
from .r2s100k.dataset import R2S100k
from .bdd100k.dataset import SemsegBDD100k
# Registry of dataset implementations
DATASET_REGISTRY = {
    'r2s100k': R2S100k,
    'semseg_bdd100k': SemsegBDD100k,
    # Add other datasets here
}


def get_params_r2s100k(config, split):
    # Create transforms
    image_transform, mask_transform = config.get_transforms()
    # Get dataset path
    base_path = Path(config.dataset_path)

    # R2S100K specific handling
    if split == "train":
        image_base = base_path / "train"
        seg_base = base_path / "Train-Labels"
    elif split == "val":
        image_base = base_path / "val"
        seg_base = base_path / "val_labels"
    elif split == "test":
        image_base = base_path / "test"
        seg_base = base_path / "test_labels"
    else:
        return None
    
    # Check if the split exists
    if not image_base.exists() or not seg_base.exists():
        print(f"Warning: R2S100K {split} split not found at {image_base} and {seg_base}")
        return None
    
    return {
        'image_base': str(image_base),
        'seg_base': str(seg_base),
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'class_names': config.class_names,
        'label_colors_list': config.label_colors_list,
        'split': split,
        'task_type': config.task_type,
    }

def get_params_semseg_bdd100k(config, split):
    # Create transforms
    image_transform, mask_transform = config.get_transforms()
    # Get dataset path
    base_path = Path(config.dataset_path)

    if split == "train":
        image_base = base_path / "images/10k/train"
        label_base = base_path / "labels/sem_seg/masks/train"
    elif split == "val":
        image_base = base_path / "images/10k/val"
        label_base = base_path / "labels/sem_seg/masks/val"
    elif split == "test":
        print(f"Warning: BDD100k doesn't have an official test dataset")
        return None
    else:
        return None
    
    # Check if the split exists
    if not image_base.exists() or not label_base.exists():
        print(f"Warning: BDD100k {split} split not found at {image_base} and {seg_base}")
        return None
    
    return {
        'image_base': str(image_base),
        'label_base': str(label_base),
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'class_names': config.class_names,
        'label_colors_list': config.label_colors_list,
        'split': split,
        'seg_type': config.task_type,
        'ignore_index': config.ignore_index,
    }

get_params = {
    'r2s100k': get_params_r2s100k,
    'semseg_bdd100k': get_params_semseg_bdd100k
}

def get_dataset(config: DataConfig, split: str = "train") -> Optional[Dataset]:
    """
    Create a dataset based on configuration.
    
    Args:
        config: Dataset configuration
        split: Dataset split ("train", "val", or "test")
        
    Returns:
        Initialized dataset or None if the split doesn't exist
    """
    if config.dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    print(f"Loading {config.task_type} dataset with {config.num_classes} classes")
    # Get dataset class
    dataset_class = DATASET_REGISTRY[config.dataset_name]
    # Get dataset-specific params
    dataset_params = get_params[config.dataset_name](config=config, split=split)
    
    if dataset_params is not None:
        return dataset_class(**dataset_params)
    else:
        return None