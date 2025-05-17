from typing import Dict, Type, Any, Optional
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from pathlib import Path

from config.base import DataConfig
from .r2s100k.dataset import R2S100k

# Registry of dataset implementations
DATASET_REGISTRY = {
    'r2s100k': R2S100k,
    # Add other datasets here
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
    
    # Get dataset class
    dataset_class = DATASET_REGISTRY[config.dataset_name]
    
    # Create transforms
    # image_transform = config.image_transform
    # mask_transform = config.mask_transform
    image_transform, mask_transform = config.get_transforms()
    # Get dataset path
    base_path = Path(config.dataset_path)
    
    if config.dataset_name == 'r2s100k':
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
        
        # Create dataset
        return dataset_class(
            image_base=str(image_base),
            seg_base=str(seg_base),
            image_transform=image_transform,
            mask_transform=mask_transform,
            class_names=config.class_names,
            label_colors_list=config.label_colors_list
        )
    
    # Add handling for other datasets here
    
    return None