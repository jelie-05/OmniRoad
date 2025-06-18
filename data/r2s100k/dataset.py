## Modified from https://github.com/moatifbutt/r2s100k
from torch.utils.data import Dataset, default_collate
import glob
import numpy as np
from PIL import Image
import torch 
import torchvision.transforms as T
from typing import Tuple, List
import cv2 

class R2S100k(Dataset):
    def __init__(self, image_base: str, seg_base: str, image_transform, mask_transform, label_colors_list, class_names, split, task_type):
        self.image_paths = glob.glob(f"{image_base}/*")
        self.seg_paths = glob.glob(f"{seg_base}/*")
        self.image_paths.sort()
        self.seg_paths.sort()

        self.label_colors_list = label_colors_list
        self.class_names = class_names
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.class_values = [self.class_names.index(cls.lower()) for cls in self.class_names]
        self.split = split

        self.test_mask_transform = T.Compose([ 
                T.Resize((144, 256)) ## From R2S100K Repo
            ])
        self.task_type = task_type

    def __len__(self):
        return len(self.image_paths)

    def get_label_mask(self, mask, class_values): 
        """
        This function encodes the pixels belonging to the same class
        in the image into the same label
        """
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for value in class_values:
            for ii, label in enumerate(self.label_colors_list):
                if value == self.label_colors_list.index(label):
                    label = np.array(label)
                    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)

        return label_mask

    def label_mask_to_color_mask(self, label_mask):
        red_map = np.zeros_like(label_mask).astype(np.uint8)
        green_map = np.zeros_like(label_mask).astype(np.uint8)
        blue_map = np.zeros_like(label_mask).astype(np.uint8)
        
        for label_num in range(0, len(self.label_colors_list)):
            if label_num in self.class_values:
                idx = label_mask == label_num
                red_map[idx] = np.array(self.label_colors_list)[label_num, 0]
                green_map[idx] = np.array(self.label_colors_list)[label_num, 1]
                blue_map[idx] = np.array(self.label_colors_list)[label_num, 2]
            
        segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
        return segmented_image

    def semantic_to_instances(self, semantic_mask: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        labels = []
        masks = []
            
        unique_classes = np.unique(semantic_mask)
        
        for class_idx in unique_classes:
            # Create binary mask for this class
            class_mask = (semantic_mask == class_idx).astype(np.uint8)
            
            # Find connected components
            # num_la bels, label_image = cv2.connectedComponents(class_mask)
            
            # for component_id in range(1, num_labels):
            #     instance_mask = (label_image == component_id).astype(np.float32)
            #     labels.append(int(class_idx))
            #     masks.append(instance_mask)
            labels.append(int(class_idx))
            masks.append(class_mask.astype(np.float32))
        
        return labels, masks
    
    def instances_to_semantic(self, targets: List[dict], ignore_index: int = 255):
        batch_size = len(targets)
        _, H, W = targets[0]['masks'].shape
        masks_list = torch.ones((batch_size, H, W))
        masks_list *= ignore_index

        for i, target in enumerate(targets):
            labels = target['labels']
            masks = target['masks']
            for label, mask in zip(labels, masks):
                masks_list[i][mask.bool()] = label

        return masks_list
        
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.seg_paths[index])
        orig_size = mask.size
        # print(image.size)
        # print(mask.size)
        if self.image_transform:
            image = self.image_transform(image)

        if self.split == 'test':
            mask = self.test_mask_transform(mask)
        else:
            if self.mask_transform:
                mask = self.mask_transform(mask)

        mask = np.array(mask)
        mask = self.get_label_mask(mask, self.class_values)

        if self.task_type == "semantic_segmentation":
            mask = torch.tensor(mask, dtype=torch.long) 
            return image, mask

        elif self.task_type == "instance_segmentation":
            labels, instance_masks = self.semantic_to_instances(mask)
            # Handle case with no instances
            if len(labels) == 0:
                labels = torch.zeros(0, dtype=torch.long)
                masks = torch.zeros(0, semantic_array.shape[0], semantic_array.shape[1], dtype=torch.float32)
            else:
                labels = torch.tensor(labels, dtype=torch.long)
                masks = torch.stack([torch.from_numpy(mask) for mask in instance_masks])
                
            # Create target dictionary
            target = {
                "labels": labels,
                "masks": masks,
                "image_id": index,
                "orig_size": orig_size,  # (H, W)
            }
            
            return image, target
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
    
    @property
    def collate_fn(self):
        def func(batch):
            if self.task_type == "semantic_segmentation":
                imgs = [e[0] for e in batch]
                targets = [e[1] for e in batch]
                imgs = default_collate(imgs)
                targets = default_collate(targets)
                return imgs, targets
            elif self.task_type == "instance_segmentation":
                imgs = [e[0] for e in batch]
                targets = [e[1] for e in batch]
                imgs = default_collate(imgs)
                return imgs, targets
        return func