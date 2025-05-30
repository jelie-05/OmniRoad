## Modified from https://github.com/moatifbutt/r2s100k
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import torch 

class R2S100k(Dataset):
    def __init__(self, image_base: str, seg_base: str, image_transform, mask_transform, label_colors_list, class_names, split):
        self.image_paths = glob.glob(f"{image_base}/*")
        self.seg_paths = glob.glob(f"{seg_base}/*")
        self.image_paths.sort()
        self.seg_paths.sort()
        # print(self.image_paths[len(self.image_paths) - 1])
        # print(self.seg_paths[len(self.image_paths) - 1])
        # assert len(self.image_paths) == len(self.seg_paths), f"Number of images ({len(self.image_paths)}) and masks ({len(self.seg_paths)}) don't match!!!"
        self.label_colors_list = label_colors_list
        self.class_names = class_names
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.class_values = [self.class_names.index(cls.lower()) for cls in self.class_names]
        self.split = split

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

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.seg_paths[index])
        # print(image.size)
        # print(mask.size)
        if self.image_transform:
            image = self.image_transform(image)

        # if self.split != 'test':
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # image = np.transpose(image, (2, 0, 1))
        mask = np.array(mask)
        # print(mask.shape)
        mask = self.get_label_mask(mask, self.class_values)

        # image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 
        # print(image.shape)
        # print(mask.shape)
        
        return image, mask