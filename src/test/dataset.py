import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class load_dataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on images.
            target_transform (callable, optional): Optional transform to be applied on labels.
        """
        self.data_frame = pd.read_csv(csv_file)
        # Remove rows with missing labels if you only want labeled data
        self.data_frame = self.data_frame.dropna(subset=['label'])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        # Get the row
        row = self.data_frame.iloc[index]
        
        # Get paths and label from CSV
        img_id = row['image_id']
        non_seg_path = row['non_seg']
        seg_path = row['seg']
        label = row['label']
        
        # Load images
        image_non_seg = Image.open(non_seg_path).convert("RGB")
        image_seg = Image.open(seg_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image_non_seg = self.transform(image_non_seg)
            image_seg = self.transform(image_seg)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image_non_seg, image_seg, label

