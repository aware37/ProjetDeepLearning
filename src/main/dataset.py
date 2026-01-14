import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class load_dataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.dropna(subset=['label'])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        
        img_id = row['image_id']
        non_seg_path = row['non_seg']
        seg_path = row['seg']
        label = row['label']
        
        image_non_seg = Image.open(non_seg_path).convert("RGB")
        image_seg = Image.open(seg_path).convert("RGB")
        
        if self.transform:
            image_non_seg = self.transform(image_non_seg)
            image_seg = self.transform(image_seg)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image_non_seg, image_seg, label


def split_dataset(dataset, train=0.8, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split =  int(train * dataset_size)

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.random.shuffle(torch.tensor(indices))

    train_indices, val_indices = indices[:split], indices[split:]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    return train_set, val_set