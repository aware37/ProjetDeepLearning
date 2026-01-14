import os
from matplotlib import image
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class load_dataset(Dataset):
    def __init__(self, image_id, non_seg, seg,label, transform=None, target_transform=None):
        self.image_label = pd.read_csv(label)
        self.image_id = image_id
        self.non_seg = non_seg
        self.seg = seg
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_label)
    
    def __getitem__(self, index):
        img_id = self.image_label.iloc[index, 0]
        non_seg_path = os.path.join(self.non_seg, img_id + ".jpg")
        seg_path = os.path.join(self.seg, img_id + ".jpg")
        image_non_seg = Image.open(non_seg_path).convert("RGB")
        image_seg = Image.open(seg_path).convert("RGB")
        label = self.image_label.iloc[index, 1]


        if self.transform:
            image_non_seg = self.transform(image_non_seg)
            image_seg = self.transform(image_seg)
        if self.target_transform:
            label = self.target_transform(label)

        return image_non_seg, image_seg, label

