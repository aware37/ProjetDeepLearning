import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class load_dataset(Dataset):
    def __init__(self, csv_file, img_size=224):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.dropna(subset=['label'])
        self.img_size = img_size
        
        # Normalisation
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        
        # Pipeline pour les image (avec normalisation)
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ])
        
        # Pipeline pour les masques (sans normalisation)
        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        
        # Récupérer les chemins depuis le CSV
        non_seg_full_path = str(row['non_seg'])
        seg_full_path = str(row['seg'])
        
        # Gérer les chemins
        if "Data_Projet_Complet" in non_seg_full_path:
            non_seg_relative = non_seg_full_path[non_seg_full_path.index("Data_Projet_Complet"):]
            seg_relative = seg_full_path[seg_full_path.index("Data_Projet_Complet"):]
        else:
            non_seg_relative = non_seg_full_path
            seg_relative = seg_full_path
        
        non_seg_path = os.path.join(BASE_DIR, non_seg_relative)
        seg_path = os.path.join(BASE_DIR, seg_relative)
        
        label = row['label']
        
        # Charger images
        image_non_seg = Image.open(non_seg_path).convert("RGB")
        image_seg = Image.open(seg_path).convert("RGB")
        
        # Transformations
        image_non_seg_tensor = self.transform_img(image_non_seg)
        image_seg_tensor = self.transform_img(image_seg)
        
        # Masque normalisé
        mask_tensor = self.transform_mask(image_seg)
        mask_binary = (mask_tensor.mean(dim=0, keepdim=True) > 0.1).float()
        mask_binary = mask_binary.expand(3, -1, -1)
        
        return image_non_seg_tensor, image_seg_tensor, mask_binary, label


def split_dataset(dataset, train_ratio=0.8, seed=42):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset, 
        [train_size, val_size], 
        generator=generator
    )
    return train_set, val_set


def get_dataloaders(train_subset, val_subset, num_workers=2, batch_size=32):
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


def create_dataloaders(csv_file, num_workers=2, batch_size=32, train_split=0.8, img_size=224, seed=42):

    full_dataset = load_dataset(csv_file, img_size=img_size)
    train_subset, val_subset = split_dataset(full_dataset, train_ratio=train_split, seed=seed)
    train_loader, val_loader = get_dataloaders(train_subset, val_subset, num_workers=num_workers, batch_size=batch_size)
    
    return train_loader, val_loader