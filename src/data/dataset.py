import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # <--- LA LIGNE MAGIQUE

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
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.dropna(subset=['label'])
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        
        non_seg_path = row['non_seg']
        seg_path = row['seg']
        label = row['label']
        
        image_non_seg = Image.open(non_seg_path).convert("RGB")
        image_seg = Image.open(seg_path).convert("RGB")
        
        if self.transform:
            image_non_seg = self.transform(image_non_seg)
            image_seg = self.transform(image_seg)
        
        return image_non_seg, image_seg, label



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



def get_dataloaders(train_subset, val_subset, batch_size=32):
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader



def build_transform(img_size=224):
    """
    Retourne un dictionnaire contenant les transformations pour train et val.
    Args:
        img_size (int): Taille cible pour le redimensionnement (défaut 224 pour ViT standard).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }
    return data_transforms

def create_dataloaders(csv_file, batch_size=32, train_split=0.8, img_size=224, seed=42):
    """
    Crée les DataLoaders pour l'entraînement et la validation.
    
    Args:
        csv_file (str): Chemin vers le fichier CSV contenant les chemins d'images et labels.
        batch_size (int): Taille des lots pour les DataLoaders.
        train_split (float): Proportion des données utilisées pour l'entraînement.
        img_size (int): Taille cible pour le redimensionnement des images.
        seed (int): Graine pour la reproductibilité lors du split.
    
    Returns:
        train_loader, val_loader: DataLoaders pour l'entraînement et la validation
    """
    # Obtenir les transformations
    data_transforms = build_transform(img_size=img_size)
    
    # Charger le dataset complet avec les transformations
    full_dataset = load_dataset(csv_file, transform=data_transforms['train'])
    
    # Diviser en ensembles d'entraînement et de validation
    train_subset, val_subset = split_dataset(full_dataset, train_ratio=train_split, seed=seed)
    
    # Appliquer les transformations appropriées
    train_subset.dataset.transform = data_transforms['train']
    val_subset.dataset.transform = data_transforms['val']
    
    # Créer les DataLoaders
    train_loader, val_loader = get_dataloaders(train_subset, val_subset, batch_size=batch_size)
    
    return train_loader, val_loader