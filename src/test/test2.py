import pandas as pd
from PIL import Image
import numpy as np

df = pd.read_csv("./data/prepared_dataset.csv")

print("="*60)
print("VÉRIFICATION DE LA QUALITÉ DES SEGMENTATIONS")
print("="*60)

problemes_seg = []
stats_ratio = []

for idx, row in df.iterrows():
    image_id = row['image_id']
    path_seg = row['seg']
    
    try:
        img_seg = Image.open(path_seg).convert('L')  # Convertir en niveaux de gris
        img_array = np.array(img_seg)
        
        # Compter les pixels non noirs (plante)
        pixels_plante = np.sum(img_array > 0)
        pixels_total = img_array.size
        ratio = pixels_plante / pixels_total
        
        stats_ratio.append(ratio)
        
        # Vérifier les cas problématiques
        if ratio < 0.01:  # Moins de 1% de plante
            problemes_seg.append(f"⚠️ {image_id}: Segmentation quasi vide ({ratio*100:.2f}%)")
        elif ratio > 0.95:  # Plus de 95% de plante
            problemes_seg.append(f"⚠️ {image_id}: Segmentation couvre presque tout ({ratio*100:.2f}%)")
            
    except Exception as e:
        problemes_seg.append(f"❌ {image_id}: Erreur - {e}")

print(f"\n✅ Segmentations analysées : {len(df)}")
print(f"⚠️ Problèmes détectés : {len(problemes_seg)}")

if stats_ratio:
    print(f"\nStatistiques sur les ratios plante/image :")
    print(f"  Moyenne : {np.mean(stats_ratio)*100:.2f}%")
    print(f"  Médiane : {np.median(stats_ratio)*100:.2f}%")
    print(f"  Min : {np.min(stats_ratio)*100:.2f}%")
    print(f"  Max : {np.max(stats_ratio)*100:.2f}%")

if problemes_seg:
    print("\nDétail des problèmes :")
    for p in problemes_seg[:20]:
        print(f"  {p}")