import pandas as pd
from PIL import Image
import os

df = pd.read_csv("./data/prepared_dataset.csv")

print("="*60)
print("VÉRIFICATION DE LA QUALITÉ DES IMAGES")
print("="*60)

problemes = []

for idx, row in df.iterrows():
    image_id = row['image_id']
    path_non_seg = row['non_seg']
    path_seg = row['seg']
    
    # Vérifier image non segmentée
    try:
        img = Image.open(path_non_seg)
        w, h = img.size
        mode = img.mode
        
        # Vérifier la taille minimale
        if w < 50 or h < 50:
            problemes.append(f"❌ {image_id}: Image trop petite ({w}x{h})")
        
        # Vérifier le mode (RGB attendu)
        if mode not in ['RGB', 'L']:
            problemes.append(f"⚠️ {image_id}: Mode inhabituel ({mode})")
            
    except Exception as e:
        problemes.append(f"❌ {image_id}: Non-seg corrompue - {e}")
    
    # Vérifier image segmentée
    try:
        img_seg = Image.open(path_seg)
        w_seg, h_seg = img_seg.size
        
        # Vérifier que les deux images ont la même taille
        if (w, h) != (w_seg, h_seg):
            problemes.append(f"⚠️ {image_id}: Tailles différentes - Non-seg:{w}x{h}, Seg:{w_seg}x{h_seg}")
            
    except Exception as e:
        problemes.append(f"❌ {image_id}: Seg corrompue - {e}")

print(f"\n✅ Images vérifiées : {len(df)}")
print(f"⚠️ Problèmes détectés : {len(problemes)}")

if problemes:
    print("\nDétail des problèmes :")
    for p in problemes[:20]:  # Afficher les 20 premiers
        print(f"  {p}")
else:
    print("\n🎉 Toutes les images sont OK !")