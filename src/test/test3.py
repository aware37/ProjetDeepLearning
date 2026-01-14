import pandas as pd
from PIL import Image
import numpy as np

df = pd.read_csv("./data/prepared_dataset.csv")

print("="*60)
print("ANALYSE DES TAILLES D'IMAGES")
print("="*60)

tailles = []

for idx, row in df.iterrows():
    img = Image.open(row['non_seg'])
    w, h = img.size
    tailles.append((w, h))

tailles_array = np.array(tailles)

print(f"\n📊 Statistiques des tailles :")
print(f"  Largeur moyenne : {np.mean(tailles_array[:, 0]):.0f} px")
print(f"  Hauteur moyenne : {np.mean(tailles_array[:, 1]):.0f} px")
print(f"  Largeur min : {np.min(tailles_array[:, 0]):.0f} px")
print(f"  Largeur max : {np.max(tailles_array[:, 0]):.0f} px")
print(f"  Hauteur min : {np.min(tailles_array[:, 1]):.0f} px")
print(f"  Hauteur max : {np.max(tailles_array[:, 1]):.0f} px")

# Tailles uniques
tailles_uniques = np.unique(tailles_array, axis=0)
print(f"\n🔍 Nombre de tailles différentes : {len(tailles_uniques)}")

if len(tailles_uniques) <= 10:
    print("\nTailles présentes :")
    for w, h in tailles_uniques:
        count = np.sum((tailles_array == [w, h]).all(axis=1))
        print(f"  {int(w)}×{int(h)} : {count} images")