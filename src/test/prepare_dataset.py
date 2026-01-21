# prepare_dataset.py
import os
import pandas as pd
from pathlib import Path

# Chemins (adaptez si besoin)
chemin_non_seg = "./Data_Projet_Complet/Data_Projet/mission_herbonaute_2000/"
chemin_seg = "./Data_Projet_Complet/Data_Projet/mission_herbonaute_2000_seg_black/"
chemin_labels = "./Data_Projet_Complet/Data_Projet/labels_1.csv"

# 1. Charger le CSV des labels
df_labels = pd.read_csv(chemin_labels, sep=';', skiprows=1)
print(f"Nombre de lignes dans le CSV : {len(df_labels)}")
print(f"\nPremières lignes du CSV :")
print(df_labels.head())

# Créer un dictionnaire code -> label pour recherche rapide
labels_dict = dict(zip(df_labels['code'], df_labels['epines']))
print(f"\nNombre de labels disponibles : {len(labels_dict)}")

# 2. Lister les fichiers images (en excluant .DS_Store)
fichiers_non_seg = set([f.replace('.jpg', '') for f in os.listdir(chemin_non_seg) if f.endswith('.jpg')])
fichiers_seg = set([f.replace('.jpg', '') for f in os.listdir(chemin_seg) if f.endswith('.jpg')])

print(f"\nNombre d'images .jpg non segmentées : {len(fichiers_non_seg)}")
print(f"Nombre d'images .jpg segmentées : {len(fichiers_seg)}")

# 3. Trouver les images présentes dans les DEUX dossiers
images_communes = fichiers_non_seg.intersection(fichiers_seg)
print(f"Nombre d'images présentes dans les deux dossiers : {len(images_communes)}")

# 4. Créer le dataset
dataset_list = []
images_avec_label = 0
images_sans_label = 0

for image_id in sorted(images_communes):
    chemin_complet_non_seg = os.path.join(chemin_non_seg, f"{image_id}.jpg")
    chemin_complet_seg = os.path.join(chemin_seg, f"{image_id}.jpg")
    
    # Chercher le label dans le dictionnaire
    if image_id in labels_dict:
        label = labels_dict[image_id]
        images_avec_label += 1
    else:
        label = None  # Pas de label trouvé
        images_sans_label += 1
    
    dataset_list.append({
        'image_id': image_id,
        'non_seg': chemin_complet_non_seg,
        'seg': chemin_complet_seg,
        'label': label
    })

print(f"\n=== Résumé ===")
print(f"Images avec label : {images_avec_label}")
print(f"Images sans label : {images_sans_label}")
print(f"Total : {len(dataset_list)}")

# 5. Afficher quelques exemples
print(f"\n5 premiers échantillons :")
for i in range(min(5, len(dataset_list))):
    item = dataset_list[i]
    print(f"\nÉchantillon {i+1}:")
    print(f"  ID:      {item['image_id']}")
    print(f"  Non seg: {item['non_seg']}")
    print(f"  Seg:     {item['seg']}")
    print(f"  Label:   {item['label']}")

# 6. Convertir en DataFrame
df_dataset = pd.DataFrame(dataset_list)
print(f"\n=== Dataset complet ===")
print(df_dataset)

# 7. Sauvegarder le dataset en CSV
output_path = "./data/raw/prepared_dataset.csv"
df_dataset.to_csv(output_path, index=False)
print(f"\n✅ Dataset sauvegardé dans : {output_path}")

# 8. Statistiques du dataset
print(f"\n=== Statistiques ===")
print(f"Nombre total d'échantillons : {len(df_dataset)}")
print(f"Échantillons avec label : {df_dataset['label'].notna().sum()}")
print(f"Échantillons sans label : {df_dataset['label'].isna().sum()}")
print(f"\nDistribution des labels (hors null) :")
print(df_dataset['label'].value_counts(dropna=True))