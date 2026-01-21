# explore_data.py
# Objectif : Lister toutes vos paires d'images et vérifier leur cohérence

import os
from pathlib import Path

# TODO: Adapter ces chemins à votre organisation
chemin_non_seg = "./Data_Projet_Complet/Data_Projet/mission_herbonaute_2000/"
chemin_seg = "./Data_Projet_Complet/Data_Projet/mission_herbonaute_2000_seg_black/"

# Lister les fichiers
fichiers_non_seg = sorted(os.listdir(chemin_non_seg))
fichiers_seg = sorted(os.listdir(chemin_seg))

print(f"Nombre d'images non segmentées : {len(fichiers_non_seg)}")
print(f"Nombre d'images segmentées : {len(fichiers_seg)}")

# Vérifier que vous avez le même nombre
assert len(fichiers_non_seg) == len(fichiers_seg), "Nombre d'images différent !"

# Afficher les 5 premières paires
print("\n5 premières paires :")
for i in range(min(5, len(fichiers_non_seg))):
    print(f"  Non seg: {fichiers_non_seg[i]}")
    print(f"  Seg:     {fichiers_seg[i]}")
    print()