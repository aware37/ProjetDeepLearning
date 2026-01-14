import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random

df = pd.read_csv("./data/prepared_dataset.csv")

# Prendre 6 échantillons aléatoires (3 avec épines, 3 sans)
echantillons_0 = df[df['label'] == 0].sample(3)
echantillons_1 = df[df['label'] == 1].sample(3)
echantillons = pd.concat([echantillons_0, echantillons_1])

fig, axes = plt.subplots(6, 2, figsize=(16, 12))

for idx, (_, row) in enumerate(echantillons.iterrows()):
    # Image non segmentée
    img_non_seg = Image.open(row['non_seg'])
    axes[idx, 0].imshow(img_non_seg)
    axes[idx, 0].set_title(f"{row['image_id']}\nLabel: {int(row['label'])}")
    axes[idx, 0].axis('off')
    
    # Image segmentée
    img_seg = Image.open(row['seg'])
    axes[idx, 1].imshow(img_seg)
    axes[idx, 1].set_title("Segmentée")
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.savefig("verification_echantillons.png", dpi=150, bbox_inches='tight')
print("✅ Visualisation sauvegardée : verification_echantillons.png")
plt.show()