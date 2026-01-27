# Classification d'Images avec CrossViT

**GitHub du projet : [https://github.com/aware37/ProjetDeepLearning.git](https://github.com/aware37/ProjetDeepLearning.git)**

Projet de Deep Learning utilisant CrossViT (Cross-Attention Multi-Scale Vision Transformer) pour la classification binaire d'images botaniques (presence/absence d'epines).

## Table des matieres

1. [Architecture du projet](#architecture-du-projet)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Utilisation](#utilisation)
5. [Configurations](#configurations)
6. [Resultats](#resultats)
7. [Details techniques](#details-techniques)

## Architecture du projet

```
PROJET_DEEPL/
├── Data_Projet_Complet/
│   └── Data_Projet/
│       ├── mission_herbonaute_2000/          # Images non segmentees
│       ├── mission_herbonaute_2000_seg_black/ # Images segmentees
│       └── labels.csv                       # Labels
│
├── data/
│   └── raw/
│       └── prepared_dataset.csv              # Dataset prepare
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                        # DataLoader et transformations
│   ├── models/
│   │   ├── crossvit.py                       # CrossViT original
│   │   ├── crossvit_2.py                     # CrossViT avec ponderation masque
│   │   └── crossvit_3.py                     # CrossViT avec IoU loss
│   ├── training/
│   │   └── training.py                       # Boucle d'entrainement
│   ├── evaluation/
│   │   ├── affichage.py                      # Visualisations
│   │   └── evaluate_iou.py                   # Calcul IoU
│   └── test/
│       ├── exploration.py                    # Exploration donnees
│       └── prepare_dataset.py                # Preparation dataset
│
├── main.py                                   # Entrainement standard
├── main_partie4.py                           # Entrainement avec IoU
├── checkpoints/                              # Modeles sauvegardes
├── results/                                  # Resultats et graphiques
└── requirements.txt                          # Dependances
```

## Installation

### Prerequis

- Python 3.8+
- CUDA 11.x (optionnel, pour GPU)

### Installation des dependances

```bash
pip install torch torchvision timm pandas pillow scikit-learn matplotlib seaborn tqdm
```

Ou avec requirements.txt:

```bash
pip install -r requirements.txt
```

## Dataset

### Preparation

Executer le script de preparation:

```bash
mkdir Data_Projet_Complet # ajouter vos images ici 
python src/test/prepare_dataset.py # modifier le chemin dossier image segmenter/non segmenter et label
```

Ce script:
- Associe les images non segmentees avec leurs versions segmentees
- Recupere les labels depuis `labels.csv`
- Genere `data/raw/prepared_dataset.csv`

### Structure du CSV

| Colonne | Description |
|---------|-------------|
| `image_id` | Identifiant unique de l'image |
| `non_seg` | Chemin vers l'image non segmentee |
| `seg` | Chemin vers l'image segmentee |
| `label` | Classe (0: absence, 1: presence d'epines) |

### Preprocessing

- Redimensionnement: 224x224 pixels
- Normalisation Image: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Masque binaire: extrait des images segmentees (seuil > 0.1)

## Utilisation

### Entrainement standard (Partie 1-3)

```bash
python main.py
```

Modifier dans `main.py` la partie a executer:

```python
# Partie 1: CrossViT original
from models.crossvit import crossvit_small_224
configs = ['A', 'B', 'C1', 'C2']
model_used = lambda: crossvit_small_224(num_classes=2, pretrained=False)
output_dir = './results/partie1'

# Partie 2: CrossViT modifie
from models.crossvit import crossvit_part2
configs = ['A', 'B', 'C1', 'C2']
model_used = lambda: crossvit_part2(num_classes=2, pretrained=False)
output_dir = './results/partie2'

# Partie 3: Ponderation par masque
from models.crossvit_2 import crossvit_part2
configs = ['C1', 'C2']
model_used = lambda: crossvit_part2(num_classes=2, pretrained=False)
output_dir = './results/partie3'
```

### Entrainement avec IoU (Partie 4)

```bash
python main_partie4.py
```

Cette partie:
- Entraine les 4 configurations en baseline sur les models des parties 1, 2 et 3
- Entraine C1 avec IoU loss
- Evalue l'IoU attention/masque
- Genere les heatmaps d'attention

### Modèles utilisés pour la Partie 4

Pour l'évaluation IoU et la génération des heatmaps (main_partie4.py), les modèles entraînés et sauvegardés lors des parties 1, 2 et 3 sont réutilisés.  
Ils sont stockés dans les dossiers :
- `checkpoints/partie1/`
- `checkpoints/partie2/`
- `checkpoints/partie3/`

Le script `main_partie4.py` charge automatiquement ces modèles pour comparer les différentes configurations.

## Configurations

### Routage des entrees

| Config | Branche Small | Branche Large | Masque |
|--------|---------------|---------------|--------|
| A | Non-seg | Non-seg | Non |
| B | Seg | Seg | Oui |
| C1 | Non-seg | Seg | Oui |
| C2 | Seg | Non-seg | Oui |

### Perte IoU

Formule combinee:

```
Loss_total = Loss_CE + lambda * Loss_IoU
Loss_IoU = 1 - IoU(attention, masque)
```

Avec `lambda = 0.1` par defaut.

## Resultats

### Fichiers generes

```
results/
├── partie1/
│   ├── curves_config_A.png
│   ├── curves_config_B.png
│   ├── curves_config_C1.png
│   ├── curves_config_C2.png
│   ├── confusion_matrix_config_*.png
│   ├── comparison_all_configs.png
│   ├── results_config_*.csv
│   └── summary.csv
├── partie2/
│   ├── .../
├── partie3/
│   ├── .../
├── partie4/
│   ├── iou_loss_impact_C1.png
│   ├── iou_distribution_*.png
│   ├── iou_summary.csv
│   └── heatmaps/
│       ├── P1_B/
│       ├── P1_C1/
│       ├── .../
│       └── P4_C1_IoUloss/
```

### Metriques

| Metrique | Description |
|----------|-------------|
| Accuracy | Precision globale |
| F1-Score | Moyenne harmonique precision/rappel |
| IoU | Alignement attention/masque (0-1) |

## Details techniques

### Architecture CrossViT

- 2 branches paralleles (Small et Large)
- Cross-Attention pour echange d'information
- Patch size: 16x16
- Embedding dimension: 192, 384
- Profondeur: `[1, 4, 0]` par branche
- 6 tetes d'attention

### Attention Rollout

1. Extraction des cartes d'attention de chaque couche
2. Produit cumulatif des attentions normalisees
3. Interpolation a la resolution image (224x224)
4. Normalisation min-max [0, 1]

### Ponderation par masque

Pour chaque patch p:

```
w_p = (epsilon + r_p)^gamma
```

- `r_p`: ratio de pixels masques dans le patch
- `epsilon = 0.01`: stabilisation numerique
- `gamma = 2.0`: exposant d'amplification

### Hyperparametres

```python
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_SPLIT = 0.8
LAMBDA_IOU = 0.1
SEED = 42
```

## Licence

Code CrossViT base sur IBM Research (Apache 2.0).