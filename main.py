import sys
import os
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import time
import copy
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from data.dataset import create_dataloaders, set_seed
from models.crossvit import crossvit_small_224, crossvit_part2, crossvit_part2_sym
from training.training import train_model_crossvit
from evaluation.affichage import plot_training_curves, plot_comparison_configs, save_results_csv, plot_confusion_matrix

def main():
    """Lance l'entraînement pour les 4 configurations A, B, C1, C2."""
    
    # Configuration
    CSV_FILE = './data/raw/prepared_dataset.csv'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    IMG_SIZE = 224
    
    # Setup
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    # Créer le dossier results
    os.makedirs('./results', exist_ok=True)
    
    # Charger les DataLoaders (une seule fois)
    print("\n Chargement des données...")
    train_loader, val_loader = create_dataloaders(
        csv_file=CSV_FILE,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        img_size=IMG_SIZE,
        seed=SEED
    )
    print(f"✓ Train set: {len(train_loader.dataset)} images")
    print(f"✓ Val set: {len(val_loader.dataset)} images")
    

    # Dictionnaire pour stocker les résultats
    all_histories = {}
    all_models = {}
    all_preds = {}
    
    # Boucle sur les 4 configurations
    # configs = ['A', 'B', 'C1', 'C2'] # Configurations partie 1
    # configs = ['A', 'B', 'C1', 'C2'] # Configurations partie 2 asymetrique
    # configs = ['A', 'B', 'C1'] # Configurations partie 2 symetrique
    configs = ['Partie3_C'] # Configurations parte 3

    for config in configs:
        print(f"\n{'='*70}")
        print(f" Configuration {config}")
        print(f"{'='*70}")
        
        # Créer le modèle
        # model = crossvit_small_224(num_classes=2, pretrained=False) # Modele partie 1
        model = crossvit_part2(num_classes=2, pretrained=False) # Modele partie 2 asymetrique
        # model = crossvit_part2_sym(num_classes=2, pretrained=False) # Modele partie 2 symetrique
        model = model.to(device)
        
        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS
        )
        
        # DataLoaders
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Entraîner
        trained_model, history, final_preds = train_model_crossvit(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            num_epochs=NUM_EPOCHS
        )
        
        # Sauvegarder
        all_histories[config] = history
        all_models[config] = trained_model
        all_preds[config] = final_preds
        
        # Courbes individuelles
        plot_training_curves(history, config, output_dir='./results')
        plot_confusion_matrix(
            y_true=final_preds['labels'],
            y_pred=final_preds['preds'],
            config_name=config,
            classes=['Absence épines', 'Présence épines'],
            output_dir='./results'
        )

    # Comparaison globale
    print(f"\n{'='*70}")
    print("Génération des courbes de comparaison...")
    print(f"{'='*70}")
    
    plot_comparison_configs(all_histories, output_dir='./results')
    save_results_csv(all_histories, output_dir='./results')
    
    # Résumé final
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ FINAL")
    print(f"{'='*70}")
    
    summary_data = []
    for config in configs:
        history = all_histories[config]
        best_acc = max(history['val_acc'])
        best_f1 = max(history['val_f1'])
        best_epoch = history['val_f1'].index(best_f1) + 1
        
        print(f"\nConfiguration {config}:")
        print(f"  ✓ Best Val Accuracy: {best_acc:.4f}")
        print(f"  ✓ Best Val F1-Score: {best_f1:.4f}")
        print(f"  ✓ Epoch: {best_epoch}")
        
        summary_data.append({
            'Config': config,
            'Best Accuracy': best_acc,
            'Best F1': best_f1,
            'Best Epoch': best_epoch
        })
    
    # Sauvegarder le résumé
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('./results/summary.csv', index=False)
    print(f"\n✓ Résumé sauvegardé: ./results/summary.csv")
    
    print(f"\n{'='*70}")
    print(" Entraînement terminé!")
    print(f"{'='*70}\n")
    
    return all_histories, all_models


if __name__ == '__main__':
    all_histories, all_models = main()