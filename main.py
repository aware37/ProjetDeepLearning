import sys
import os
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import pandas as pd

from data.dataset import create_dataloaders, set_seed
from training.training import train_model_crossvit
from evaluation.affichage import plot_training_curves, plot_comparison_configs, save_results_csv, plot_confusion_matrix


def main():
    """Lance l'entraînement pour les configurations choisies."""
    
    # Configuration
    CSV_FILE = './data/raw/prepared_dataset.csv'
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    IMG_SIZE = 224
    
    # CHOIX DE LA PARTIE
    
    # Partie 1: CrossViT original
    from models.crossvit import crossvit_small_224
    configs = ['A', 'B', 'C1', 'C2']
    model_used = lambda: crossvit_small_224(num_classes=2, pretrained=False)
    output_dir = './results/partie1'
    checkpoint_dir = './checkpoints/partie1'
    
    # Partie 2: CrossViT 
    # from models.crossvit import crossvit_part2
    # configs = ['A', 'B', 'C1', 'C2']
    # model_used = lambda: crossvit_part2(num_classes=2, pretrained=False)
    # output_dir = './results/partie2'
    # checkpoint_dir = './checkpoints/partie2'
    
    # Partie 3: Pondération par masque
    # from models.crossvit_2 import crossvit_part2
    # configs = ['C1', 'C2']
    # model_used = lambda: crossvit_part2(num_classes=2, pretrained=False)
    # output_dir = './results/partie3'
    # checkpoint_dir = './checkpoints/partie3'
        
    # Setup
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Charger les DataLoaders
    print("\nChargement des données...")
    train_loader, val_loader = create_dataloaders(
        csv_file=CSV_FILE,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        img_size=IMG_SIZE,
        seed=SEED
    )
    print(f"Train set: {len(train_loader.dataset)} images")
    print(f"Val set: {len(val_loader.dataset)} images")
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Dictionnaires pour stocker les résultats
    all_histories = {}
    all_models = {}
    all_preds = {}
    
    for config in configs:
        print(f"\n{'-'*10}")
        print(f"Configuration {config}")
        
        # Créer le modèle
        model = model_used()
        model = model.to(device)
        
        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS
        )
        
        # Entraîner
        trained_model, history, final_preds = train_model_crossvit(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            num_epochs=NUM_EPOCHS,
            checkpoint_dir=checkpoint_dir
        )
        
        # Sauvegarder
        all_histories[config] = history
        all_models[config] = trained_model
        all_preds[config] = final_preds
        
        # Courbes individuelles
        plot_training_curves(history, config, output_dir=output_dir)
        plot_confusion_matrix(
            y_true=final_preds['labels'],
            y_pred=final_preds['preds'],
            config_name=config,
            classes=['Absence épines', 'Présence épines'],
            output_dir=output_dir
        )
    
    # Comparaison des configurations    
    plot_comparison_configs(all_histories, output_dir=output_dir)
    save_results_csv(all_histories, output_dir=output_dir)
    
    # Résumé final
    print(f"\n{'-'*10}")
    print("RÉSUMÉ FINAL")
    
    summary_data = []
    for config in configs:
        history = all_histories[config]
        best_acc = max(history['val_acc'])
        best_f1 = max(history['val_f1'])
        best_epoch = history['val_f1'].index(best_f1) + 1
        
        print(f"\nConfiguration {config}:")
        print(f"Best Val Accuracy: {best_acc:.4f}")
        print(f"Best Val F1-Score: {best_f1:.4f}")
        print(f"Best Epoch: {best_epoch}")
        
        summary_data.append({
            'Config': config,
            'Best_Accuracy': best_acc,
            'Best_F1': best_f1,
            'Best_Epoch': best_epoch
        })
    
    pd.DataFrame(summary_data).to_csv(f'{output_dir}/summary.csv', index=False)
    print(f"\nRésumé sauvegardé: {output_dir}/summary.csv")
    
    print(f"\n{'-'*10}")
    print(f"Entraînement terminé!")
    print(f"Résultats sauvegardés dans: {output_dir}")
    print(f"Checkpoints sauvegardés dans: {checkpoint_dir}")
    print(f"{'-'*10}\n")
    
    return all_histories, all_models


if __name__ == '__main__':
    all_histories, all_models = main()