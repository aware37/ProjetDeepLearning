import sys
import os
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import pandas as pd

from data.dataset import create_dataloaders, set_seed
from models.crossvit_3 import crossvit_part2
from training.training import train_model_crossvit
from evaluation.affichage import plot_attention_heatmaps, plot_training_curves, plot_comparison_configs, save_results_csv, plot_confusion_matrix, plot_iou_distribution, plot_iou_loss_comparison
from evaluation.evaluate_iou import evaluate_iou 

def main_partie4():
    """Partie 4: Heatmaps, Rollout & IoU."""
    
    # Configuration
    CSV_FILE = './data/raw/prepared_dataset.csv'
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    IMG_SIZE = 224
    
    # Configurations à tester
    configs = ['A', 'B', 'C1', 'C2']
    model_fn = lambda: crossvit_part2(num_classes=2, pretrained=False)
    output_dir = './results/partie4'
    checkpoint_dir = './checkpoints/partie4'
    
    # Setup
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'-'*10}")
    print(f"Partie 4: Heatmaps, Rollout & IoU")
    print(f"{'-'*10}")
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/heatmaps', exist_ok=True)
    
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
    all_iou_results = {}
    
    # Baseline sans IoU loss
    print(f"\n{'-'*10}")
    print("Training BASELINE sans IoU loss")
    print(f"{'-'*10}")
    
    for config in configs:
        print(f"\n{'-'*10}")
        print(f"Configuration {config} - BASELINE")
        print(f"{'-'*10}")
        
        # Créer et entraîner le modèle
        model = model_fn().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        trained_model, history, final_preds = train_model_crossvit(
            model=model,
            dataloaders=dataloaders,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            num_epochs=NUM_EPOCHS,
            use_iou_loss=False,
            checkpoint_dir=checkpoint_dir
        )
        
        key = f'{config}_baseline'
        all_histories[key] = history
        all_models[key] = trained_model
        all_preds[key] = final_preds
        
        # Sauvegardes
        plot_training_curves(history, key, output_dir=output_dir)
        plot_confusion_matrix(
            y_true=final_preds['labels'],
            y_pred=final_preds['preds'],
            config_name=key,
            classes=['Absence épines', 'Présence épines'],
            output_dir=output_dir
        )
        
        # Évaluation IoU
        print(f"\nÉvaluation IoU pour {config}...")
        mean_iou, std_iou, iou_scores = evaluate_iou(
            trained_model, val_loader, config, device, threshold=0.8
        )
        all_iou_results[key] = {'mean': mean_iou, 'std': std_iou, 'scores': iou_scores}
        
        if len(iou_scores) > 0:
            plot_iou_distribution(iou_scores, key, output_dir=output_dir)
        
        # Heatmaps
        print(f"Génération des heatmaps pour {config}...")
        heatmap_dir = f'{output_dir}/heatmaps/{key}'
        os.makedirs(heatmap_dir, exist_ok=True)
        plot_attention_heatmaps(
            trained_model, val_loader, config, device,
            num_samples=5, output_dir=heatmap_dir
        )
    
    # Training avec IoU loss
    print(f"\n{'-'*10}")
    print("Training avec IoU loss")
    print(f"{'-'*10}")
    
    config_iou = 'C1'
    
    model_iou = model_fn().to(device)
    optimizer_iou = torch.optim.AdamW(model_iou.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_iou = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iou, T_max=NUM_EPOCHS)
    
    trained_model_iou, history_iou, preds_iou = train_model_crossvit(
        model=model_iou,
        dataloaders=dataloaders,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer_iou,
        scheduler=scheduler_iou,
        config=config_iou,
        device=device,
        num_epochs=NUM_EPOCHS,
        use_iou_loss=True,
        lambda_iou=0.1,
        checkpoint_dir=checkpoint_dir
    )
    
    key_iou = f'{config_iou}_iou'
    all_histories[key_iou] = history_iou
    all_models[key_iou] = trained_model_iou
    all_preds[key_iou] = preds_iou
    
    # Sauvegardes
    plot_training_curves(history_iou, key_iou, output_dir=output_dir)
    plot_confusion_matrix(
        y_true=preds_iou['labels'],
        y_pred=preds_iou['preds'],
        config_name=key_iou,
        classes=['Absence épines', 'Présence épines'],
        output_dir=output_dir
    )
    
    # Évaluation IoU
    mean_iou_iou, std_iou_iou, iou_scores_iou = evaluate_iou(
        trained_model_iou, val_loader, config_iou, device, threshold=0.8
    )
    all_iou_results[key_iou] = {'mean': mean_iou_iou, 'std': std_iou_iou, 'scores': iou_scores_iou}
    
    # Comparaison baseline vs IoU loss
    baseline_key = f'{config_iou}_baseline'
    if baseline_key in all_histories:
        plot_iou_loss_comparison(
            all_histories[baseline_key], history_iou, config_iou, output_dir=output_dir
        )
    
    # Heatmaps
    heatmap_dir_iou = f'{output_dir}/heatmaps/{key_iou}'
    os.makedirs(heatmap_dir_iou, exist_ok=True)
    plot_attention_heatmaps(
        trained_model_iou, val_loader, config_iou, device,
        num_samples=5, output_dir=heatmap_dir_iou
    )
    
    # Tableaux récapitulatifs
    print(f"\n{'-'*10}")
    print("Génération des résultats...")
    print(f"{'-'*10}")
    
    plot_comparison_configs(all_histories, output_dir=output_dir)
    save_results_csv(all_histories, output_dir=output_dir)
    
    # Tableau IoU
    print(f"\n{'-'*10}")
    print("RÉSULTATS IoU")
    print(f"{'-'*10}")
    
    iou_data = []
    for cfg, data in all_iou_results.items():
        print(f"  Config {cfg}: IoU = {data['mean']:.4f} +/- {data['std']:.4f}")
        iou_data.append({'Config': cfg, 'IoU_mean': data['mean'], 'IoU_std': data['std']})
    
    pd.DataFrame(iou_data).to_csv(f'{output_dir}/iou_summary.csv', index=False)
    
    # Résumé final
    print(f"\n{'-'*10}")
    print("RÉSUMÉ FINAL - PARTIE 4")
    print(f"{'-'*10}")
    
    summary_data = []
    for config, history in all_histories.items():
        best_acc = max(history['val_acc'])
        best_f1 = max(history['val_f1'])
        best_epoch = history['val_f1'].index(best_f1) + 1
        iou_mean = all_iou_results.get(config, {}).get('mean', None)
        
        print(f"\nConfiguration {config}:")
        print(f"Best Val Accuracy: {best_acc:.4f}")
        print(f"Best Val F1-Score: {best_f1:.4f}")
        print(f"Best Epoch: {best_epoch}")
        if iou_mean is not None:
            print(f"IoU: {iou_mean:.4f}")
        
        summary_data.append({
            'Config': config,
            'Best_Accuracy': best_acc,
            'Best_F1': best_f1,
            'Best_Epoch': best_epoch,
            'IoU_mean': iou_mean
        })
    
    pd.DataFrame(summary_data).to_csv(f'{output_dir}/summary.csv', index=False)
    
    # Tableau Ablation
    print(f"\n{'-'*10}")
    print("TABLEAU ABLATION")
    print(f"{'-'*10}")
    
    ablation_data = []
    for config, history in all_histories.items():
        ablation_data.append({
            'Config': config,
            'Masque_pondere': 'Oui' if 'Partie3' in config else 'Non',
            'IoU_loss': 'Oui' if '_iou' in config else 'Non',
            'Best_Acc': f"{max(history['val_acc']):.4f}",
            'Best_F1': f"{max(history['val_f1']):.4f}",
            'IoU_mean': f"{all_iou_results.get(config, {}).get('mean', 0):.4f}"
        })
    
    ablation_df = pd.DataFrame(ablation_data)
    ablation_df.to_csv(f'{output_dir}/ablation_study.csv', index=False)
    print(ablation_df.to_string(index=False))
    
    print(f"\n{'-'*10}")
    print(f"PARTIE 4 terminée!")
    print(f"Résultats sauvegardés dans: {output_dir}")
    print(f"Checkpoints sauvegardés dans: {checkpoint_dir}")
    print(f"{'-'*10}\n")
    
    return all_histories, all_models, all_iou_results


if __name__ == '__main__':
    all_histories, all_models, all_iou_results = main_partie4()