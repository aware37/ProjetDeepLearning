import sys
import os
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import pandas as pd

from data.dataset import create_dataloaders, set_seed
from models.crossvit_3 import crossvit_part2, crossvit_small_224
from training.training import train_model_crossvit
from evaluation.affichage import plot_attention_heatmaps, plot_iou_distribution, plot_iou_loss_comparison
from evaluation.evaluate_iou import evaluate_iou 

def main_partie4():
    """
    Partie 4: Heatmaps, Rollout & IoU
    a) Attention rollout (implémenté dans le modèle)
    b) Heatmaps pour quelques échantillons (visualisation)
    c) IoU moyen ± std pour chaque config (A/B/C1/C2 + variante même résolution)
    d) Entraîner UN modèle avec IoU loss et comparer
    """
    
    # Configuration
    CSV_FILE = './data/raw/prepared_dataset.csv'
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    IMG_SIZE = 224
    
    output_dir = './results/partie4'
    checkpoint_dir = './checkpoints/partie4'
    
    # Setup
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/heatmaps', exist_ok=True)
    
    # Charger les DataLoaders
    print("Chargement des données...")
    train_loader, val_loader = create_dataloaders(
        csv_file=CSV_FILE,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        img_size=IMG_SIZE,
        seed=SEED
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}\n")
    
    # PARTIE 4c : Évaluation IoU des modèles entraînés
    
    # Modèles à évaluer
    modeles_a_evaluer = [
        # Partie 1 (B, C1, C2 seulement)
        {'name': 'P1_B',  'path': './checkpoints/partie1/best_model_config_B.pth',  'config': 'B',  'part': 1},
        {'name': 'P1_C1', 'path': './checkpoints/partie1/best_model_config_C1.pth', 'config': 'C1', 'part': 1},
        {'name': 'P1_C2', 'path': './checkpoints/partie1/best_model_config_C2.pth', 'config': 'C2', 'part': 1},
        
        # Partie 2 (même résolution)
        {'name': 'P2_B',  'path': './checkpoints/partie2/best_model_config_B.pth',  'config': 'B',  'part': 2},
        {'name': 'P2_C1', 'path': './checkpoints/partie2/best_model_config_C1.pth', 'config': 'C1', 'part': 2},
        {'name': 'P2_C2', 'path': './checkpoints/partie2/best_model_config_C2.pth', 'config': 'C2', 'part': 2},
        
        # Partie 3 (pondération par patch - C1/C2 seulement)
        {'name': 'P3_C1', 'path': './checkpoints/partie3/best_model_config_C1.pth', 'config': 'C1', 'part': 3},
        {'name': 'P3_C2', 'path': './checkpoints/partie3/best_model_config_C2.pth', 'config': 'C2', 'part': 3},
    ]
    
    iou_results = []
    
    for modele in modeles_a_evaluer:
        name = modele['name']
        path = modele['path']
        config = modele['config']
        part = modele.get('part', 1)
        
        print(f"\n{'-'*60}")
        print(f"Évaluation: {name} (Config {config})")
        print(f"{'-'*60}")
        
        if not os.path.exists(path):
            print(f"Checkpoint introuvable: {path}")
            continue
        
        # Charger le modèle
        if part == 1:
            model = crossvit_small_224(num_classes=2, pretrained=False)
        else:
            model = crossvit_part2(num_classes=2, pretrained=False)
        
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()
            print(f"Modèle chargé")
        except Exception as e:
            print(f"Erreur: {e}")
            continue
        
        # IoU
        mean_iou, std_iou, iou_scores = evaluate_iou(
            model, val_loader, config, device, threshold=0.8
        )
        
        iou_results.append({
            'Model': name,
            'Partie': part,
            'Config': config,
            'IoU_mean': mean_iou,
            'IoU_std': std_iou,
            'n_samples': len(iou_scores)
        })
        
        # Distribution IoU
        if len(iou_scores) > 0:
            plot_iou_distribution(iou_scores, name, output_dir=output_dir)
        
        # Heatmaps (4b) 3 échantillons
        print(f"Génération de 3 heatmaps...")
        heatmap_dir = f'{output_dir}/heatmaps/{name}'
        os.makedirs(heatmap_dir, exist_ok=True)
        plot_attention_heatmaps(
            model, val_loader, config, device,
            num_samples=3, output_dir=heatmap_dir
        )
    
    # PARTIE 4d : Entraînement avec IoU loss
    
    # Meilleure config de la Partie 2
    config_iou = 'C1'
    lambda_iou = 0.1
        
    model_iou = crossvit_part2(num_classes=2, pretrained=False).to(device)
    optimizer_iou = torch.optim.AdamW(model_iou.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_iou = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iou, T_max=NUM_EPOCHS)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
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
        lambda_iou=lambda_iou,
        checkpoint_dir=checkpoint_dir
    )
    
    # Évaluation du modèle avec IoU loss
    print(f"\nÉvaluation finale du modèle avec IoU loss...")
    mean_iou_iou, std_iou_iou, iou_scores_iou = evaluate_iou(
        trained_model_iou, val_loader, config_iou, device, threshold=0.8
    )
    
    iou_results.append({
        'Model': f'P4_{config_iou}_IoUloss',
        'Partie': 4,
        'Config': config_iou,
        'IoU_mean': mean_iou_iou,
        'IoU_std': std_iou_iou,
        'n_samples': len(iou_scores_iou)
    })
    
    # Distribution et heatmaps
    if len(iou_scores_iou) > 0:
        plot_iou_distribution(iou_scores_iou, f'P4_{config_iou}_IoUloss', output_dir=output_dir)
    
    heatmap_dir_iou = f'{output_dir}/heatmaps/P4_{config_iou}_IoUloss'
    os.makedirs(heatmap_dir_iou, exist_ok=True)
    plot_attention_heatmaps(
        trained_model_iou, val_loader, config_iou, device,
        num_samples=3, output_dir=heatmap_dir_iou
    )
    
    # Comparaison baseline vs IoU loss
    print("Comparaison: Baseline vs IoU loss")
    
    baseline_name = f'P2_{config_iou}'
    baseline_result = next((r for r in iou_results if r['Model'] == baseline_name), None)
    
    if baseline_result:
        print(f"Baseline ({baseline_name}):")
        print(f"  IoU = {baseline_result['IoU_mean']:.4f} ± {baseline_result['IoU_std']:.4f}")
        print(f"\nAvec IoU loss:")
        print(f"  IoU = {mean_iou_iou:.4f} ± {std_iou_iou:.4f}")
        gain = mean_iou_iou - baseline_result['IoU_mean']
        print(f"\nGain IoU: {gain:+.4f}")
        
        # Graphique de comparaison
        baseline_csv = f'./results/partie2/results_config_{config_iou}.csv'
        if os.path.exists(baseline_csv):
            df_baseline = pd.read_csv(baseline_csv)
            history_baseline = {k: df_baseline[k].tolist() for k in df_baseline.columns if k != 'epoch'}
            plot_iou_loss_comparison(history_baseline, history_iou, config_iou, output_dir=output_dir)
        
    # Tableau récapitulatif IoU
    iou_df = pd.DataFrame(iou_results)
    print("\nTableau IoU:")
    print(iou_df.to_string(index=False))
    
    iou_df.to_csv(f'{output_dir}/iou_summary.csv', index=False)
    print(f"\nSauvegardé: {output_dir}/iou_summary.csv")
    
    # Meilleur IoU
    best_iou = iou_df.loc[iou_df['IoU_mean'].idxmax()]
    print(f"\nMeilleur IoU: {best_iou['Model']} → {best_iou['IoU_mean']:.4f}")
        
    return iou_df


if __name__ == '__main__':
    iou_results = main_partie4()
