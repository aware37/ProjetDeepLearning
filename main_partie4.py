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
    
    Principe:
    - 4a, 4b, 4c: Evaluer les modeles DEJA entraines (parties 1, 2, 3)
    - 4d: Entrainer UN SEUL nouveau modele avec IoU loss
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
    print(f"\n{'-'*10}")
    print(f"PARTIE 4: Heatmaps, Rollout & IoU")
    print(f"{'-'*10}")
    print(f"Device: {device}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/heatmaps', exist_ok=True)
    
    # Charger les DataLoaders
    print("Chargement des donnees...")
    train_loader, val_loader = create_dataloaders(
        csv_file=CSV_FILE,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        img_size=IMG_SIZE,
        seed=SEED
    )
    print(f"Train set: {len(train_loader.dataset)} images")
    print(f"Val set: {len(val_loader.dataset)} images\n")
    
    # Evaluation des modeles existants (4a, 4b, 4c)
    print(f"{'='*10}")
    print("PHASE 1: Evaluation des modeles DEJA entraines")
    print(f"{'='*10}\n")
    
    # Liste des modeles a evaluer
    modeles_a_evaluer = [
        # Partie 1 (crossvit_small_224, dim=384)
        {'name': 'P1_A', 'path': './checkpoints/partie1/best_model_config_A.pth',  'config': 'A',  'part': 1},
        {'name': 'P1_B', 'path': './checkpoints/partie1/best_model_config_B.pth',  'config': 'B',  'part': 1},
        {'name': 'P1_C1','path': './checkpoints/partie1/best_model_config_C1.pth', 'config': 'C1', 'part': 1},
        {'name': 'P1_C2','path': './checkpoints/partie1/best_model_config_C2.pth', 'config': 'C2', 'part': 1},
        # Partie 2 (crossvit_part2, dim=384)
        {'name': 'P2_A', 'path': './checkpoints/partie2/best_model_config_A.pth',  'config': 'A',  'part': 2},
        {'name': 'P2_B', 'path': './checkpoints/partie2/best_model_config_B.pth',  'config': 'B',  'part': 2},
        {'name': 'P2_C1','path': './checkpoints/partie2/best_model_config_C1.pth', 'config': 'C1', 'part': 2},
        {'name': 'P2_C2','path': './checkpoints/partie2/best_model_config_C2.pth', 'config': 'C2', 'part': 2},
        # Partie 3 (crossvit_part2, dim=192)
        {'name': 'P3_C1','path': './checkpoints/partie3/best_model_config_C1.pth', 'config': 'C1', 'part': 3},
        {'name': 'P3_C2','path': './checkpoints/partie3/best_model_config_C2.pth', 'config': 'C2', 'part': 3},
    ]
    
    all_iou_results = {}
    
    for modele in modeles_a_evaluer:
        name = modele['name']
        path = modele['path']
        config = modele['config']
        part = modele.get('part', 1)
        
        print(f"\n{'-'*10}")
        print(f"Evaluation: {name} ({config})")
        print(f"Checkpoint: {path}")
        print(f"{'-'*10}")
        
        if not os.path.exists(path):
            print(f"Checkpoint introuvable, skip.")
            continue
        
        # Charger le modele
        if part == 1:
            model = crossvit_small_224(num_classes=2, pretrained=False).to(device)
        else:
            # crossvit_part2 doit accepter embed_dim ; sinon retire cet argument
            model = crossvit_part2(num_classes=2, pretrained=False).to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            print(f"Modele charge avec succes")
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            continue
        
        # Evaluation IoU (4c)
        print(f"\nEvaluation IoU...")
        mean_iou, std_iou, iou_scores = evaluate_iou(
            model, val_loader, config, device, threshold=0.8
        )
        all_iou_results[name] = {
            'mean': mean_iou, 
            'std': std_iou, 
            'scores': iou_scores,
            'config': config
        }
        
        # Distribution IoU
        if len(iou_scores) > 0:
            plot_iou_distribution(iou_scores, name, output_dir=output_dir)
        
        # Heatmaps (4b)
        print(f"Generation des heatmaps...")
        heatmap_dir = f'{output_dir}/heatmaps/{name}'
        os.makedirs(heatmap_dir, exist_ok=True)
        plot_attention_heatmaps(
            model, val_loader, config, device,
            num_samples=5, output_dir=heatmap_dir
        )
        
        print(f"Evaluation terminee pour {name}")
    
    # Entrainement avec IoU loss (4d)
    print(f"\n{'-'*10}")
    print("PHASE 2: Training AVEC IoU loss (Partie 4d)")
    print(f"{'-'*10}\n")
    
    config_iou = 'C1'
    
    print(f"Configuration choisie: {config_iou}")
    print(f"Lambda IoU: 0.1\n")
    
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
        lambda_iou=0.1,
        checkpoint_dir=checkpoint_dir
    )
    
    # Evaluation du modele avec IoU loss
    print(f"\nEvaluation finale du modele avec IoU loss...")
    mean_iou_iou, std_iou_iou, iou_scores_iou = evaluate_iou(
        trained_model_iou, val_loader, config_iou, device, threshold=0.8
    )
    
    key_iou = f'P4_{config_iou}_iou'
    all_iou_results[key_iou] = {
        'mean': mean_iou_iou, 
        'std': std_iou_iou, 
        'scores': iou_scores_iou,
        'config': config_iou
    }
    
    # Distribution IoU
    if len(iou_scores_iou) > 0:
        plot_iou_distribution(iou_scores_iou, key_iou, output_dir=output_dir)
    
    # Heatmaps
    heatmap_dir_iou = f'{output_dir}/heatmaps/{key_iou}'
    os.makedirs(heatmap_dir_iou, exist_ok=True)
    plot_attention_heatmaps(
        trained_model_iou, val_loader, config_iou, device,
        num_samples=5, output_dir=heatmap_dir_iou
    )
    
    # Comparaison baseline vs IoU loss
    baseline_keys = [f'P1_{config_iou}', f'P2_{config_iou}', f'P3_{config_iou}']
    for baseline_key in baseline_keys:
        if baseline_key in all_iou_results:
            print(f"\nComparaison {baseline_key} vs {key_iou}:")
            print(f"  Baseline IoU: {all_iou_results[baseline_key]['mean']:.4f}")
            print(f"  Avec IoU loss: {all_iou_results[key_iou]['mean']:.4f}")
            gain = all_iou_results[key_iou]['mean'] - all_iou_results[baseline_key]['mean']
            print(f"  Gain: {gain:+.4f}")
    
    #TABLEAUX RECAPITULATIFS
    print(f"\n{'-'*10}")
    print("GENERATION DES RESULTATS FINAUX")
    print(f"{'-'*10}\n")
    
    # Tableau IoU (4c)
    print(f"{'-'*10}")
    print("RESULTATS IoU (Partie 4c)")
    print(f"{'-'*10}")
    
    iou_data = []
    for name, data in all_iou_results.items():
        print(f"  {name:20s}: IoU = {data['mean']:.4f} +/- {data['std']:.4f}")
        iou_data.append({
            'Model': name, 
            'Config': data['config'],
            'IoU_mean': data['mean'], 
            'IoU_std': data['std']
        })
    
    iou_df = pd.DataFrame(iou_data)
    iou_df.to_csv(f'{output_dir}/iou_summary.csv', index=False)
    print(f"\nTableau IoU sauvegarde: {output_dir}/iou_summary.csv")
    
    # Tableau Ablation (4d)
    print(f"\n{'-'*10}")
    print("TABLEAU ABLATION (Partie 4d)")
    print(f"{'-'*10}")
    
    ablation_data = []
    
    # Partie 1
    for cfg in ['A', 'B', 'C1', 'C2']:
        key = f'P1_{cfg}'
        if key in all_iou_results:
            ablation_data.append({
                'Model': key,
                'Partie': '1',
                'Config': cfg,
                'Masque_pondere': 'Non',
                'IoU_loss': 'Non',
                'IoU_mean': f"{all_iou_results[key]['mean']:.4f}"
            })
    
    # Partie 2
    for cfg in ['A', 'B', 'C1', 'C2']:
        key = f'P2_{cfg}'
        if key in all_iou_results:
            ablation_data.append({
                'Model': key,
                'Partie': '2',
                'Config': cfg,
                'Masque_pondere': 'Non',
                'IoU_loss': 'Non',
                'IoU_mean': f"{all_iou_results[key]['mean']:.4f}"
            })
    
    # Partie 3
    for cfg in ['C1', 'C2']:
        key = f'P3_{cfg}'
        if key in all_iou_results:
            ablation_data.append({
                'Model': key,
                'Partie': '3',
                'Config': cfg,
                'Masque_pondere': 'Oui',
                'IoU_loss': 'Non',
                'IoU_mean': f"{all_iou_results[key]['mean']:.4f}"
            })
    
    # Partie 4 avec IoU loss
    if key_iou in all_iou_results:
        ablation_data.append({
            'Model': key_iou,
            'Partie': '4',
            'Config': config_iou,
            'Masque_pondere': 'Non',
            'IoU_loss': 'Oui',
            'IoU_mean': f"{all_iou_results[key_iou]['mean']:.4f}"
        })
    
    ablation_df = pd.DataFrame(ablation_data)
    ablation_df.to_csv(f'{output_dir}/ablation_study.csv', index=False)
    print(ablation_df.to_string(index=False))
    print(f"\nTableau ablation sauvegarde: {output_dir}/ablation_study.csv")
    
    # Sauvegarde des IoU détaillés
    print(f"\n{'-'*10}")
    print("SAUVEGARDE DES IoU DETAILLES")
    print(f"{'-'*10}")
    
    # Sauvegarder tous les scores IoU par modèle
    all_iou_scores_df = []
    for name, data in all_iou_results.items():
        if len(data['scores']) > 0:
            for idx, score in enumerate(data['scores']):
                all_iou_scores_df.append({
                    'Model': name,
                    'Config': data['config'],
                    'Sample_ID': idx,
                    'IoU_Score': score
                })
    
    if all_iou_scores_df:
        iou_scores_df = pd.DataFrame(all_iou_scores_df)
        iou_scores_df.to_csv(f'{output_dir}/iou_scores_detailed.csv', index=False)
        print(f"IoU détaillés sauvegardés: {output_dir}/iou_scores_detailed.csv")
        print(f"Total: {len(iou_scores_df)} scores individuels")
    
    # Resume final
    print(f"\n{'-'*10}")
    print("RESUME - PARTIE 4")
    print(f"{'-'*10}")
    print(f"\n{len(all_iou_results)} modeles evalues")
    print(f"Heatmaps generees dans: {output_dir}/heatmaps/")
    print(f"Resultats sauvegardes dans: {output_dir}/")
    print(f"Checkpoints sauvegardes dans: {checkpoint_dir}/")
    
    # Meilleur IoU
    if all_iou_results:
        best_model = max(all_iou_results.items(), key=lambda x: x[1]['mean'])
        print(f"\nMeilleur IoU: {best_model[0]} avec {best_model[1]['mean']:.4f}")
    
    print(f"\n{'-'*10}\n")
    
    return all_iou_results


if __name__ == '__main__':
    all_iou_results = main_partie4()