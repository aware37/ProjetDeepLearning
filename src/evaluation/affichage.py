import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_training_curves(history, config_name, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid()
    
    # F1-Score
    axes[2].plot(history['train_f1'], label='Train')
    axes[2].plot(history['val_f1'], label='Val')
    axes[2].set_title('F1-Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid()
    
    plt.suptitle(f'Configuration {config_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder dans results/
    output_path = os.path.join(output_dir, f'curves_config_{config_name}.png')
    plt.savefig(output_path, dpi=150)
    print(f"Courbes sauvegardées : {output_path}")
    plt.close()


def plot_comparison_configs(all_histories, output_dir='./results'):
    """Compare les 4 configurations (A, B, C1, C2) sur un même graphique."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = list(all_histories.keys())
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C1': '#2ca02c', 'C2': '#d62728',
               'C1_baseline': '#8c564b', 'C1_iou': '#e377c2'}
    
    # Val Loss
    ax = axes[0, 0]
    for config in configs:
        ax.plot(all_histories[config]['val_loss'], label=f'Config {config}', 
                color=colors[config], linewidth=2)
    ax.set_title('Validation Loss - Comparison', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid()
    
    # Val Accuracy
    ax = axes[0, 1]
    for config in configs:
        ax.plot(all_histories[config]['val_acc'], label=f'Config {config}', 
                color=colors[config], linewidth=2)
    ax.set_title('Validation Accuracy - Comparison', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid()
    
    ax = axes[1, 0]
    for config in configs:
        ax.plot(all_histories[config]['val_f1'], label=f'Config {config}', 
                color=colors[config], linewidth=2)
    ax.set_title('Validation F1-Score - Comparison', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid()
    
    ax = axes[1, 1]
    final_accs = [max(all_histories[config]['val_acc']) for config in configs]
    final_f1s = [max(all_histories[config]['val_f1']) for config in configs]
    
    x = range(len(configs))
    width = 0.35
    ax.bar([i - width/2 for i in x], final_accs, width, label='Best Accuracy', 
           color='skyblue')
    ax.bar([i + width/2 for i in x], final_f1s, width, label='Best F1-Score', 
           color='lightcoral')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Final Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Config {c}' for c in configs])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'comparison_all_configs.png')
    plt.savefig(output_path, dpi=150)
    print(f"Comparaison sauvegardée : {output_path}")
    plt.close()


def save_results_csv(histories, output_dir='./results'):
    """Sauvegarde les résultats d'entraînement dans des fichiers CSV"""
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    for config, history in histories.items():
        df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'train_f1': history['train_f1'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc'],
            'val_f1': history['val_f1']
        })
        
        output_path = os.path.join(output_dir, f'results_config_{config}.csv')
        df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés : {output_path}")

    
def plot_confusion_matrix(y_true, y_pred, classes, config_name, output_dir='./results'):
    """Trace et sauvegarde la matrice de confusion"""
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - Config {config_name}', fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'confusion_matrix_config_{config_name}.png')
    plt.savefig(output_path, dpi=150)
    print(f"Matrice de confusion sauvegardée : {output_path}")
    plt.close()


def plot_attention_heatmaps(model, dataloader, config, device, num_samples=5, output_dir='./results/heatmaps'):
    """Genere et sauvegarde des heatmaps d'attention pour des échantillons"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0
    
    with torch.no_grad():
        for inputs_non_seg, inputs_seg, mask, labels in dataloader:
            if sample_count >= num_samples:
                break
            
            if config == 'A':
                input_S, input_L, mask_input = inputs_non_seg, inputs_non_seg, None
            elif config == 'B':
                input_S, input_L, mask_input = inputs_seg, inputs_seg, mask
            elif config == 'C1':
                input_S, input_L, mask_input = inputs_non_seg, inputs_seg, mask
            elif config == 'C2':
                input_S, input_L, mask_input = inputs_seg, inputs_non_seg, mask
            else:
                continue
            
            input_S = input_S.to(device)
            input_L = input_L.to(device)
            if mask_input is not None:
                mask_input = mask_input.to(device)
            
            _ = model(input_S, input_L, mask=mask_input)
            
            heatmap = model.get_heatmap(branch_idx=0)
            
            if heatmap is None:
                continue
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = input_L[0].cpu() * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            heatmap_np = heatmap[0].cpu().numpy()
            
            if mask_input is not None:
                mask_np = mask_input[0, 0].cpu().numpy() if mask_input.dim() == 4 else mask_input[0].cpu().numpy()
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            else:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(img)
            axes[0].set_title('Image Originale')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap_np, cmap='jet')
            axes[1].set_title('Attention Rollout')
            axes[1].axis('off')
            
            axes[2].imshow(img)
            axes[2].imshow(heatmap_np, cmap='jet', alpha=0.5)
            axes[2].set_title('Superposition')
            axes[2].axis('off')
            
            if mask_input is not None:
                axes[3].imshow(mask_np, cmap='gray')
                axes[3].set_title('Masque GT')
                axes[3].axis('off')
            
            plt.suptitle(f'Config {config} - Sample {sample_count+1} - Label: {labels[0].item()}', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'heatmap_{config}_sample_{sample_count}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
    
    print(f"{sample_count} heatmaps sauvegardées dans {output_dir}")


def plot_iou_distribution(iou_scores, config, output_dir='./results'):
    """Trace et sauvegarde la distribution des scores IoU"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.hist(iou_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(np.mean(iou_scores), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(iou_scores):.3f}')
    plt.axvline(np.mean(iou_scores) + np.std(iou_scores), color='orange', linestyle=':', 
                label=f'Std: ±{np.std(iou_scores):.3f}')
    plt.axvline(np.mean(iou_scores) - np.std(iou_scores), color='orange', linestyle=':')
    plt.xlabel('IoU Score')
    plt.ylabel('Fréquence')
    plt.title(f'Distribution IoU - Config {config}', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(output_dir, f'iou_distribution_{config}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution IoU sauvegardée: {save_path}")


def plot_iou_loss_comparison(history_baseline, history_iou, config, output_dir='./results'):
    """Compare IoU loss sur les performances"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history_baseline['val_acc'], label='Sans IoU loss', linewidth=2, color='blue')
    axes[0, 0].plot(history_iou['val_acc'], label='Avec IoU loss', linewidth=2, color='red')
    axes[0, 0].set_title('Validation Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(history_baseline['val_f1'], label='Sans IoU loss', linewidth=2, color='blue')
    axes[0, 1].plot(history_iou['val_f1'], label='Avec IoU loss', linewidth=2, color='red')
    axes[0, 1].set_title('Validation F1-Score', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(history_baseline['val_loss'], label='Sans IoU loss', linewidth=2, color='blue')
    axes[1, 0].plot(history_iou['val_loss'], label='Avec IoU loss', linewidth=2, color='red')
    axes[1, 0].set_title('Validation Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    if 'val_iou_loss' in history_iou:
        axes[1, 1].plot(history_iou['val_iou_loss'], label='IoU Loss', linewidth=2, color='orange')
        axes[1, 1].set_title('IoU Loss Evolution', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU Loss (1 - IoU)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'Impact IoU Loss - Config {config}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'iou_loss_impact_{config}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparaison IoU loss sauvegardée: {save_path}")