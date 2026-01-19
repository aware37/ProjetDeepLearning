import os
import matplotlib.pyplot as plt

def plot_training_curves(history, config_name, output_dir='./results'):
    """Plot les courbes d'apprentissage et les sauvegarde dans results/."""
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid()
    
    # F1-Score
    axes[2].plot(history['train_f1'], label='Train', marker='o')
    axes[2].plot(history['val_f1'], label='Val', marker='s')
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
    print(f"✓ Courbes sauvegardées : {output_path}")
    plt.close()


def plot_comparison_configs(all_histories, output_dir='./results'):
    """Compare les 4 configurations (A, B, C1, C2) sur un même graphique."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = list(all_histories.keys())
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C1': '#2ca02c', 'C2': '#d62728'}
    
    # Val Loss
    ax = axes[0, 0]
    for config in configs:
        ax.plot(all_histories[config]['val_loss'], label=f'Config {config}', 
                color=colors[config], marker='o', linewidth=2)
    ax.set_title('Validation Loss - Comparison', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid()
    
    # Val Accuracy
    ax = axes[0, 1]
    for config in configs:
        ax.plot(all_histories[config]['val_acc'], label=f'Config {config}', 
                color=colors[config], marker='o', linewidth=2)
    ax.set_title('Validation Accuracy - Comparison', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid()
    
    # Val F1-Score
    ax = axes[1, 0]
    for config in configs:
        ax.plot(all_histories[config]['val_f1'], label=f'Config {config}', 
                color=colors[config], marker='o', linewidth=2)
    ax.set_title('Validation F1-Score - Comparison', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid()
    
    # Bar chart final results
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
    
    # Sauvegarder
    output_path = os.path.join(output_dir, 'comparison_all_configs.png')
    plt.savefig(output_path, dpi=150)
    print(f"✓ Comparaison sauvegardée : {output_path}")
    plt.close()


def save_results_csv(histories, output_dir='./results'):
    """Sauvegarde les résultats en CSV pour chaque configuration."""
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
        print(f"✓ Résultats sauvegardés : {output_path}")