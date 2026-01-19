import time
import os
import copy
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def train_model_crossvit(model, dataloaders, criterion, optimizer, scheduler, config, device, num_epochs=25):
    """
    Entraîne le modèle CrossViT avec routage des inputs selon la configuration.
    
    Args:
        model: Modèle CrossViT
        dataloaders: dict avec 'train' et 'val' DataLoaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config (str): 'A', 'B', 'C1', ou 'C2' pour diriger les inputs
        device: torch device
        num_epochs: Nombre d'epochs
    
    Returns:
        model: Meilleur modèle entraîné
        history: Dict avec les métriques
    """
    since = time.time()

    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = f"checkpoints/best_model_config_{config}.pth"
    
    # Historique pour les courbes
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs} [Config {config}]')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Barre de progression
            pbar = tqdm(dataloaders[phase], desc=f'{phase.upper()}')
            
            # Iterate over data
            for inputs_non_seg, inputs_seg, labels in pbar:
                inputs_non_seg = inputs_non_seg.to(device)
                inputs_seg = inputs_seg.to(device)
                labels = labels.to(device).long()

                # On décide qui va dans la branche Large (L) et Small (S)
                if config == 'A':
                    # Config A : images non segmentées uniquement
                    input_L, input_S = inputs_non_seg, inputs_non_seg
                elif config == 'B':
                    # Config B : images segmentées uniquement
                    input_L, input_S = inputs_seg, inputs_seg
                elif config == 'C1':
                    # Config C1 : segmentées -> Large, non segmentées -> Small
                    input_L, input_S = inputs_seg, inputs_non_seg
                elif config == 'C2':
                    # Config C2 : segmentées -> Small, non segmentées -> Large
                    input_L, input_S = inputs_non_seg, inputs_seg
                else:
                    raise ValueError(f"Config {config} inconnue. Utilisez 'A', 'B', 'C1' ou 'C2'")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Le modèle prend 2 entrées : (input_small, input_large)
                    outputs = model(input_S, input_L)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize (seulement en train)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs_non_seg.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Pour le F1-score
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})

            if phase == 'train':
                scheduler.step()

            # Calcul des métriques
            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}')
            
            # Sauvegarde l'historique
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_f1'].append(epoch_f1)

            # Sauvegarde du meilleur modèle (basé sur F1-score en validation)
            # Note: On peut aussi utiliser Accuracy si souhaité
            if phase == 'val' and epoch_f1 > best_val_f1:
                best_val_f1 = epoch_f1
                best_val_acc = epoch_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), best_model_path)
                print(f'✓ Meilleur modèle sauvegardé (Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f})')

    time_elapsed = time.time() - since
    print(f'\n{"-"*10}')
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Accuracy: {best_val_acc:.4f}')
    print(f'Best val F1-Score: {best_val_f1:.4f}')
    print(f'{"-"*10}')

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model, history