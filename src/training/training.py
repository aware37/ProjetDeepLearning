import time
import os
import copy
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def train_model_crossvit(model, dataloaders, criterion, optimizer, scheduler, config, device, 
                         num_epochs=25, use_iou_loss=False, lambda_iou=0.1, checkpoint_dir='checkpoints'):
    """
    Entraîne le modèle CrossViT avec routage des inputs selon la configuration.
    
    Args:
        model: Modèle CrossViT
        dataloaders: dict avec 'train' et 'val' DataLoaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config (str): 'A', 'B', 'C1', 'C2'
        device: torch device
        num_epochs: Nombre d'epochs
        use_iou_loss: Activer la perte IoU
        lambda_iou: Poids de la perte IoU
        checkpoint_dir: Répertoire pour sauvegarder les modèles
    
    Returns:
        model: Meilleur modèle entraîné
        history: Dict avec les métriques
        final_preds: Dict avec les prédictions du meilleur modèle
    """
    since = time.time()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f'best_model_config_{config}.pth')
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'train_iou_loss': [], 'val_iou_loss': []
    }

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_wts = None
    best_preds = None
    best_labels = None

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs} [Config {config}]')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_iou_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            pbar = tqdm(dataloaders[phase], desc=f'{phase.upper()}')
            
            for inputs_non_seg, inputs_seg, mask, labels in pbar:
                inputs_non_seg = inputs_non_seg.to(device)
                inputs_seg = inputs_seg.to(device)
                mask = mask.to(device)
                labels = labels.to(device).long()

                if config == 'A':
                    input_S, input_L, mask_input = inputs_non_seg, inputs_non_seg, None
                elif config == 'B':
                    input_S, input_L, mask_input = inputs_seg, inputs_seg, mask
                elif config == 'C1':
                    input_S, input_L, mask_input = inputs_non_seg, inputs_seg, mask
                elif config == 'C2':
                    input_S, input_L, mask_input = inputs_seg, inputs_non_seg, mask
                else:
                    raise ValueError(f"Config {config} inconnue")

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    try:
                        if use_iou_loss and mask_input is not None:
                            outputs, iou_loss = model(input_S, input_L, mask=mask_input, return_iou_loss=True)
                        elif mask_input is not None:
                            outputs = model(input_S, input_L, mask=mask_input)
                            iou_loss = torch.tensor(0.0, device=device)
                        else:
                            outputs = model(input_S, input_L)
                            iou_loss = torch.tensor(0.0, device=device)
                    except TypeError:
                        # Le modèle ne prend pas l'argument mask
                        outputs = model(input_S, input_L)
                        iou_loss = torch.tensor(0.0, device=device)

                    _, preds = torch.max(outputs, 1)
                    ce_loss = criterion(outputs, labels)
                    
                    if use_iou_loss and mask_input is not None:
                        loss = ce_loss + lambda_iou * iou_loss
                    else:
                        loss = ce_loss

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                batch_size = inputs_non_seg.size(0)
                running_loss += loss.item() * batch_size
                running_iou_loss += iou_loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                postfix = {'loss': loss.item()}
                if use_iou_loss and mask_input is not None:
                    postfix['iou_loss'] = iou_loss.item()
                pbar.set_postfix(postfix)

            if phase == 'train':
                scheduler.step()

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_iou_loss = running_iou_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            msg = f'{phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}'
            if use_iou_loss:
                msg += f' | IoU Loss: {epoch_iou_loss:.4f}'
            print(msg)
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_f1'].append(epoch_f1)
            history[f'{phase}_iou_loss'].append(epoch_iou_loss)

            if phase == 'val' and epoch_f1 > best_val_f1:
                best_val_f1 = epoch_f1
                best_val_acc = epoch_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())
                best_preds = all_preds.copy()
                best_labels = all_labels.copy()
                torch.save(model.state_dict(), best_model_path)
                print(f'Meilleur modèle sauvegardé (Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f})')
                print(f'Chemin: {best_model_path}')

    time_elapsed = time.time() - since
    print(f'\n{"-"*60}')
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Accuracy: {best_val_acc:.4f}')
    print(f'Best val F1-Score: {best_val_f1:.4f}')
    print(f'{"-"*60}')

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    final_preds = {
        'preds': best_preds,
        'labels': best_labels
    }

    return model, history, final_preds