import torch
import numpy as np
from tqdm import tqdm

def evaluate_iou(model, dataloader, config, device, threshold=0.8):
    model.eval()
    iou_scores = []
    
    with torch.no_grad():
        for inputs_non_seg, inputs_seg, mask, labels in tqdm(dataloader):
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
            
            if mask_input is None:
                continue
            
            input_S = input_S.to(device)
            input_L = input_L.to(device)
            mask_input = mask_input.to(device)
            
            # Forward pour remplir l'attention
            _ = model(input_S, input_L, mask=mask_input)
            
            # Calculer IoU
            heatmap = model.get_heatmap(branch_idx=0)
            if heatmap is not None:
                iou = model.compute_iou(heatmap, mask_input, threshold)
                iou_scores.append(iou.item())
    
    # Statistiques
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    
    print(f"\nConfig {config} - IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    return mean_iou, std_iou, iou_scores
