import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def evaluate_iou(model, dataloader, config, device, threshold=0.8):
    """Évalue l'IoU entre attention rollout et masque de segmentation"""
    
    if config == 'A':
        print("Config A ne supporte pas l'IoU car pas de masque")
        return 0.0, 0.0, []
    
    # Déterminer quelle branche utiliser selon config
    if config == 'B' or config == 'C1':
        branch_idx = 1
        branch_name = 'L (Large)'
    elif config == 'C2':
        branch_idx = 0
        branch_name = 'S (Small)'
    else:
        raise ValueError(f"Config {config} inconnue")
    
    print(f"Config {config}: Branche {branch_name} (index {branch_idx}), threshold={threshold}")
    
    model.eval()
    iou_scores = []
    
    with torch.no_grad():
        for inputs_non_seg, inputs_seg, mask, labels in tqdm(dataloader, desc=f"IoU {config}"):
            if config == 'B':
                input_S, input_L, mask_input = inputs_seg, inputs_seg, mask
            elif config == 'C1':
                input_S, input_L, mask_input = inputs_non_seg, inputs_seg, mask
            elif config == 'C2':
                input_S, input_L, mask_input = inputs_seg, inputs_non_seg, mask
            else:
                continue
            
            input_S = input_S.to(device)
            input_L = input_L.to(device)
            mask_input = mask_input.to(device)
            
            _ = model(input_S, input_L, mask=mask_input)
            
            heatmap = model.get_heatmap(branch_idx=branch_idx)
            if heatmap is None:
                continue
            
            # Calculer IoU pour chaque image du batch
            batch_size = heatmap.shape[0]
            for i in range(batch_size):
                heatmap_i = heatmap[i]
                mask_i = mask_input[i, 0] if mask_input.dim() == 4 else mask_input[i]
                
                iou = compute_one_iou(heatmap_i, mask_i, threshold)
                iou_scores.append(iou)
    
    if len(iou_scores) == 0:
        print(f"Aucun score IoU calculé pour config {config}")
        return 0.0, 0.0, []
    
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    
    print(f"Config {config} - Branche {branch_name}: IoU = {mean_iou:.4f} ± {std_iou:.4f} (n={len(iou_scores)})")
    
    return mean_iou, std_iou, iou_scores


def compute_one_iou(heatmap, mask, threshold=0.8):
    """Calcule l'IoU en binarisant la heatmap par quantile"""
    # Redimensionner
    if heatmap.shape != mask.shape:
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=mask.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    # Binarisation par quantile
    threshold_value = torch.quantile(heatmap.flatten(), threshold)
    heatmap_bin = (heatmap >= threshold_value).float()
    mask_bin = (mask > 0.5).float()
    
    # IoU
    intersection = (heatmap_bin * mask_bin).sum()
    union = heatmap_bin.sum() + mask_bin.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()