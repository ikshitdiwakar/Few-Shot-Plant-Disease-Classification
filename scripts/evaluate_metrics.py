"""
=============================================================================
Project: Few-Shot Plant Disease Classification & Localization
Backbone: DeiT-Small (Patch-Token Extractor)
Methodology: Prototypical Networks + Cosine Distance
Key Empirical Results:
    - Step 4 Classification: 82.6% 5-Shot Accuracy (600 Validation Episodes)
    - Step 5 Localization:   20.0% Pointing-Game Accuracy | 11.9% Mean IoU (Top-20%)
=============================================================================
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def print_project_summary():
    """Prints the core empirical findings extracted from the master research reports."""
    summary = {
        "Project": "Few-Shot Plant Disease Classification & Localization",
        "Backbone": "DeiT-Small Transformer",
        "Classification Performance (Step 4)": "82.6% 5-Shot Accuracy (Cosine Distance)",
        "Localization Performance (Step 5)": "20.0% Pointing-Game Accuracy | 11.9% Mean IoU"
    }
    print("==================================================")
    for k, v in summary.items():
        print(f"[*] {k}: {v}")
    print("==================================================")

def evaluate_localization_batch(episodes_dict, manifest_df, model, device, n_episodes=50):
    """
    Evaluates batched Grad-CAM localization heatmaps against ground-truth PlantSeg masks.
    Computes Pointing-Game Accuracy and Mean IoU at top-20% threshold.
    """
    print_project_summary()
    
    hits = []
    ious = []
    sample_eps = episodes_dict["1shot"][:n_episodes]
    
    for idx, ep in enumerate(sample_eps):
        classes = list(ep["support"].keys())
        c = classes[0]
        sup_paths = [manifest_df.loc[n, "image_path"] for n in ep["support"][c]]
        q_name = ep["query"][c][0]
        q_path = manifest_df.loc[q_name, "image_path"]
        q_mask_path = manifest_df.loc[q_name, "mask_path"]
        
        try:
            hit, iou, _ = compute_gradcam_clean(q_path, q_mask_path, sup_paths, model, device)
            hits.append(hit)
            ious.append(iou)
        except Exception as e:
            continue
            
    mean_hit = sum(hits) / len(hits) if hits else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    
    print(f"\n[Evaluation Results across {len(hits)} novel episodes]")
    print(f" -> Pointing-Game Accuracy: {mean_hit * 100:.1f}%")
    print(f" -> Mean IoU (Top-20% threshold): {mean_iou * 100:.1f}%")
    
    return mean_hit, mean_iou
