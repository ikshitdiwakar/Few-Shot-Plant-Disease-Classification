"""
Grad-CAM Evaluation & Patch Token Localization Script
Project: Few-Shot Plant Disease Classification & Localization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

def compute_gradcam_clean(image_path, mask_path, support_imgs, model, device):
    """Computes Grad-CAM heatmap for a query image against support prototype using patch tokens."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    
    mask = Image.open(mask_path)
    mask_resized = mask.resize((224, 224), Image.NEAREST)
    mask_np = (np.array(mask_resized) > 0).astype(np.float32)

    _, q_feat = model(input_tensor)
    
    with torch.no_grad():
        sup_feats = []
        for s_path in support_imgs:
            s_img = Image.open(s_path).convert("RGB")
            s_tensor = transform(s_img).unsqueeze(0).to(device)
            _, s_f = model(s_tensor)
            sup_feats.append(s_f)
        prototype = torch.stack(sup_feats).mean(dim=0)

    q_norm = F.normalize(q_feat, p=2, dim=1)
    p_norm = F.normalize(prototype, p=2, dim=1)
    score = (q_norm * p_norm).sum()

    model.zero_grad()
    score.backward()

    if model.activations is None or model.gradients is None:
        return 0.0, 0.0, np.zeros((224, 224))

    grads = model.gradients.detach()[:, 1:, :]
    acts = model.activations.detach()[:, 1:, :]

    weights = torch.norm(grads, dim=2, keepdim=True)
    cam = (weights * acts).sum(dim=2)
    cam = F.relu(cam)

    cam = cam.view(1, 14, 14)
    cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    max_y, max_x = np.unravel_index(cam.argmax(), cam.shape)
    pointing_hit = float(mask_np[max_y, max_x] > 0)

    thresh = np.percentile(cam, 80)
    cam_bin = (cam >= thresh).astype(np.float32)
    
    intersection = (cam_bin * mask_np).sum()
    union = ((cam_bin + mask_np) > 0).sum()
    iou = intersection / union if union > 0 else 0.0

    return pointing_hit, iou, cam
