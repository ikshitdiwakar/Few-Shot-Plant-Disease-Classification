# 🌿 Few-Shot Plant Disease Classification & Localization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision research repository implementing **Few-Shot Prototypical Networks** with **DeiT-Small** transformer backbones and **Grad-CAM spatial localization** on the PlantSeg dataset.

---

## 📊 Key Empirical Results

| Experimental Stage | Protocol / Configuration | Quantitative Performance |
| :--- | :--- | :--- |
| **Step 4: Classification** | DeiT-Small (Patch Mean) + Cosine Distance (600 Episodes) | **82.6%** (5-Shot Accuracy) |
| **Step 5: Localization** | Patch-Token Grad-CAM vs. PlantSeg Pixel-Level Masks | **20.0%** Pointing-Game Accuracy / **11.9%** Mean IoU |

---

## 📂 Repository Structure

* **`Master Research Report - Few-Shot Plant Disease...pdf`** — Comprehensive master synthesis report covering Steps 1 through 6.
* **`Step 3 / 4 / 5 Result PDFs`** — Granular technical reports detailing data audits, classification metrics, and spatial heatmaps.
* **`gradcam_evaluation.py`** — Core script for extracting transformer patch tokens and calculating Grad-CAM overlap metrics.
* **`episodes_novel.json`** — Standardized evaluation episode configurations for novel-class validation.
* **`plantseg_manifest_split.csv`** — Cleaned dataset split manifest linking images to pixel-level disease masks.

---

## 🚀 Reproduction Instructions

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/ikshitdiwakar/Few-Shot-Plant-Disease-Classification.git](https://github.com/ikshitdiwakar/Few-Shot-Plant-Disease-Classification.git)
   cd Few-Shot-Plant-Disease-Classification
