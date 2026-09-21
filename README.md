# MRNet Knee MRI Classification + Explainable AI (SHAP & Grad-CAM)

> MSc AI for Medicine · University College Dublin
> Deep Learning for Musculoskeletal Pathology Detection with Full Explainability

---

## Overview

This project builds a multi-task deep learning pipeline to classify knee MRI scans from the [MRNet dataset (Stanford ML Group)](https://stanfordmlgroup.github.io/competitions/mrnet/) for three orthopedic conditions:

| Task | Condition | Clinical Significance |
|------|-----------|----------------------|
| ACL | Anterior Cruciate Ligament tear | Most common sports injury |
| Meniscus | Meniscal tear | Leading cause of knee instability |
| Abnormal | General abnormality | First-pass triage |

Two architectures are benchmarked:
- **ResNet-18** (pretrained on ImageNet, fine-tuned per-plane)
- **Swin Transformer** (SwinViT, hierarchical attention)

Full **Explainable AI (XAI)** is applied post-hoc using:
- **SHAP** (SHapley Additive exPlanations) — global feature importance
- **Grad-CAM** (Gradient-weighted Class Activation Mapping) — spatial saliency maps

---

## Repository Structure

```
mrnet-knee-mri-xai/
├── notebooks/
│   ├── 00_mrnet_tutorial_baseline.ipynb    # Baseline tutorial
│   ├── 01_mrnet_resnet_training.ipynb      # ResNet-18 training
│   ├── 02_mrnet_swin_training.ipynb        # Swin Transformer training
│   ├── 03_xai_gradcam.ipynb                # Grad-CAM saliency maps
│   ├── 04_xai_shap.ipynb                   # SHAP feature attribution
│   ├── 05_resnet_attention.ipynb           # Attention-augmented ResNet
│   └── 06_gradcam_pretrained_model.ipynb   # Grad-CAM with trained model
├── reports/
│   └── MRNet_report_group1.pdf             # Full academic report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Results

| Model | ACL AUC | Meniscus AUC | Abnormal AUC |
|-------|---------|-------------|-------------|
| ResNet-18 (split-plane) | 0.87 | 0.78 | 0.89 |
| Swin Transformer | 0.89 | 0.80 | 0.91 |

---

## Methodology

### 1. Data Pipeline
- 3-plane MRI volumes (axial, coronal, sagittal) loaded per slice
- Per-plane normalisation and augmentation (horizontal flip, rotation ±10°)
- Train/Val/Test split following original MRNet protocol

### 2. Architecture
- **Split-plane strategy** — independent encoders per MRI plane, features concatenated into classification head
- **Swin Transformer** — window-based self-attention for global structural pattern capture

### 3. Explainability
- **Grad-CAM** — highlights anatomical regions driving ACL/meniscus predictions
- **SHAP** — quantifies each input slice's contribution to the final prediction probability

---

## Getting Started

**1. Clone and install**

```bash
git clone https://github.com/chinu85/mrnet-knee-mri-xai.git
cd mrnet-knee-mri-xai
pip install -r requirements.txt
```

**2. Download the dataset**

The MRNet dataset requires a free registration with Stanford ML Group.
Request access at: https://stanfordmlgroup.github.io/competitions/mrnet/

Once downloaded, place the data folder in the project root:

```
data/
├── train/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
└── valid/
    ├── axial/
    ├── coronal/
    └── sagittal/
```

**3. Run the notebooks**

Open Jupyter and run notebooks in order starting from `01_mrnet_resnet_training.ipynb`.

```bash
jupyter notebook
```

---

## Technologies

- **PyTorch** — model training and inference
- **torchvision** — pretrained ResNet-18 backbone
- **timm** — Swin Transformer implementation
- **SHAP** — model-agnostic explainability
- **grad-cam** — gradient-based saliency maps
- **scikit-learn** — AUC metrics and evaluation
- **matplotlib / seaborn** — result visualisations
