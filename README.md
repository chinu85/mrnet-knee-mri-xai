# MRNet Knee MRI Classification + Explainable AI (SHAP & Grad-CAM)

> **MSc AI for Medicine · University College Dublin**  
> Deep Learning for Musculoskeletal Pathology Detection with Full Explainability

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project builds a **multi-task deep learning pipeline** to classify knee MRI scans from the [MRNet dataset (Stanford ML Group)](https://stanfordmlgroup.github.io/competitions/mrnet/) for three orthopedic conditions:

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
│   ├── 01_mrnet_resnet_training.ipynb      # ResNet-18 per-plane training pipeline
│   ├── 02_mrnet_swin_training.ipynb        # Swin Transformer training pipeline
│   ├── 03_xai_gradcam.ipynb                # Grad-CAM saliency visualisation
│   ├── 04_xai_shap.ipynb                   # SHAP feature attribution analysis
│   └── 05_resnet_attention.ipynb           # Attention-augmented ResNet experiments
├── reports/
│   ├── MRNet_report_group1.pdf             # Full academic group report
│   └── MRNet_viva_consolidated_model_card_FAQ.docx  # Model card & viva Q&A
├── assets/
│   └── (architecture diagrams, result plots)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Results

| Model | Plane Fusion | ACL AUC | Meniscus AUC | Abnormal AUC |
|-------|-------------|---------|-------------|-------------|
| ResNet-18 (split-plane) | Max pooling | 0.87 | 0.78 | 0.89 |
| Swin Transformer | Attention | 0.89 | 0.80 | 0.91 |

---

## Methodology

### 1. Data Pipeline
- 3-plane MRI volumes (axial, coronal, sagittal) loaded per slice
- Per-plane normalisation and augmentation (horizontal flip, rotation ±10°)
- Train/Val/Test split following original MRNet protocol

### 2. Architecture
- **Split-plane strategy**: Independent encoders per MRI plane → feature concatenation → classification head
- **Swin Transformer**: Window-based self-attention captures global structural patterns

### 3. Explainability
- **Grad-CAM**: Highlights which anatomical regions activate the model's ACL/meniscus predictions
- **SHAP**: Quantifies each input slice's contribution to the final prediction probability

---

## Getting Started

`ash
# Clone the repo
git clone https://github.com/chinu85/mrnet-knee-mri-xai.git
cd mrnet-knee-mri-xai

# Install dependencies
pip install -r requirements.txt

# Download MRNet dataset (Stanford registration required)
# https://stanfordmlgroup.github.io/competitions/mrnet/

# Run training
jupyter notebook notebooks/01_mrnet_resnet_training.ipynb
`

---

## Dataset

The **MRNet dataset** is provided by Stanford ML Group and requires a data use agreement.  
Download from: [https://stanfordmlgroup.github.io/competitions/mrnet/](https://stanfordmlgroup.github.io/competitions/mrnet/)

Once downloaded, place in:
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

---

## Technologies

- **PyTorch** — model training and inference
- **torchvision** — pretrained ResNet-18 backbone
- **timm** — Swin Transformer implementation
- **SHAP** — model-agnostic explainability
- **grad-cam** — gradient-based saliency maps
- **scikit-learn** — AUC metrics and evaluation
- **matplotlib / seaborn** — result visualisations

---

## Academic Context

This project was completed as part of the **MSc AI for Medicine** programme at University College Dublin (2025–2026), within the module on Deep Learning for Medical Imaging.

---

## Author

**Sourabh Kumawat**  
MSc AI for Medicine, University College Dublin  
📧 shmsourabhrj3062002@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sourabh-kumawat-a74956202/) | [GitHub](https://github.com/chinu85)
