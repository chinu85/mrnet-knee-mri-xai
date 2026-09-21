# MRNet Knee MRI Classification + Explainable AI (SHAP & Grad-CAM)

> MSc AI for Medicine · University College Dublin
> Deep Learning for Musculoskeletal Pathology Detection with Full Explainability

---

## Overview

This project benchmarks deep learning architectures for automated knee MRI classification using the [Stanford MRNet dataset](https://stanfordmlgroup.github.io/competitions/mrnet/) — 1,370 knee MRI exams across three anatomical planes (axial, coronal, sagittal), labelled for three clinical targets: ACL tear, meniscal tear, and general abnormality.

Three architectures are compared:
- **ResNet-18 (max-pool fusion)** — ImageNet-pretrained, max-pooled slice aggregation
- **ResNet-18 (attention pooling)** — learned per-slice attention weights, 2-stage training
- **Swin Transformer (Swin-Tiny)** — window-based self-attention, attention pooling

All models use a **multi-plane fusion** strategy: three separate models (one per plane) whose output probabilities are combined via Nelder-Mead optimised weighted sum.

Full **Explainable AI (XAI)** is applied using **Grad-CAM** to generate spatial saliency maps and qualitatively validate anatomical localisation across true positives, false positives, false negatives, and true negatives.

---

## Repository Structure

```
mrnet-knee-mri-xai/
├── notebooks/
│   ├── 00_mrnet_tutorial_baseline.ipynb    # Baseline tutorial
│   ├── 01_mrnet_resnet_training.ipynb      # ResNet-18 max-pool training
│   ├── 02_mrnet_swin_training.ipynb        # Swin Transformer training
│   ├── 03_xai_gradcam.ipynb                # Grad-CAM saliency maps
│   ├── 04_xai_shap.ipynb                   # SHAP feature attribution
│   ├── 05_resnet_attention.ipynb           # Attention-pooling ResNet
│   └── 06_gradcam_pretrained_model.ipynb   # Grad-CAM with trained model
├── reports/
│   └── MRNet_report_group1.pdf             # Full academic group report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Results (Holdout Test Set, n=119)

**Aggregate Performance**

| Model | ROC-AUC | Sensitivity | Specificity | F1 |
|-------|---------|------------|------------|-----|
| ResNet-18 (max-pool fusion) | **0.898** | **90.5%** | 75.0% | **0.863** |
| ResNet-18 (attention fusion) | 0.859 | 81.6% | **76.3%** | 0.816 |
| Swin Transformer fusion | 0.887 | 88.1% | 73.1% | 0.843 |

**Per-Task ROC-AUC**

| Task | ResNet-18 (max-pool) | ResNet-18 (attention) | Swin Transformer |
|------|---------------------|----------------------|-----------------|
| Abnormality | 0.873 | 0.872 | **0.928** |
| ACL Tear | **0.968** | 0.919 | 0.927 |
| Meniscal Tear | 0.798 | 0.765 | **0.804** |

---

## Methodology

### 1. Dataset & Preprocessing
- 1,370 knee MRI exams — axial, coronal, sagittal series per exam
- Class distribution: 80.6% abnormal, 23.3% ACL tear, 37.1% meniscal tear
- 85:15 train/validation split from 1,251 training exams; 119-exam holdout test set
- Augmentation: random rotation, horizontal flip, affine translation
- Intensity normalisation to [0, 1]

### 2. Architecture
- **Slice aggregation**: Each 3D volume → per-slice features via backbone → aggregated with either max-pool or learned attention
- **Multi-plane fusion**: Separate model per plane; final probabilities fused via Nelder-Mead optimised weights
- **Weighted BCE loss**: pos_weight = negative/positive ratio to handle class imbalance
- **Auxiliary experiments**: WGAN-GP synthetic augmentation; DINOv3 self-supervised baseline

### 3. Explainability (Grad-CAM)
- Applied to the sagittal ResNet-18 ACL model
- **True Positives**: Tight activation localised near ACL anatomical region
- **False Positives**: Diffuse peripheral activation on non-pathological features
- **False Negatives**: Weak activation — model missed subtle single-plane tears
- Finding: Grad-CAM is most valuable as an **error analysis tool**, not as proof of radiologist-level reasoning

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
