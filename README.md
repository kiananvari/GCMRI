# Anonymous Repository
### Self-Supervised Multi-Contrast MRI Reconstruction using Permutation-Driven Contrast Invariance

This repository contains the **anonymous implementation** associated with a paper currently under review at **MICCAI**.

---

## Overview

Multi-contrast MRI reconstruction aims to exploit **complementary anatomical information across different MRI contrasts** to improve image reconstruction from **undersampled k-space data**.

However, many existing learning-based approaches implicitly rely on:

- **Fixed contrast ordering**
- **Predefined contrast roles**

These design choices can limit robustness when **contrast availability** or **acquisition conditions** vary.

In this work, we propose a **contrast-invariant self-supervised reconstruction framework** that treats input contrasts as an **unordered set** rather than a fixed channel stack.

To enforce this property, we introduce **permutation-based training**, which removes contrast-order shortcuts and encourages the network to learn **content-driven interactions across contrasts**.

The proposed method operates in a **self-supervised setting**, enabling training directly from **undersampled k-space data without requiring fully sampled reference images**.

---

## Key Features

- ✔ **Self-supervised reconstruction** from undersampled multi-coil k-space data  
- ✔ **Permutation-driven contrast invariance** to eliminate contrast-order shortcuts  
- ✔ **Content-driven contrast fusion architecture** for multi-contrast interaction  
- ✔ **Spatial–frequency feature integration** for improved artifact suppression  
- ✔ **Prompt-based conditioning** supporting multiple acceleration factors and sampling trajectories  
- ✔ **Robust generalization** to unseen contrast configurations and acquisition settings  

---

## Datasets

Experiments in the paper were conducted using publicly available datasets:

### BraTS
Multi-contrast brain MRI dataset including:
- **T1-weighted**
- **T2-weighted**
- **FLAIR**
- **T1ce**

### M4Raw
Multi-contrast MRI dataset providing **raw k-space measurements**.

Data preprocessing and preparation scripts are included in this repository.

---

## Code Availability

This repository includes:

- Training scripts
- Evaluation scripts
- Model implementation
- Data preprocessing utilities
- Configuration files for experiments

---

## Anonymity Notice

To preserve **double-blind review**, author identities and affiliations have been removed from this repository.

The full implementation and documentation will be released publicly upon acceptance.
