Anonymous Repository for "Self-Supervised Multi-Contrast MRI Reconstruction using Permutation-Driven Contrast Invariance"

This repository contains the anonymous implementation associated with a paper currently under review at MICCAI.

Overview

Multi-contrast MRI reconstruction aims to exploit complementary information across different MRI contrasts to improve image reconstruction from undersampled k-space data. However, many existing approaches implicitly rely on fixed contrast ordering or predefined contrast roles, which can limit robustness when contrast availability or acquisition conditions vary.

In this work, we propose a contrast-invariant self-supervised reconstruction framework that treats input contrasts as an unordered set rather than a fixed channel stack. To enforce this property, we introduce permutation-based training, which removes contrast-order shortcuts and encourages the network to learn content-driven interactions across contrasts.

The proposed method operates in a self-supervised setting, enabling training directly from undersampled k-space data without requiring fully sampled reference images.

Key Features

Self-supervised reconstruction from undersampled multi-coil k-space data (no fully sampled references required)

Permutation-driven contrast invariance to eliminate contrast-order shortcuts

Content-driven fusion across contrasts using a contrast fusion architecture

Spatial–frequency feature integration for improved artifact suppression

Prompt-based conditioning to support multiple acceleration factors and sampling trajectories

Robust generalization to unseen contrast configurations and acquisition settings

Datasets

Experiments in the paper were conducted using publicly available datasets:

BraTS – multi-contrast brain MRI dataset (T1w, T2w, FLAIR, T1ce)

M4Raw – multi-contrast MRI dataset with raw k-space measurements

Data preprocessing and preparation scripts are provided in this repository.

Code Availability

The repository includes:

Training and evaluation scripts

Data preprocessing utilities

Model implementation

Configuration files for experiments

To preserve anonymity during the review process, author identities and affiliations have been removed. Code will be fully released upon acceptance.
