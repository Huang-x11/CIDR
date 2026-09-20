# CIDR

**Causality-Inspired Disentangled Representation for Robust Whole Slide Image Classification**

## Overview

CIDR is a representation-learning module inserted between a frozen pathology foundation model (PFM) and an existing multiple instance learning (MIL) model. It separates patch features into a label-causative factor **S** and a residual factor **Z**. Guided by a Causal Information Bottleneck objective, feature reconstruction preserves information, slide-level supervision makes S discriminative, and cross-slide HSIC regularization reduces dependence between the two factors. Only S is passed to the MIL classifier.

The paper evaluates CIDR on **Camelyon16**, combining **CTransPath, CONCH, and UNI** with seven MIL methods. Across these 21 combinations, the reported average gains over the corresponding baselines are **2.43, 1.02, 3.62, and 5.13 percentage points** in accuracy, AUC, F1, and MCC, respectively.

## Installation

Use **Python 3.10** and a compatible NVIDIA GPU. The following commands use the CUDA 11.6 build from the local development environment:

```bash
conda create -n cidr python=3.10 -y
conda activate cidr
python -m pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
```

Dependency versions are based on the local environment; a clean installation and full reproduction have not yet been verified.

## Data Preparation

Prepare one CPU float32 `.pt` feature tensor per slide, with shape `[number_of_patches, feature_dimension]`. Feature extraction is performed separately.

```text
data/
├── data_c16_ctranspath.csv
├── c16_ctranspath/pt_files/<slide_id>.pt  # 768 dimensions
├── data_c16_conch.csv
├── c16_conch/pt_files/<slide_id>.pt       # 512 dimensions
├── data_c16_uni.csv
└── c16_uni/pt_files/<slide_id>.pt         # 1024 dimensions
```

Each CSV contains `x` (feature filename), `y` (0: normal, 1: tumor), and `data_type` (`train`, `valid`, or `test`). The loader uses the basename of `x` and resolves it through the directory layout above. The supplied splits contain **216 training, 54 validation, and 129 test slides**.

## Training

The model is implemented in `core/model.py`. Before running, replace the remaining `from core.model_v2 import CDG` line in `train.py` with:

```python
from core.model import CDG
```

Run from the repository root:

```bash
python train.py
```

By default, the script sequentially runs all **21 combinations** of the three feature sets and seven backbones: `MaxMIL`, `ABMIL`, `DSMIL`, `TransMIL`, `CLAM-SB`, `CLAM-MB`, and `DTFD-AFS`. `MaxMIL` corresponds to MaxPooling in the paper, and `DTFD-AFS` is the selected DTFD variant. Edit `TASKS` and `BACKBONES` in `train.py` to select fewer experiments; there are no command-line configuration arguments.

The defaults use 4 S concepts, 12 Z concepts, 16 dimensions per concept, and an HSIC weight of `1e-3`. Training uses AdamW with learning rate `3e-4`, weight decay `1e-4`, batch size 1, gradient accumulation over 2 steps, and at most 200 epochs. Keep batch size 1 while HSIC is enabled.

## Evaluation

After training, the best checkpoint by validation accuracy is used to predict the test split. Each run saves its checkpoints, logs, and `results.json` under:

```text
logs1/<TASK>/<BACKBONE>/version_<N>/
```

In `Metrics.ipynb`, run the imports and `get_metrics` definition, then evaluate the desired result file, for example:

```python
get_metrics('./logs1/C16_CTRANSPATH/DSMIL/version_0/results.json')
```

Replace the task, backbone, and version as needed. The metric cells report **accuracy, AUC, F1, and MCC**; the later split-export cells are not required for evaluation.
