
# Multi-Backbone LeJEPA Pre-Training & Linear Probe Evaluation Pipeline

A complete PyTorch framework for self-supervised representation learning using **LeJEPA (Joint-Embedding Predictive Architecture with SIGReg regularization)** and downstream **stabilized linear probing** across multiple CNN backbones.

This repository supports end-to-end pre-training and downstream benchmark evaluation for:
* **Custom ConvNet (CNN)**
* **ResNet-18**
* **EfficientNet-B0**
* **MobileNet-V2**

---

## 📌 Architecture & Methodology

The pipeline operates in two main sequential phases:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1: LEJEPA PRE-TRAINING                         │
│                                                                        │
│   ┌──────────────┐    Quadrant Crop    ┌──────────────┐                │
│   │ Input Image  │ ──────────────────> │ 2 Random     │                │
│   │  (Full Size) │                     │ Image Tiles  │                │
│   └──────────────┘                     └──────┬───────┘                │
│                                               │                        │
│                                               ▼                        │
│                                      ┌─────────────────┐               │
│                                      │ Shared Backbone │               │
│                                      └────────┬────────┘               │
│                                               │                        │
│                                               ▼                        │
│                                      ┌─────────────────┐               │
│                                      │    MLP Head     │               │
│                                      └────────┬────────┘               │
│                                               │                        │
│                                               ▼                        │
│                       Loss: Invariance + SIGReg Regularization          │
└───────────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼ (Saved Checkpoints)
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 2: DOWNSTREAM LINEAR PROBE BENCHMARK               │
│                                                                        │
│  • Backbones: Custom CNN, ResNet-18, EfficientNet-B0, MobileNet-V2     │
│  • Weights: LeJEPA vs. ImageNet-1k vs. Pure Random                     │
│  • Regimes: 1-Shot, 10-Shot, and Full-Dataset (Across 5 Seeds)         │
│  • Preprocessing: Class Zoom (Alfalfa, Haylage, TMR) + Quadrant Tiling │
└────────────────────────────────────────────────────────────────────────┘

```

### Phase 1: Self-Supervised Pre-Training (LeJEPA)

* **Tile-Based Multi-View Sampling**: Input images are cropped into 4 equal spatial quadrants (tiles). Two distinct tiles are sampled per image per step.
* **Optimization Objective**: Minimizes pairwise tile embedding variance while applying SIGReg statistical regularization to prevent representation collapse:

$$\mathcal{L}_{\text{total}} = \lambda \cdot \mathcal{L}_{\text{sigreg}} + (1 - \lambda) \cdot \mathcal{L}_{\text{inv}}$$


* **Training Techniques**: Warmup + Cosine Annealing learning rate scheduling, mixed precision (`torch.amp`), and logging via Weights & Biases (W&B).

### Phase 2: Downstream Linear Probe Evaluation

* **Static Feature Caching**: Pre-extracts and caches embeddings from frozen backbones to optimize compute efficiency (DirectML / AMD GPU & CPU compatible).
* **Initialization Benchmarking**: Directly compares LeJEPA pre-trained weights against supervised ImageNet-1k weights and Random baseline initializations.
* **Few-Shot Stabilization**: Evaluates 1-shot, 10-shot, and full-dataset regimes across 5 random seeds (`0, 1, 2, 3, 4`), calculating mean accuracy and standard deviation.

---

## ⚙️ Configuration Parameters

### 1. Pre-Training Parameters (`pretrain_*.py`)

| Parameter | Default | Description |
| --- | --- | --- |
| `IMAGE_SIZE` | `128` | Pre-training tile spatial resolution |
| `BATCH_SIZE` | `8` | Batch size per GPU step |
| `EPOCHS` | `120` | Total pre-training epochs |
| `PROJ_DIM` | `128` | MLP projector output dimension |
| `LR` | `0.001788` | Initial AdamW learning rate |
| `WEIGHT_DECAY` | `0.005292` | Optimizer weight decay |
| `LAMBDA` | `0.5` | Weighting ratio between SIGReg and Invariance losses |

### 2. Evaluation Parameters (`Custom_and_Backbone_Evaluation.ipynb`)

| Parameter | Default | Description |
| --- | --- | --- |
| `IMAGE_SIZE` | `224` | Downstream image input dimension |
| `BATCH_SIZE` | `32` | Feature extraction batch size |
| `HEAD_BATCH_SIZE` | `128` | Linear classification head training batch size |
| `EPOCHS` | `60` / `100` | Linear head training epochs |
| `MULTIPLE_SEEDS` | `[0, 1, 2, 3, 4]` | Random seeds for few-shot stabilization |
| `ZOOM_CLASSES` | `["alfalfa", "haylage", "tmr"]` | Target classes for center-crop zoom |
| `CHOSEN_ZOOM` | `0.15` | Center-crop ratio applied to target zoom classes |

---

## 🛠 Repository Structure

```text
.
├── pretrain/
│   ├── pretrain_custom_cnn.py         # LeJEPA pre-training for Custom ConvNet
│   ├── pretrain_resnet18.py           # LeJEPA pre-training for ResNet-18
│   ├── pretrain_efficientnetb0.py    # LeJEPA pre-training for EfficientNet-B0
│   └── pretrain_mobilenetv2.py       # LeJEPA pre-training for MobileNet-V2
├── notebooks/
│   └── Custom_and_Backbone_Evaluation.ipynb # Complete evaluation and probing notebook
├── requirements.txt                   # Environment dependencies
└── README.md                          # Documentation

```

---

## 📦 Requirements & Installation

Install all required Python packages:

```bash
pip install torch torchvision torch-directml numpy pandas pillow matplotlib scikit-learn openpyxl tqdm wandb

```

---

## 🚀 Execution Guide

### Step 1: Run Pre-Training Scripts

Execute the pre-training script corresponding to the backbone you want to pre-train:

```bash
# Pre-train Custom CNN
python pretrain/pretrain_custom_cnn.py

# Pre-train ResNet-18
python pretrain/pretrain_resnet18.py

# Pre-train EfficientNet-B0
python pretrain/pretrain_efficientnetb0.py

# Pre-train MobileNet-V2
python pretrain/pretrain_mobilenetv2.py

```

*Checkpoints (`.pth`) will automatically be saved in your specified `CHECKPOINT_DIR` folder*.

### Step 2: Run Linear Probe Evaluation

1. Update directory paths in `Custom_and_Backbone_Evaluation.ipynb`:
* `CLASSIFICATION_DATASET_ROOT`
* Model checkpoint paths (`LEJEPA_RESNET`, `LEJEPA_EFFICIENTNET`, etc.)
* `OUTPUT_ROOT`


2. Open and run all cells in `Custom_and_Backbone_Evaluation.ipynb`.

---

## 📊 Output Artifacts

Running the pipeline generates:

1. **Checkpoints**: Saved model weights (e.g., `lejepa_convnet_encoder_tile_loss.pth`).
2. **Tabular Results**: `classification_overall_summary.csv` and `classification_results_workbook.xlsx` detailing accuracy ($\text{mean} \pm \text{std}$) across 1-shot, 10-shot, and full regimes.
3. **Visual Plots**:
* `publication_grid_original.png` & `publication_grid_processed.png`: Preprocessing sample grids.
* `confusion_matrix_full.png`: Normalized multi-class confusion matrices.
* `embedding_pca_2d_full.png` & `embedding_pca_3d_full.png`: 2D and 3D PCA cluster scatter plots.



---


```

```
