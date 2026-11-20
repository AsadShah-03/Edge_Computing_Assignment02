# Edge Computing & Intelligent Systems — Assignment 04

**Course:** Edge Computing & Intelligent Systems
**Deadline:** 20-Nov-2026
**Total Marks:** 100

---

## Overview

This repository contains a complete implementation of two major tasks:

| Task | Topic | Marks |
|------|-------|-------|
| Q1 | End-to-End Knowledge Distillation Framework | 50 |
| Q2 | Automated Neural Architecture Search for Edge Devices | 50 |

Both tasks are implemented in a single Google Colab-ready Jupyter notebook.

---

## Repository Structure

```
Edge_Computing_Assignment04/
├── Assignment04_KD_NAS.ipynb    # Main notebook (Q1 + Q2)
├── Assignment 4.pdf             # Original assignment specification
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

> Checkpoints, ONNX models, and CSV results are generated at runtime inside
> `checkpoints/`, `onnx_models/`, and `results/` directories (excluded from git).

---

## Question 1: Knowledge Distillation Framework

### Architecture

```
Teacher: ResNet-50 (CIFAR-100, 100 classes)
    |
    |-- Soft targets (temperature-scaled KL divergence)
    |-- Feature maps (intermediate layer L2 alignment)
    |-- Hard labels (cross-entropy)
    |
    v
Students:
  - MobileNetV2
  - SqueezeNet 1.1
  - EfficientNet-Lite0 (custom implementation)
```

### Key Components

| Component | Description |
|-----------|-------------|
| `TeacherResNet50` | ResNet-50 adapted for 32x32 input with forward hook for feature extraction |
| `StudentMobileNetV2` | MobileNetV2 adapted for CIFAR-100 with feature hook |
| `StudentSqueezeNet` | SqueezeNet 1.1 adapted for CIFAR-100 with feature hook |
| `StudentEfficientNetLite0` | Custom MBConv-based EfficientNet-Lite0 from scratch |
| `KnowledgeDistillationLoss` | Combined soft-target KL + hard CE + feature L2 loss |
| `FeatureAdapter` | 1x1 conv to align student/teacher feature channel dimensions |

### Distillation Configuration

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Temperature (T) | 4.0 | Softens teacher probability distribution |
| Alpha | 0.3 | Weight for hard-label cross-entropy |
| Beta | 0.1 | Weight for feature distillation loss |
| Teacher epochs | 60 | ResNet-50 training epochs |
| Student KD epochs | 50 | Distillation training epochs |

### Loss Formula

```
L_total = alpha * L_hard + (1 - alpha) * T^2 * KL(student || teacher) + beta * L_feature
```

---

## Question 2: Neural Architecture Search

### Search Space

| Dimension | Options |
|-----------|---------|
| Convolutional depth | 2, 3, 4, 5 blocks |
| Channels per block | 16, 32, 48, 64, 96 |
| Kernel size | 3, 5 |
| Block type | Standard conv / Depthwise separable |
| Activation function | ReLU, ReLU6, Hardswish, SiLU |
| Input resolution | 24x24, 32x32 |

### NAS Strategy

**Bayesian Optimization via Optuna (TPE Sampler)**

Each trial:
1. Sample architecture configuration from the search space
2. Build `NASNet` dynamically from the config
3. Train for 5 proxy epochs on 10,000 CIFAR-10 samples
4. Measure CPU inference latency and ONNX model size
5. Reject if constraints violated; compute composite score otherwise
6. Report score back to Optuna for next trial suggestion

### Hardware Constraints (Edge Deployment)

| Constraint | Limit |
|-----------|-------|
| CPU inference latency | <= 50 ms |
| ONNX model storage size | <= 5 MB |

### Scoring Objective

```
score = accuracy - 5.0 * (latency / max_latency) - 3.0 * (size / max_size)
```

Infeasible architectures receive a penalty of -1000 and are excluded from selection.

---

## Setup and Execution

### Requirements

```
torch >= 2.0
torchvision >= 0.15
onnx >= 1.14
onnxruntime >= 1.16
optuna >= 3.0
timm >= 0.9
psutil >= 5.9
pandas >= 2.0
numpy >= 1.24
```

### Running on Google Colab

1. Upload `Assignment04_KD_NAS.ipynb` to Google Colab.
2. Set runtime to **GPU (T4 or better)**: `Runtime > Change runtime type > T4 GPU`.
3. Run all cells from top to bottom using `Runtime > Run all`.
4. All dependencies are installed automatically in the first cell.

### Running Locally

```bash
pip install -r requirements.txt
jupyter notebook Assignment04_KD_NAS.ipynb
```

---

## Output Files

All outputs are generated at runtime:

| Directory | Contents |
|-----------|----------|
| `checkpoints/` | `.pth` model weights for teacher, baseline students, KD students, NAS best |
| `onnx_models/` | `.onnx` exported models for CPU inference |
| `results/` | CSV and JSON benchmark and comparison files |

### Result Files

| File | Description |
|------|-------------|
| `results/q1_comparison.csv` | Baseline vs KD accuracy comparison for all 3 students |
| `results/q1_onnx_benchmark.csv` | ONNX size and CPU latency for distilled student models |
| `results/q2_nas_search_history.csv` | Per-trial NAS results (accuracy, latency, size, score) |
| `results/q2_best_config.json` | Best discovered architecture configuration |
| `results/q2_onnx_benchmark.json` | NAS model ONNX benchmark results |
| `results/final_benchmark.csv` | Combined benchmark across all models |

---

## Estimated Training Time (Google Colab T4 GPU)

| Stage | Time |
|-------|------|
| Q1: Teacher training (60 epochs) | ~35-45 min |
| Q1: Baseline students (3 x 50 epochs) | ~35-45 min |
| Q1: KD students (3 x 50 epochs) | ~40-50 min |
| Q2: NAS search (20 trials x 5 epochs) | ~25-35 min |
| Q2: Final model training (60 epochs) | ~15-20 min |
| **Total (first run)** | **~2.5 - 3.2 hours** |
| **Total (subsequent runs, checkpoints cached)** | **~3-5 minutes** |

---

## References

1. Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop.
2. Howard, A. et al. (2019). *Searching for MobileNetV3.* ICCV.
3. Iandola, F. et al. (2016). *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters.* arXiv.
4. Tan, M., Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
5. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
