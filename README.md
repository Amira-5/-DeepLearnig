# Multimodal Deep Learning: From Pixels to Language
## Milestone 1: Data Preprocessing, Image Classification, Hyperparameter Tuning & Explainability

**Course:** CSE 485 / CSE 488 - Deep Learning  
**Institution:** Faculty of Engineering, Ain Shams University  
**Dataset:** CUB-200-2011 with Captions (`Multimodal-Fatima/CUB_train` & `CUB_test`)  

---

## 1. Project Overview

Milestone 1 establishes the foundational computer vision pipeline for fine-grained bird species classification across 200 categories. The visual encoder trained in this phase serves as the core feature extraction backbone for downstream multimodal tasks (Image Colorization and Image Captioning) in Milestone 2.

### Primary Objectives Completed
* **Task 1.1 (EDA & Data Preprocessing):** Analyzed class distributions, image dimensions, caption statistics, and exact RGB pixel duplicates. Exported stratified `train`, `validation`, and `test` metadata splits.
* **Task 1.2 (Classification & Encoder Training):** Constructed and trained a custom 5-block CNN baseline from scratch (`simple_cnn`), followed by transfer learning fine-tuning using an ImageNet-pretrained ResNet-18.
* **Hyperparameter Tuning:** Performed systematic comparative experiments between **Grid Search** and **Random Search** across learning rate, weight decay, and batch size configurations.
* **Model Explainability (Grad-CAM):** Implemented Gradient-weighted Class Activation Mapping on ResNet-18's final residual layer (`layer4`) to evaluate spatial focus across correct and misclassified species.

---

## 2. Directory Structure

The repository follows a clean, modular Python package structure adhering to SOLID software engineering design principles:

```text
DeepLearning-Project/
├── configs/                          # YAML configuration files for training runs
│   ├── cls_scratch.yaml              # Custom CNN baseline parameters
│   └── cls_resnet18.yaml             # Pretrained ResNet-18 transfer learning parameters
├── data/
│   ├── raw/                          # Raw dataset Hugging Face cache
│   └── processed/                    # Preprocessed datasets and CSV metadata
│       ├── metadata/                 # train.csv, validation.csv, test.csv
│       └── preprocessing_info.json
├── notebooks/                        # Experimental notebooks (prototyping only)
├── outputs/                          # Artifacts generated during execution
│   ├── checkpoints/                  # Saved PyTorch model weights (.pt)
│   ├── eda/                          # Summary statistics, duplicate logs, class distribution CSVs
│   ├── figures/                      # Learning curves, confusion matrices, Grad-CAM heatmaps
│   └── logs/                         # Training history JSONs and hyperparameter CSV results
├── src/                              # Core Python source package
│   ├── data/
│   │   ├── dataset.py                # PyTorch Dataset implementation (`CUBProcessedDataset`)
│   │   ├── loaders.py                # Data augmentation and DataLoader builders
│   │   └── preprocess.py             # Data scanning, cleaning, hashing, and split export
│   ├── evaluation/
│   │   ├── classification.py         # Evaluation loops and confusion matrix generation
│   │   └── metrics.py                # Accuracy, Precision, Recall, Macro-F1 computation
│   ├── models/
│   │   ├── classifier.py             # ImageClassifier wrapper (Encoder + Pool + Linear)
│   │   └── encoder.py                # SimpleCNNEncoder & TransferLearningEncoder
│   ├── training/
│   │   ├── classification.py         # Task-specific training setup
│   │   ├── trainer.py                # Generic PyTorch fit/training loop
│   │   └── tune.py                   # Grid Search vs. Random Search engine
│   ├── visualization/
│   │   ├── confusion.py              # Confusion matrix plotter
│   │   ├── curves.py                 # Loss and accuracy curve plotters
│   │   ├── gradcam.py                # Grad-CAM heatmap extraction script
│   │   └── tune_vis.py               # Hyperparameter search visualizer
│   └── cli.py                        # Unified Command Line Interface entry point
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

---

## 3. Environment Setup & Installation

### Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA 12.1+ support (recommended)

### Installation Steps

1. **Clone the Repository:**
   ```powershell
   git clone https://github.com/Amira-5/-DeepLearnig.git
   cd DeepLearning-Project
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows PowerShell
   # source venv/bin/activate    # Linux / macOS
   ```

3. **Install Dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Execution Workflow & CLI Commands

All training, evaluation, tuning, and visualization pipelines execute via pure Python modules and CLI commands.

### Step 1: Run Data Preprocessing & EDA Scan
Processes raw dataset records, parses 10 captions per record, strips exact RGB pixel duplicates, and exports stratified train/val/test metadata splits:

```powershell
python -m src.data.preprocess
```
*Outputs saved to `outputs/eda/` and `data/processed/metadata/`.*

---

### Step 2: Model Training

#### Option A: Train Custom Baseline CNN from Scratch (`simple_cnn`)
```powershell
python -m src.cli train --task classification --config configs/cls_scratch.yaml
```

#### Option B: Fine-Tune Pretrained ResNet-18 Transfer Learning
```powershell
python -m src.cli train --task classification --config configs/cls_resnet18.yaml
```

*CLI Command Overrides:*
* Change epoch count: `--epochs 15`
* Fast sanity test run: `--max-batches 5`

---

### Step 3: Model Evaluation & Metrics
Evaluates model checkpoints on validation or test splits, saving logs to `outputs/logs/` and confusion matrices to `outputs/figures/`:

```powershell
# Evaluate Fine-Tuned ResNet-18 on Test Split
python -m src.cli evaluate --task classification --config configs/cls_resnet18.yaml --split test

# Evaluate Custom CNN Baseline on Test Split
python -m src.cli evaluate --task classification --config configs/cls_scratch.yaml --split test
```

---

### Step 4: Hyperparameter Tuning (Grid Search vs. Random Search)
Runs comparative hyperparameter tuning experiments across learning rate, weight decay, and batch size:

```powershell
# 1. Run Search Loop
python -m src.training.tune

# 2. Generate 4-Panel Comparative Plots
python -m src.visualization.tune_vis
```
*Outputs:*
* Detailed logs: `outputs/logs/hyperparameter_tuning_results.csv`
* Summary stats: `outputs/logs/hyperparameter_tuning_summary.json`
* Visualization: `outputs/figures/hyperparameter_tuning_comparison.png`

---

### Step 5: Model Explainability (Grad-CAM)
Extracts spatial feature importance heatmaps from `layer4` of fine-tuned ResNet-18 and generates diagnostic grids for correct vs. misclassified test samples:

```powershell
python -m src.visualization.gradcam
```
*Output saved to `outputs/figures/gradcam_analysis.png`.*

---

## 5. Milestone 1 Benchmarking & Results

### Model Performance Comparison

| Model Architecture | Weights Initialization | Top-1 Test Accuracy | Test Macro F1-Score | Training Time / Epoch |
| :--- | :--- | :---: | :---: | :---: |
| **Custom CNN (`simple_cnn`)** | Scratch (Random) | ~18.4% | ~0.165 | ~12s |
| **ResNet-18** | Pretrained (ImageNet) | **~81.2%** | **~0.806** | ~24s |

### Hyperparameter Tuning Results (Custom CNN Baseline)

| Method | Total Trials | Total Time (s) | Best Validation Accuracy | Optimal Hyperparameters ($lr, wd, bs$) |
| :--- | :---: | :---: | :---: | :--- |
| **Grid Search** | 12 | ~145 s | 0.2140 | ($1 \times 10^{-3}$, $1 \times 10^{-4}$, 32) |
| **Random Search** | 12 | ~142 s | **0.2285** | ($7.4 \times 10^{-4}$, $3.2 \times 10^{-4}$, 32) |

---

## 6. Deliverables & Verification Checklist

- [x] **Pure Python Architecture:** Modular code layout under `src/` following SOLID principles.
- [x] **Command Line Interface:** Working CLI for `--task classification` (`train` and `evaluate`).
- [x] **EDA & Data Preprocessing:** Automated script stripping duplicate image hashes and structuring 10 captions.
- [x] **Baseline CNN Model:** Custom encoder trained from scratch (`cls_scratch.pt`).
- [x] **Transfer Learning Model:** Pretrained ResNet-18 fine-tuned on CUB-200-2011 (`cls_resnet18.pt`).
- [x] **Hyperparameter Tuning:** Empirical comparison of Grid Search vs. Random Search (`hyperparameter_tuning_comparison.png`).
- [x] **Model Explainability:** Grad-CAM heatmaps analyzing correct vs. incorrect predictions (`gradcam_analysis.png`).
