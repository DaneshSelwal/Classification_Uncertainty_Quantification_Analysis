# Classification Uncertainty Quantification Analysis

> **Status**: This project is under ongoing development. Documentation and features are being actively improved.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-in%20development-yellow.svg)](.)

A comprehensive framework for uncertainty quantification in remote sensing image classification. This repository provides multiple uncertainty estimation methods including Bayesian deep learning, ensemble techniques, conformal prediction, and evidential deep learning for hyperspectral and multispectral image analysis.

## Overview

This project implements and compares various uncertainty quantification approaches for classification tasks in remote sensing:

| Method | Description | Key Features |
|--------|-------------|--------------|
| **Baseline** | Standard uncertainty estimation | Monte Carlo Dropout, Temperature Scaling |
| **CREDIT** | Confidence-calibrated uncertainty | Calibration-aware training |
| **DAPM** | Deep Adaptive Predictive Modeling | Full pipeline with adaptive mechanisms |
| **Ensemble (CreDE)** | Deep Ensemble methods | Credal Deep Ensembles for robust uncertainty |
| **MultiCP** | Multi-head Conformal Prediction | Distribution-free uncertainty sets |
| **SACP** | Self-Adaptive Conformal Prediction | Online calibration and adaptation |

## Repository Structure

```
Classification_Uncertainty_Quantification_Analysis/
├── baseline/                    # Baseline models (AlexNet, GFNet, ViT)
│   ├── model_training.ipynb     # Training pipeline
│   ├── model_uncertainty_comparison.ipynb
│   └── results/                 # Experiment outputs
├── credit/                      # CREDIT uncertainty workflow
│   ├── model_training_credit.ipynb
│   └── results/
├── dapm/                        # Deep Adaptive Predictive Modeling
│   ├── model_training_dapm_full.ipynb
│   ├── model_uncertainty_dapm_full.ipynb
│   └── results/
├── ensemble/                    # Ensemble & CreDE methods
│   ├── model_training_ensembles.ipynb
│   ├── model_uncertainty_credal_ensemble.ipynb
│   └── results/
├── multi_cp/                    # Multi-head Conformal Prediction
│   ├── model_training_multihead.ipynb
│   ├── model_uncertainty_multicp.ipynb
│   └── results/
├── sacp/                        # Self-Adaptive Conformal Prediction
│   ├── model_sacp_comparison.ipynb
│   └── results/
├── data/                        # Datasets (not tracked in git)
│   ├── Indian_pines/
│   ├── pavia/
│   ├── multispectral/
│   └── [...]
├── tools/                       # Utility scripts for maintenance
│   ├── patch_notebooks.py       # Standardization tool
│   ├── clean_notebooks.py       # Refinement tool
│   └── automate_sacp.py         # Experiment automation
├── .gitignore                   # Excludes large model artifacts
├── LICENSE                      # MIT License
├── README.md                    # This file
└── SKILL.md                     # Workflow conventions
```

## Professional Features (New)

This repository has been standardized for professional research workflows:

- **Unified Google Colab Integration**: All notebooks are pre-configured for seamless Google Colab execution. They include automatic drive mounting and path resolution logic.
- **Path Standardization**: Uses a robust `REPO_ROOT` based path logic to ensure all notebooks correctly reference shared data and save results in their respective module subfolders.
- **Automated Experimentation**: Key experiments are automated. For example, the SACP comparison now includes a built-in sensitivity loop for window sizes `[3, 5, 7, 9]`.
- **Cleaned & Refined Notebooks**: All notebooks follow a professional "Documentation First" structure, with redundant setup code consolidated and legacy snippets removed.

## Features

- **Multiple Architectures**: Support for AlexNet, Global Filter Networks (GFNet), and Vision Transformers (ViT)
- **Uncertainty Methods**: Bayesian (MC Dropout), Ensemble, Conformal Prediction, Evidential Deep Learning
- **Metrics**: Accuracy, Cohen's Kappa, Expected Calibration Error (ECE), Brier Score
- **Visualization**: Confusion matrices, calibration curves, uncertainty histograms
- **Reproducible Pipelines**: Modular Jupyter notebooks for each method

## Requirements

### System Requirements

- Python 3.9 or higher
- GPU recommended for training (CUDA 11.2+)
- 16GB+ RAM recommended for large hyperspectral datasets

### Python Dependencies

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install core dependencies
pip install tensorflow>=2.10.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0

# For Jupyter notebooks
pip install jupyter>=1.0.0
pip install ipywidgets>=7.6.0
```

### Optional Dependencies

```bash
# For Vision Transformer support
pip install vit-keras>=0.1.0

# For advanced conformal prediction
pip install mapie>=0.3.0

# For experiment tracking
pip install tensorboard>=2.10.0
```

## Quick Start

### 1. Dataset Setup

Place your hyperspectral/multispectral datasets in the `data/` directory:

```bash
# Example dataset structure
data/
├── Indian_pines/
│   ├── indian_pines_corrected.mat
│   └── indian_pines_gt.mat
├── pavia/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
└── multispectral/
    └── [your dataset]
```

### 2. Run Baseline Experiments

```bash
# Navigate to repository root
cd classification_uncertainty_analysis

# Open Jupyter Notebook
jupyter notebook baseline/model_training.ipynb
```

**Notebook Workflow:**
1. Run all cells in `Model_training.ipynb` to train models
2. Models and results are saved to `Baseline/results/`
3. Open `Model_uncertainty_comparison.ipynb` for uncertainty analysis

### 3. Run Specific Uncertainty Methods

Each module follows a consistent two-stage pipeline:

```bash
# Stage 1: Train models
jupyter notebook <Module>/Model_training_*.ipynb

# Stage 2: Compute uncertainty metrics
jupyter notebook <Module>/Model_uncertainty_*.ipynb
```

| Module | Training Notebook | Uncertainty Notebook |
|--------|-------------------|---------------------|
| baseline | `model_training.ipynb` | `model_uncertainty_comparison.ipynb` |
| credit | `model_training_credit.ipynb` | (integrated) |
| dapm | `model_training_dapm_full.ipynb` | `model_uncertainty_dapm_full.ipynb` |
| ensemble | `model_training_ensembles.ipynb` | `model_uncertainty_credal_ensemble.ipynb` |
| multi_cp | `model_training_multihead.ipynb` | `model_uncertainty_multicp.ipynb` |
| sacp | (integrated) | `model_sacp_comparison.ipynb` |

## Usage Examples

### Training a Baseline Model

```python
import tensorflow as tf
from pathlib import Path

# Configuration
DATA_DIR = Path("data/multispectral")
MODEL_SAVE_DIR = Path("baseline/results/models")
RESULTS_DIR = Path("baseline/results")
```

# Load data (example for .mat files)
from scipy.io import loadmat
data = loadmat(DATA_DIR / "dataset.mat")
X, y = data["features"], data["labels"]

# Train model (see notebooks for full implementation)
# Models: AlexNet, GFNet, ViT
```

### Computing Uncertainty Metrics

```python
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np

# After obtaining predictions and uncertainties
accuracy = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

# Expected Calibration Error
def compute_ece(confidences, predictions, targets, n_bins=10):
    # Implementation in uncertainty notebooks
    pass

ece = compute_ece(confidences, predictions, y_true)
```

## Output Format

Each module generates results in a standardized format:

```
<Module>/results/
├── models/              # Trained model weights (.keras or .h5)
├── predictions.csv      # Model predictions with confidence scores
├── metrics.json         # Accuracy, Kappa, ECE, Brier Score
├── confusion_matrix.png # Visualization
└── calibration_curve.png # Reliability diagram
```

### Metrics JSON Format

```json
{
  "model": "ViT",
  "dataset": "Indian_pines",
  "accuracy": 0.9234,
  "kappa": 0.9156,
  "ece": 0.0423,
  "brier_score": 0.0312,
  "uncertainty_method": "MC_Dropout",
  "num_samples": 50,
  "training_time_seconds": 1847.23
}
```

## Configuration

Modify experiment parameters in notebook cells:

```python
# Hyperparameters
CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "dropout_rate": 0.5,
    "mc_samples": 50,        # For MC Dropout
    "ensemble_size": 5,      # For Deep Ensembles
    "conformity_score": "softmax",  # For Conformal Prediction
}
```

## Datasets

This framework supports various hyperspectral and multispectral datasets:

| Dataset | Bands | Classes | Resolution |
|---------|-------|---------|------------|
| Indian Pines | 220 | 16 | 145x145 |
| Pavia University | 103 | 9 | 610x340 |
| Salinas | 204 | 16 | 512x217 |
| Custom | Variable | Variable | Variable |

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size or enable memory growth
tf.config.experimental.set_memory_growth(gpu, True)
```

**Dataset Loading Errors**
- Ensure `.mat` files are in correct format (MATLAB v7.3+)
- Check that label indices start from 0 or 1 (adjust in preprocessing)

**Notebook Kernel Crashes**
- Reduce model complexity or input patch size
- Close other memory-intensive applications

## Contributing

This project is under active development. Contributions are welcome:

1. Follow the workflow conventions in `SKILL.md`
2. Keep module outputs in respective `results/` directories
3. Do not commit large model artifacts under `models/`
4. Update documentation when adding new methods

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{classification_uncertainty_2026,
  title = {Classification Uncertainty Quantification Analysis},
  author = {Selwal, Danesh},
  year = {2026},
  url = {https://github.com/DaneshSelwal/Classification_Uncertainty_Quantification_Analysis}
}
```

## Contact

- **Repository**: https://github.com/DaneshSelwal/Classification_Uncertainty_Quantification_Analysis
- **Issues**: Please file bugs and feature requests on the GitHub issue tracker

---

*Last updated: April 2026*
