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

## New Portable Architecture (May 2026)

The repository has been restructured to support **Modular Portability**. Every module is now a self-contained unit designed for immediate execution on platforms like Google Colab without complex external data dependencies.

### Key Changes:
- **Local Data Storage**: Each module folder (`baseline`, `credit`, etc.) and each `examples/` subfolder now contains its own `data/` directory.
- **Pre-Configured Paths**: Notebooks are updated to point to these local `data/` folders by default.
- **Colab Ready**: Simply upload a specific module folder (e.g., `indian_pines_uncertainty_quantification`) to your Google Drive, open the notebooks in Colab, and run.

## Repository Structure

```
Classification_Uncertainty_Quantification_Analysis/
├── classification_uncertainty_quantification/ # Main 6-band multispectral framework
│   ├── baseline/                # Baseline models & local data
│   │   ├── data/                # [NEW] Contains 6-band data.csv, ref.csv
│   │   └── ...
│   ├── credit/                  # CREDIT workflow & local data
│   ├── dapm/                    # DAPM workflow & local data
│   ├── ensemble/                # Ensemble methods & local data
│   ├── multi_cp/                # Multi-head CP & local data
│   └── sacp/                    # Self-Adaptive CP & local data
├── examples/                    # Dataset-specific self-contained suites
│   ├── 372_band_uncertainty_quantification/
│   │   ├── data/                # [NEW] Contains full_gt.mat, etc.
│   │   └── ...
│   ├── indian_pines_uncertainty_quantification/
│   │   ├── data/                # [NEW] Contains Indian_pines.mat, etc.
│   │   └── ...
│   └── [Other dataset-specific suites...]
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Professional Features

- **Unified Google Colab Integration**: All notebooks include automatic drive mounting and path resolution logic.
- **Automated Experimentation**: Key experiments are automated (e.g., SACP sensitivity loops).
- **Cleaned & Refined Notebooks**: Modular "Documentation First" structure with consolidated setup code.

## Requirements

### System Requirements
- Python 3.9 or higher
- GPU recommended for training (CUDA 11.2+)
- 16GB+ RAM recommended for large datasets

### Core Dependencies
```bash
pip install tensorflow>=2.10.0 numpy pandas scikit-learn matplotlib seaborn jupyter openpyxl
```

## Quick Start

### 1. Choose a Module
Decide which uncertainty method or dataset you want to analyze.

### 2. Upload to Google Drive
Upload the specific folder (e.g., `classification_uncertainty_quantification/baseline`) to your Drive.

### 3. Run in Colab
1. Open the `.ipynb` files in Google Colab.
2. Ensure the `REPO_ROOT` path in the first code cell matches your Drive folder location.
3. Run the cells. The data will be loaded automatically from the local `data/` folder within that directory.

## Features Matrix
| Module | Training Notebook | Uncertainty Notebook |
|--------|-------------------|---------------------|
| baseline | `model_training.ipynb` | `model_uncertainty_comparison.ipynb` |
| credit | `model_training_credit.ipynb` | (integrated) |
| dapm | `model_training_dapm_full.ipynb` | `model_uncertainty_dapm_full.ipynb` |
| ensemble | `model_training_ensembles.ipynb` | `model_uncertainty_credal_ensemble.ipynb` |
| multi_cp | `model_training_multihead.ipynb` | `model_uncertainty_multicp.ipynb` |
| sacp | (integrated) | `model_sacp_comparison.ipynb` |

## Citation

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

---
*Last updated: May 2026*
