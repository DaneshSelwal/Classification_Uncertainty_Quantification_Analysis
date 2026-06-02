# 🚀 Advanced Classification Uncertainty Quantification: Remote Sensing & Hyperspectral Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
[![Uncertainty](https://img.shields.io/badge/Uncertainty-Conformal%20%7C%20Ensemble%20%7C%20Bayesian-green?style=for-the-badge)](https://img.shields.io/badge/Uncertainty-Conformal%20%7C%20Ensemble%20%7C%20Bayesian-green?style=for-the-badge)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

Welcome to the **Classification Uncertainty Quantification (UQ) Analysis** framework. This repository provides a comprehensive, modular pipeline for quantifying predictive uncertainty in complex classification tasks, specifically optimized for high-dimensional remote sensing and multispectral imagery.

Beyond traditional point-wise accuracy, this framework implements state-of-the-art methods including **Conformal Prediction**, **Credal Deep Ensembles**, and **Deep Adaptive Predictive Modeling**, ensuring every classification decision is backed by a statistically valid measure of confidence.

---

## 📑 Table of Contents

1. [📌 Project Overview](#-project-overview)
2. [📂 Repository Structure](#-repository-structure)
3. [📊 Dataset & Usage](#-dataset--usage)
4. [🛠️ Workflow & Methodology](#-workflow--methodology)
    * [Method 1: Baseline Uncertainty](#method-1-baseline-uncertainty)
    * [Method 2: CREDIT (Calibration-Aware)](#method-2-credit-calibration-aware)
    * [Method 3: DAPM (Deep Adaptive Modeling)](#method-3-dapm-deep-adaptive-modeling)
    * [Method 4: Ensemble (CreDE)](#method-4-ensemble-crede)
    * [Method 5: MultiCP (Multi-head Conformal)](#method-5-multicp-multi-head-conformal)
    * [Method 6: SACP (Self-Adaptive Conformal)](#method-6-sacp-self-adaptive-conformal)
5. [🚀 Getting Started](#-getting-started)
6. [📚 Resources & References](#-resources--references)

---

## 📌 Project Overview

This framework addresses the critical need for reliability in machine learning models deployed for Earth Observation and spatial analysis. By integrating multiple UQ paradigms, it allows researchers to:

* **Identify Out-of-Distribution (OOD) Pixels**: Detect areas where the model is likely to fail.
* **Calibrate Probabilistic Outputs**: Ensure that a 90% confidence score actually corresponds to 90% accuracy.
* **Generate Valid Prediction Sets**: Use Conformal Prediction to produce sets of classes that contain the true label with a user-specified probability (e.g., 95%).
* **Analyze Spatial Uncertainty**: Visualize uncertainty maps across entire scenes (Hyperspectral/Multispectral).

**Key Features:**
* **Deep Architectures**: AlexNet CNN, GFNet (Global Filter Network), and ViT-UNet (Vision Transformer).
* **Modular Design**: Each method is self-contained with its own training logic and result visualization.
* **Remote Sensing Ready**: Built-in support for multispectral data structures.

---

## 📂 Repository Structure

The project is organized into a modular architecture optimized for portability and experimental reproducibility:

```text
.
├── baseline/      # 📂 Standard UQ (MC Dropout, Temp Scaling)
├── credit/        # 📂 CREDIT: Calibration-aware training
├── dapm/          # 📂 DAPM: Deep Adaptive Predictive Modeling
├── data/          # 📊 Raw Datasets (data.csv, multispectral/)
├── ensemble/      # 📂 CreDE: Credal Deep Ensembles
├── multicp/       # 📂 MultiCP: Multi-head Conformal Prediction
├── sacp/          # 📂 SACP: Self-Adaptive Conformal Prediction
├── examples/      # 📁 Example Dataset Suites (untouched)
└── README.md      # 📑 Documentation
```

---

## 📊 Dataset & Usage

**This is a Template Pipeline.**

To use this repository with your own classification data:

1. **Prepare Data**: Ensure your features and labels are in `.csv` or `.npy` format.
2. **Path Configuration**:
    * Place your data in the `data/` directory.
    * Notebooks use relative paths (e.g., `../data/data.csv`) for compatibility with Google Colab and local environments.
3. **Model Selection**: Choose from AlexNet, GFNet, or ViT-UNet by modifying the configuration in the respective notebooks.

---

## 🛠️ Workflow & Methodology

---

### Method 1: Baseline Uncertainty
**Location**: `baseline/`
Establishes the performance and uncertainty floor using standard Bayesian approximations.
* **Techniques**: Monte Carlo Dropout, Temperature Scaling.
* **Models**: AlexNet, GFNet, ViT-UNet.

### Method 2: CREDIT
**Location**: `credit/`
Confidence-Calibrated Robustness for Deep Image Classification.
* **Focus**: Improving the alignment between model confidence and actual performance.
* **Output**: Calibrated spatial uncertainty maps.

### Method 3: DAPM
**Location**: `dapm/`
Deep Adaptive Predictive Modeling for classification under distribution shift.
* **Architecture**: Includes Encoder, Diffusion, and Multi-head decoders.
* **Result**: High-resolution p-value distributions and uncertainty masks.

### Method 4: Ensemble (CreDE)
**Location**: `ensemble/`
Credal Deep Ensembles (CreDE) for robust uncertainty.
* **Process**: Training multiple model instances to capture epistemic uncertainty.
* **Metrics**: Credal entropy and variance-based measures.

### Method 5: MultiCP
**Location**: `multicp/`
Multi-head Conformal Prediction.
* **Goal**: Distribution-free uncertainty sets with finite-sample validity.
* **Output**: Performance measures across different significance levels ($\alpha$).

### Method 6: SACP
**Location**: `sacp/`
Self-Adaptive Conformal Prediction.
* **Advantage**: Online calibration and adaptation across varying spatial windows (ws=3, 5, 7, 9).
* **Summary**: Combined per-class coverage reports for all models.

---

## 🚀 Getting Started (Colab-First)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DaneshSelwal/Classification_Uncertainty_Quantification_Analysis
   ```
2. **Upload to Google Drive**:
   Upload the repository folder to your `MyDrive/` directory.
3. **Run in Colab**:
   Navigate to any phase (e.g., `baseline/Model_training.ipynb`) and open with Google Colab.
   * Notebooks are pre-configured to mount `/content/drive`.
4. **Execution Order**:
   **Data Prep** $\rightarrow$ **Baseline Training** $\rightarrow$ **Advanced UQ Methods** $\rightarrow$ **Visualization**.

---

## 📚 Resources & References

This project leverages state-of-the-art research in Uncertainty Quantification and Remote Sensing.

### 📖 Research Papers

* **Conformal Prediction Tutorial**: Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. [ArXiv:2107.07511](https://arxiv.org/abs/2107.07511)
* **Adaptive Conformal Prediction**: Gibbs, I., & Candes, E. (2021). *Adaptive conformal inference under distribution shift*. [ArXiv:2106.01682](https://arxiv.org/abs/2106.01682)
* **Non-Exchangeable CP**: Barber, R. F., et al. (2023). *Conformal prediction beyond exchangeability*. [ArXiv:2202.13415](https://arxiv.org/abs/2202.13415)
* **Adaptive Coverage Policies (ACP)**: [ArXiv:2510.04318](https://arxiv.org/pdf/2510.04318)
* **Classification Calibration**: *On Calibration of Modern Neural Networks*. [ICML 2017](https://arxiv.org/abs/1706.04599)
* **Deep Ensembles**: Lakshminarayanan, B., et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [NeurIPS 2017](https://arxiv.org/abs/1612.01474)

### 🛠️ Libraries & Frameworks

* **MAPIE**: Model Agnostic Prediction Interval Estimator. [GitHub](https://github.com/scikit-learn-contrib/MAPIE)
* **PUNCC**: Predictive UNCertainty Calibration and Conformalization. [GitHub](https://github.com/deel-ai/puncc)
* **SACP (Self-Adaptive CP)**: [GitHub](https://github.com/J4ckLiu/SACP)
* **HCM (Hyperspherical Confidence Mapping)**: [GitHub](https://github.com/Abandoned-Puppy/HCM)
* **ACP (Adaptive Coverage Policies)**: [GitHub](https://github.com/GauthierE/adaptive-coverage-policies)
* **Optuna**: Bayesian Hyperparameter Optimization. [Website](https://optuna.org/)

---
_This repository is developed under the guidance of Dr. Mahesh Pal._
