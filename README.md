# Advanced Classification Uncertainty Quantification: Remote Sensing & Hyperspectral Analysis

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
    * [Method 7: SCMCP (Spatial MultiCP)](#method-7-scmcp-spatial-multicp)
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
* **Modular Design**: Each method is self-contained with its own training logic, detailed theory/summary logs in `.md` format, and result visualization.
* **Remote Sensing Ready**: Built-in support for multispectral data structures.

---

## 📂 Repository Structure

The project is organized into a modular architecture optimized for portability and experimental reproducibility. The main source code is organized within the `Classification/` directory:

```text
.
├── Classification/    # 📂 Main Project Folder (Upload this to MyDrive/)
│   ├── baseline/      # 📂 Standard UQ (MC Dropout, Temp Scaling)
│   ├── credit/        # 📂 CREDIT: Calibration-aware training
│   ├── dapm/          # 📂 DAPM: Deep Adaptive Predictive Modeling
│   ├── data/          # 📊 Raw Datasets (data.csv, multispectral/)
│   ├── ensemble/      # 📂 CreDE: Credal Deep Ensembles
│   ├── multicp/       # 📂 MultiCP: Multi-head Conformal Prediction
│   ├── multicp_sacp/  # 📂 SCMCP: Spatial MultiCP (New)
│   └── sacp/          # 📂 SACP: Self-Adaptive Conformal Prediction
├── examples/          # 📁 Example Dataset Suites (untouched)
└── README.md          # 📑 Documentation
```

---

## 📊 Dataset & Usage

**This is a Template Pipeline.**

To use this repository with your own classification data:

1. **Prepare Data**: Ensure your features and labels are in `.csv` or `.npy` format.
2. **Path Configuration**:
    * Place your data in the `Classification/data/` directory.
    * Notebooks use relative paths (e.g., `../data/data.csv`) for compatibility with Google Colab and local environments.
3. **Model Selection**: Choose from AlexNet, GFNet, or ViT-UNet by modifying the configuration in the respective notebooks.

---

## 🛠️ Workflow & Methodology

> **Note**: Each method subfolder in `Classification/` includes a comprehensive `.md` file detailing the mathematical framework, implementation logic, and experimental logs.

---

### Method 1: Baseline Uncertainty
**Location**: `Classification/baseline`
Establishes the performance and uncertainty floor using standard Bayesian approximations.
* **Techniques**: Monte Carlo Dropout, Temperature Scaling.
* **Models**: AlexNet, GFNet, ViT-UNet.

### Method 2: CREDIT
**Location**: `Classification/credit`
Confidence-Calibrated Robustness for Deep Image Classification.
* **Focus**: Improving the alignment between model confidence and actual performance.
* **Output**: Calibrated spatial uncertainty maps.

### Method 3: DAPM
**Location**: `Classification/dapm`
Deep Adaptive Predictive Modeling for classification under distribution shift.
* **Architecture**: Includes Encoder, Diffusion, and Multi-head decoders.
* **Result**: High-resolution p-value distributions and uncertainty masks.

### Method 4: Ensemble (CreDE)
**Location**: `Classification/ensemble`
Credal Deep Ensembles (CreDE) for robust uncertainty.
* **Process**: Training multiple model instances to capture epistemic uncertainty.
* **Metrics**: Credal entropy and variance-based measures.

### Method 5: MultiCP
**Location**: `Classification/multicp`
Multi-head Conformal Prediction.
* **Goal**: Distribution-free uncertainty sets with finite-sample validity.
* **Output**: Performance measures across different significance levels ($\alpha$).

### Method 6: SACP
**Location**: `Classification/sacp`
Self-Adaptive Conformal Prediction.
* **Advantage**: Online calibration and adaptation across varying spatial windows (ws=3, 5, 7, 9).
* **Summary**: Combined per-class coverage reports for all models.

### Method 7: SCMCP (Spatial MultiCP)
**Location**: `Classification/multicp_sacp`
Spatial Multi-Head Conformal Prediction.
* **Core**: Combines spatial probability smoothing with multi-head conformal intersection.
* **Feature**: Produces tight, spatially-coherent uncertainty regions by intersecting calibrated sets from $K$ heads after local smoothing.

---

## 🚀 Getting Started (Colab-First)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DaneshSelwal/Classification_Uncertainty_Quantification_Analysis
   ```
2. **Upload to Google Drive**:
   Upload the `Classification/` folder to your `MyDrive/` directory.
3. **Run in Colab**:
   Navigate to any phase (e.g., `baseline/Model_training.ipynb`) and open with Google Colab.
   * Notebooks are pre-configured to mount `/content/drive` and find data in `/content/drive/MyDrive/Classification/data/`.
4. **Execution Order**:
   **Data Prep** $\rightarrow$ **Baseline Training** $\rightarrow$ **Advanced UQ Methods** $\rightarrow$ **Visualization**.

---

## 📚 Resources & References

This project leverages state-of-the-art research in Uncertainty Quantification and Remote Sensing. Below are the key resources and research papers utilized in this pipeline:

### 📖 Research Papers

* **Weighted Aggregation of Conformity Scores**: *Weighted Aggregation of Conformity Scores for Classification*. [ArXiv:2407.10230](https://arxiv.org/abs/2407.10230)
* **Uncertainty Sets for Image Classifiers**: *Uncertainty Sets for Image Classifiers using Conformal Prediction*. [ArXiv:2009.14193](https://arxiv.org/abs/2009.14193)
* **Conformal Prediction via Label Ranking**: *Conformal Prediction for Deep Classifier via Label Ranking*. [ArXiv:2310.06430](https://arxiv.org/abs/2310.06430)
* **Credal Ensemble Distillation**: *Credal Ensemble Distillation for Uncertainty Quantification*. [ArXiv:2511.13766](https://arxiv.org/abs/2511.13766)
* **Calibration of Modern Neural Networks**: *On Calibration of Modern Neural Networks*. [ICML 2017](https://arxiv.org/abs/1706.04599)
* **Deep Ensembles**: Lakshminarayanan, B., et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [NeurIPS 2017](https://arxiv.org/abs/1612.01474)

### 🛠️ Libraries & Frameworks

* **SACP (Self-Adaptive CP)**: [GitHub](https://github.com/J4ckLiu/SACP)
* **TensorFlow / Keras**: Deep learning frameworks utilized for modeling architectures like AlexNet, GFNet, and ViT-UNet.

---
_This repository is developed under the guidance of Dr. Mahesh Pal._
