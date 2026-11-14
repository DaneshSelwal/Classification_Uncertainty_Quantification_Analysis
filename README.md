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
    * [Method 8: Focal Loss & CB Focal Loss](#method-8-focal-loss--cb-focal-loss)
    * [Method 9: EDL (Evidential Deep Learning)](#method-9-edl-evidential-deep-learning)
    * [Method 10: CDL (Credal Deep Learning)](#method-10-cdl-credal-deep-learning)
    * [Method 11: MambaHSI](#method-11-mambahsi)
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
│   ├── dofa/          # 📂 DOFA: Dynamic Wavelength Tokenization
│   ├── ensemble/      # 📂 CreDE: Credal Deep Ensembles
│   ├── georsclip/     # 📂 GeoRSCLIP: Geospatial Vision-Language Foundation Model (New)
│   ├── mambahsi/      # 📂 MambaHSI Architecture for classification
│   ├── multicp/       # 📂 MultiCP: Multi-head Conformal Prediction
│   ├── multicp_sacp/  # 📂 SCMCP: Spatial MultiCP
│   ├── sacp/          # 📂 SACP: Self-Adaptive Conformal Prediction
│   ├── spatialgcn/    # 📂 SpatialGCN: Spatial Graph Convolutional Networks (New)
│   └── trials/        # 🧪 Experimental Methods
│       ├── cb_focal_loss/
│       ├── cdl/
│       ├── conformal_reg/
│       ├── edl/
│       ├── edl_v2/
│       └── focal_loss/
├── examples/          # 📁 Dataset Suites (Indian Pines, Pavia Uni, DIAS, 6-Band, Hisar-3, Hissar-25)
│   ├── indian_pines_uncertainty_quantification/
│   ├── pavia_university_uncertainty_quantification/
│   ├── dias_uncertainty_quantification/
│   ├── 6_band_uncertainty_quantification/
│   ├── hisar_3mts_uncertainty_quantification/
│   └── hissar_25_uncertainty_quantification/
├── kaggle_staging/    # 🚀 Automated Kaggle GPU/CPU runner staging & packaging
│   ├── create_staged_runners.py # Builds all 8 unified GPU (k11-k14) & CPU (k15-k18) runners
│   ├── verify_staging.py        # Verifies 193 staging, packaging & security checks
│   ├── ip-suite-runner-k11/     # ... through hissar25-suite-runner-k14/ (GPU runners)
│   └── ip-cpu-runner-k15/       # ... through hissar25-cpu-runner-k18/ (CPU runners)
├── scripts/           # 🧪 Automated verification, symbol-repair, and synchronization tools
│   ├── fix_all_missing_symbols.py # Cross-suite AST symbol completeness & import fixer
│   └── verify_suites.py         # 224-check notebook schema, AST & regression verifier
├── kaggle_name_mapping.json # 🗂️ Obfuscation mapping for remote execution
└── README.md          # 📑 Documentation
```

---

## 🌐 Dataset Suites & Uncertainty Quantification Benchmarks

Each dataset suite in `examples/<dataset_suite>/` contains full standalone Jupyter notebooks, run configurations, evaluation logs, and visualization reports. Large trained model weights (`.keras`) are decoupled and hosted on Hugging Face Hub to keep the git repository lightweight and fast.

| Dataset Suite | Scene Dimensions | Bands | Classes | Primary Backbones | UQ Methods Evaluated | Hugging Face Model Repository |
|---|---|---|---|---|---|---|
| **Indian Pines** | $145 \times 145$ | 220 | 16 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-IndianPines-Models`](https://huggingface.co/DaneshSelwal/Classification-IndianPines-Models) |
| **Pavia University** | $610 \times 340$ | 103 | 9 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-PaviaUniversity-Models`](https://huggingface.co/DaneshSelwal/Classification-PaviaUniversity-Models) |
| **DIAS (65-Band)** | $512 \times 512$ | 65 | 11 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-DIAS-65Bands`](https://huggingface.co/DaneshSelwal/Classification-DIAS-65Bands) |
| **6-Band Multispectral** | $330 \times 307$ | 6 | 6 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-6Bands-Models`](https://huggingface.co/DaneshSelwal/Classification-6Bands-Models) |
| **Hisar-3** | $1733 \times 2647$ | 4 | 8 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-Hisar3-Models`](https://huggingface.co/DaneshSelwal/Classification-Hisar3-Models) |
| **Hissar-25** | $722 \times 1014$ | 4 | 8 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-Hissar25-Models`](https://huggingface.co/DaneshSelwal/Classification-Hissar25-Models) |

### 🚀 Kaggle Execution Runners, Datasets & Model Repositories

Each of the four active benchmark suites is paired with a Kaggle dataset, an 11-stage GPU runner, a 10-stage CPU UQ/distillation runner (reusing pre-trained backbones and ensemble members from Hugging Face), and a public Hugging Face Hub model repository.

| Suite | Local Directory | Kaggle Dataset | GPU Runner (11 Stages) | CPU Runner (10 Stages) | Hugging Face Model Repo | Pre-Trained Weights on HF |
|---|---|---|---|---|---|---|
| **Indian Pines** | [`indian_pines_uncertainty_quantification/`](examples/indian_pines_uncertainty_quantification/) | [`daneshselwal/ip-dataset-k01`](https://www.kaggle.com/datasets/daneshselwal/ip-dataset-k01) | [`daneshselwal/ip-suite-runner-k11`](https://www.kaggle.com/code/daneshselwal/ip-suite-runner-k11) | [`daneshselwal/ip-cpu-runner-k15`](https://www.kaggle.com/code/daneshselwal/ip-cpu-runner-k15) | [`Classification-IndianPines-Models`](https://huggingface.co/DaneshSelwal/Classification-IndianPines-Models) | 6 backbones + 15 ensembles |
| **Pavia University** | [`pavia_university_uncertainty_quantification/`](examples/pavia_university_uncertainty_quantification/) | [`daneshselwal/pavia-dataset-k02`](https://www.kaggle.com/datasets/daneshselwal/pavia-dataset-k02) | [`daneshselwal/pavia-suite-runner-k12`](https://www.kaggle.com/code/daneshselwal/pavia-suite-runner-k12) | [`daneshselwal/pavia-cpu-runner-k16`](https://www.kaggle.com/code/daneshselwal/pavia-cpu-runner-k16) | [`Classification-PaviaUniversity-Models`](https://huggingface.co/DaneshSelwal/Classification-PaviaUniversity-Models) | 6 backbones + 15 ensembles |
| **Hisar-3** | [`hisar_3mts_uncertainty_quantification/`](examples/hisar_3mts_uncertainty_quantification/) | [`daneshselwal/hisar3-dataset-k03`](https://www.kaggle.com/datasets/daneshselwal/hisar3-dataset-k03) | [`daneshselwal/hisar3-suite-runner-k13`](https://www.kaggle.com/code/daneshselwal/hisar3-suite-runner-k13) | [`daneshselwal/hisar3-cpu-runner-k17`](https://www.kaggle.com/code/daneshselwal/hisar3-cpu-runner-k17) | [`Classification-Hisar3-Models`](https://huggingface.co/DaneshSelwal/Classification-Hisar3-Models) | 6 backbones |
| **Hissar-25** | [`hissar_25_uncertainty_quantification/`](examples/hissar_25_uncertainty_quantification/) | [`daneshselwal/hissar25-dataset-k04`](https://www.kaggle.com/datasets/daneshselwal/hissar25-dataset-k04) | [`daneshselwal/hissar25-suite-runner-k14`](https://www.kaggle.com/code/daneshselwal/hissar25-suite-runner-k14) | [`daneshselwal/hissar25-cpu-runner-k18`](https://www.kaggle.com/code/daneshselwal/hissar25-cpu-runner-k18) | [`Classification-Hissar25-Models`](https://huggingface.co/DaneshSelwal/Classification-Hissar25-Models) | 6 backbones |

### 📋 11-Stage Uncertainty Quantification Pipeline (Per Suite)

Baseline backbone weights (`AlexNet_CNN`, `GFNet`, `ViT_UNet`) trained by [`baseline/model_training.ipynb`](examples/indian_pines_uncertainty_quantification/baseline/model_training.ipynb) are hosted on Hugging Face Hub and automatically pulled at runner startup, followed by the 11 downstream stages executed via Papermill (CPU runners `k15`–`k18` skip Stage 3 and pull `models/ensembles/*.keras` from Hugging Face):

| Stage | Notebook Path (relative to suite dir) | Role & Method | Runner Scope |
|---|---|---|---|
| **0 (Pretrained)** | `baseline/model_training.ipynb` | Train & export `AlexNet_CNN`, `GFNet`, `ViT_UNet` backbones (hosted on HF) | Pre-staged on HF |
| **1** | `baseline/model_uncertainty_comparison.ipynb` | Baseline Conformal Prediction comparison (`SplitCP`, `CcCP`, `RC3P`, `ClCP`, `RAPS`) | GPU & CPU |
| **2** | `sacp/model_sacp_comparison.ipynb` | Spatial-Aware Conformal Prediction (`SACP`) evaluation | GPU & CPU |
| **3** | `ensemble/model_training_ensembles.ipynb` | Deep Credal Ensemble training ($M=5$ per backbone; reuses HF weights if present) | GPU only |
| **4** | `credit/model_training_credit.ipynb` | CREDIT distillation training and credal uncertainty evaluation | GPU & CPU |
| **5** | `ensemble/model_uncertainty_credal_ensemble.ipynb` | CreDE credal ensemble uncertainty quantification | GPU & CPU |
| **6** | `multi_cp/model_training_multihead.ipynb` | Multi-head backbone training for Multi-CP | GPU & CPU |
| **7** | `multi_cp/model_uncertainty_multicp.ipynb` | Multi-Head Conformal Prediction (`Multi-CP`) evaluation | GPU & CPU |
| **8** | `dapm/model_training_dapm_full.ipynb` | Dynamic Adaptive Prediction Mapping (`DAPM`) training | GPU & CPU |
| **9** | `dapm/model_uncertainty_dapm_full.ipynb` | `DAPM` domain-adaptive uncertainty quantification | GPU & CPU |
| **10** | `dapm/Model_uncertainty_dapm_cp.ipynb` | `DAPM` Conformal Prediction evaluation | GPU & CPU |
| **11** | `multicp_sacp/Model_uncertainty_multicp_sacp.ipynb` | Spatial MultiCP (`SCMCP`) evaluation | GPU & CPU |

#### Staging & Verification Commands

```bash
python scripts/verify_suites.py          # Runs 224 structural, AST & regression checks across all 4 suites
python kaggle_staging/create_staged_runners.py  # Rebuilds all 8 Kaggle GPU/CPU runner notebooks
python kaggle_staging/verify_staging.py  # Runs 193 staging, packaging & token-hygiene checks
```

### Pipeline Architecture
1. **Scene Ingestion & Geometry Normalization**: Robust percentile scaling `[2.0, 98.0]` per spectral band.
2. **Patch Extraction**: $9 \times 9 \times B$ spatial-spectral patches with background zero-exclusion.
3. **Deep Backbones**: Lightweight AlexNet CNN, Global Filter Network (GFNet) frequency domain mixer, and Vision Transformer UNet (ViT-UNet).
4. **Statistical Uncertainty**: Conformal Prediction guarantees marginal and class-conditional coverage ($1 - \alpha$), Spatial-Aware CP ensures contiguous certainty maps, and Credal Deep Ensembles capture epistemic ambiguity.
5. **Evaluation Metrics**: Overall Accuracy (OA), Average Accuracy (AA), Cohen's Kappa ($\kappa$), Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), Brier score, Prediction Set Efficiency, and Empirical Coverage.


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

### Method 8: Focal Loss & CB Focal Loss
**Location**: `Classification/trials/focal_loss` & `Classification/trials/cb_focal_loss`
* **Focus**: Addressing class imbalance using standard and Class-Balanced Focal Loss.
* **Output**: Improved minority class prediction with respective uncertainty estimates.

### Method 9: EDL (Evidential Deep Learning)
**Location**: `Classification/trials/edl` & `Classification/trials/edl_v2`
* **Core**: Uses Subjective Logic to place a Dirichlet distribution over class probabilities.
* **Goal**: Directly quantifies epistemic and aleatoric uncertainty without sampling.

### Method 10: CDL (Credal Deep Learning)
**Location**: `Classification/trials/cdl`
* **Goal**: Robust uncertainty quantification using imprecise probabilities and credal sets.

### Method 11: MambaHSI
**Location**: `Classification/mambahsi`
* **Architecture**: State-of-the-art Mamba state-space models adapted for Hyperspectral Imagery (HSI).
* **Focus**: Efficient long-range spatial-spectral dependencies modeling.

---

### Method 12: DOFA
**Location**: `Classification/dofa`
Dynamic Wavelength Tokenization.
* **Architecture**: DOFA Spectral and DOFA Hiera Fusion for robust multispectral tokenization.
* **Focus**: Analyzing prediction stability across continuous spectral channels.

---

### Method 13: GeoRSCLIP
**Location**: `Classification/georsclip`
Geospatial Vision-Language Foundation Model.
* **Architecture**: Text-head fine-tuning on GeoRSCLIP representations.
* **Focus**: Leveraging multimodal pre-training for remote sensing tasks.

### Method 14: SpatialGCN
**Location**: `Classification/spatialgcn`
Spatial Graph Convolutional Networks.
* **Architecture**: GCN processing on spatial grid topologies.
* **Focus**: Capturing complex spatial dependencies in classification maps.

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
