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
├── examples/          # 📁 11 Benchmark Dataset Suites (4 Completed, 2 Continuation, 5 Staged Ready)
│   ├── indian_pines_uncertainty_quantification/
│   ├── pavia_university_uncertainty_quantification/
│   ├── dias_uncertainty_quantification/
│   ├── 6_band_uncertainty_quantification/
│   ├── hisar_3mts_uncertainty_quantification/
│   ├── hissar_25_uncertainty_quantification/
│   ├── pavia_uncertainty_quantification/
│   ├── 372_band_uncertainty_quantification/
│   ├── multispectral_uncertainty_quantification/
│   ├── achla_multi_temporal_uncertainty_quantification/
│   └── planet_data_hisar_uncertainty_quantification/
├── kaggle_staging/    # 🚀 Automated Kaggle GPU/CPU runner staging & packaging
│   ├── create_staged_runners.py # Builds all 18 unified GPU (k11-k14, k19-k23) & CPU (k15-k18, k24-k28) runners
│   ├── verify_staging.py        # Verifies 928 staging, packaging & token-hygiene checks
│   ├── ip-suite-runner-k11/     # ... through planethisar-suite-runner-k23/ (9 GPU runners)
│   └── ip-cpu-runner-k15/       # ... through planethisar-cpu-runner-k28/ (9 CPU runners)
├── scripts/           # 🧪 Automated verification, symbol-repair, and synchronization tools
│   ├── fix_all_missing_symbols.py # Cross-suite AST symbol completeness & import fixer
│   └── verify_suites.py         # 449-check notebook schema, AST & regression verifier across 11 suites
├── kaggle_name_mapping.json # 🗂️ Obfuscation mapping for all 9 datasets (k01-k09) & 18 runners
└── README.md          # 📑 Documentation
```

---

## 🌐 Dataset Suites & Uncertainty Quantification Benchmarks

Each dataset suite in `examples/<dataset_suite>/` contains full standalone Jupyter notebooks, run configurations, evaluation logs, and visualization reports. Large trained model weights (`.keras`) are decoupled and hosted on Hugging Face Hub to keep the git repository lightweight and fast.

| Dataset Suite | Scene Dimensions | Bands | Classes | Primary Backbones | UQ Methods Evaluated | Hugging Face Model Repository |
|---|---|---|---|---|---|---|
| [**Indian Pines**](examples/indian_pines_uncertainty_quantification/) | $145 \times 145$ | 220 | 16 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-IndianPines-Models`](https://huggingface.co/DaneshSelwal/Classification-IndianPines-Models) |
| [**Pavia University**](examples/pavia_university_uncertainty_quantification/) | $610 \times 340$ | 103 | 9 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-PaviaUniversity-Models`](https://huggingface.co/DaneshSelwal/Classification-PaviaUniversity-Models) |
| [**DIAS (65-Band)**](examples/dias_uncertainty_quantification/) | $512 \times 512$ | 65 | 11 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-DIAS-65Bands`](https://huggingface.co/DaneshSelwal/Classification-DIAS-65Bands) |
| [**6-Band Multispectral**](examples/6_band_uncertainty_quantification/) | $330 \times 307$ | 6 | 6 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM + Foundation Models | [`Classification-6Bands-Models`](https://huggingface.co/DaneshSelwal/Classification-6Bands-Models) |
| [**Hisar-3**](examples/hisar_3mts_uncertainty_quantification/) | $1733 \times 2647$ | 4 | 8 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-Hisar3-Models`](https://huggingface.co/DaneshSelwal/Classification-Hisar3-Models) |
| [**Hissar-25**](examples/hissar_25_uncertainty_quantification/) | $722 \times 1014$ | 4 | 8 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-Hissar25-Models`](https://huggingface.co/DaneshSelwal/Classification-Hissar25-Models) |
| [**Pavia Centre**](examples/pavia_uncertainty_quantification/) | $1096 \times 715$ | 102 | 9 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-PaviaCentre-Models`](https://huggingface.co/DaneshSelwal/Classification-PaviaCentre-Models) |
| [**372-Band Hyperspectral**](examples/372_band_uncertainty_quantification/) | $1101 \times 566$ | 372 | 13 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-372Band-Models`](https://huggingface.co/DaneshSelwal/Classification-372Band-Models) |
| [**ETM 108 Multispectral**](examples/multispectral_uncertainty_quantification/) | $330 \times 307$ | 108 | 7 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-ETM108-Models`](https://huggingface.co/DaneshSelwal/Classification-ETM108-Models) |
| [**Achla Multi-Temporal SAR**](examples/achla_multi_temporal_uncertainty_quantification/) | $964 \times 1028$ | 32 | 12 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-AchlaSAR-Models`](https://huggingface.co/DaneshSelwal/Classification-AchlaSAR-Models) |
| [**Planet Hisar CSV**](examples/planet_data_hisar_uncertainty_quantification/) | $1733 \times 2647$ | 4 | 8 | AlexNet, GFNet, ViT-UNet | Split-CP, Multi-CP, SACP, SCMCP, CREDIT, CreDE, DAPM | [`Classification-PlanetHisar-Models`](https://huggingface.co/DaneshSelwal/Classification-PlanetHisar-Models) |

### 🚀 Kaggle Execution Runners, Datasets & Model Repositories

Each dataset suite is paired with a private Kaggle dataset, remote execution runners (unified GPU/CPU suite runners or dedicated method kernels), and a public Hugging Face Hub model repository for zero-weight git cleanliness.

| Suite | Local Directory | Kaggle Dataset | GPU Runner | CPU Runner | Hugging Face Model Repo | Pre-Trained Weights on HF |
|---|---|---|---|---|---|---|
| **Indian Pines** | [`indian_pines_uncertainty_quantification/`](examples/indian_pines_uncertainty_quantification/) | [`daneshselwal/ip-dataset-k01`](https://www.kaggle.com/datasets/daneshselwal/ip-dataset-k01) | [`daneshselwal/ip-suite-runner-k11`](https://www.kaggle.com/code/daneshselwal/ip-suite-runner-k11) | [`daneshselwal/ip-cpu-runner-k15`](https://www.kaggle.com/code/daneshselwal/ip-cpu-runner-k15) | [`Classification-IndianPines-Models`](https://huggingface.co/DaneshSelwal/Classification-IndianPines-Models) | 6 backbones + 15 ensembles (21/21 models) |
| **Pavia University** | [`pavia_university_uncertainty_quantification/`](examples/pavia_university_uncertainty_quantification/) | [`daneshselwal/pavia-dataset-k02`](https://www.kaggle.com/datasets/daneshselwal/pavia-dataset-k02) | [`daneshselwal/pavia-suite-runner-k12`](https://www.kaggle.com/code/daneshselwal/pavia-suite-runner-k12) | [`daneshselwal/pavia-cpu-runner-k16`](https://www.kaggle.com/code/daneshselwal/pavia-cpu-runner-k16) | [`Classification-PaviaUniversity-Models`](https://huggingface.co/DaneshSelwal/Classification-PaviaUniversity-Models) | 6 backbones + 15 ensembles (21/21 models) |
| **DIAS (65-Band)** | [`dias_uncertainty_quantification/`](examples/dias_uncertainty_quantification/) | [`daneshselwal/dias-dataset-k01`](https://www.kaggle.com/datasets/daneshselwal/dias-dataset-k01) | [`daneshselwal/dias-*-k*`](https://www.kaggle.com/daneshselwal/code) (20 dedicated method kernels) | — | [`Classification-DIAS-65Bands`](https://huggingface.co/DaneshSelwal/Classification-DIAS-65Bands) | 51 models hosted (Backbones, Ensembles, MultiCP, DAPM, SACP) |
| **6-Band Multispectral** | [`6_band_uncertainty_quantification/`](examples/6_band_uncertainty_quantification/) | [`daneshselwal/dias-6-bands-data`](https://www.kaggle.com/datasets/daneshselwal/dias-6-bands-data) | [`daneshselwal/cuq6-*-k*`](https://www.kaggle.com/daneshselwal/code) (17 dedicated method kernels) | — | [`Classification-6Bands-Models`](https://huggingface.co/DaneshSelwal/Classification-6Bands-Models) | 6 backbones + extended models |
| **Hisar-3** | [`hisar_3mts_uncertainty_quantification/`](examples/hisar_3mts_uncertainty_quantification/) | [`daneshselwal/hisar3-dataset-k03`](https://www.kaggle.com/datasets/daneshselwal/hisar3-dataset-k03) | [`daneshjangra/hisar3-suite-runner-k13`](https://www.kaggle.com/code/daneshjangra/hisar3-suite-runner-k13) (`QUEUED` Stg 3–11) | [`daneshselwal/hisar3-cpu-runner-k17`](https://www.kaggle.com/code/daneshselwal/hisar3-cpu-runner-k17) / [`geminicli/...-s1`](https://www.kaggle.com/code/geminicli/hisar3-cpu-runner-k17-s1) / [`daneshselwal1/...-s2`](https://www.kaggle.com/code/daneshselwal1/hisar3-cpu-runner-k17-s2) | [`Classification-Hisar3-Models`](https://huggingface.co/DaneshSelwal/Classification-Hisar3-Models) | `22` weights on HF (6 backbones, 12 ensembles, 4 AlexNet MultiHead); `k13` GPU `QUEUED` + 3 CPU shards `RUNNING` |
| **Hissar-25** | [`hissar_25_uncertainty_quantification/`](examples/hissar_25_uncertainty_quantification/) | [`daneshselwal/hissar25-dataset-k04`](https://www.kaggle.com/datasets/daneshselwal/hissar25-dataset-k04) | [`daneshjangra/hissar25-suite-runner-k14`](https://www.kaggle.com/code/daneshjangra/hissar25-suite-runner-k14) | [`daneshjangra/hissar25-cpu-runner-k18`](https://www.kaggle.com/code/daneshjangra/hissar25-cpu-runner-k18) | [`Classification-Hissar25-Models`](https://huggingface.co/DaneshSelwal/Classification-Hissar25-Models) | `45` weights on HF (6 backbones, 15 ensembles, 6 MultiHead, 18 DAPM Full); `daneshjangra/k18` `RUNNING` Stg 4, 5, 11 |
| **Pavia Centre** | [`pavia_uncertainty_quantification/`](examples/pavia_uncertainty_quantification/) | [`daneshselwal/pavia-centre-dataset-k05`](https://www.kaggle.com/datasets/daneshselwal/pavia-centre-dataset-k05) | [`daneshjangra/pavia-suite-runner-k19`](https://www.kaggle.com/code/daneshjangra/pavia-suite-runner-k19) (`QUEUED` Stg 3–11) | [`daneshselwal/pavia-cpu-runner-k24`](https://www.kaggle.com/code/daneshselwal/pavia-cpu-runner-k24) / [`geminicli/...-s1`](https://www.kaggle.com/code/geminicli/pavia-cpu-runner-k24-s1) / [`geminicli/...-s2`](https://www.kaggle.com/code/geminicli/pavia-cpu-runner-k24-s2) | [`Classification-PaviaCentre-Models`](https://huggingface.co/DaneshSelwal/Classification-PaviaCentre-Models) | `19` weights on HF (6 backbones, 11 ensembles, AlexNet CREDIT/MultiHead); `k19` GPU `QUEUED` + 3 CPU shards `RUNNING` |
| **372-Band** | [`372_band_uncertainty_quantification/`](examples/372_band_uncertainty_quantification/) | [`daneshselwal/band372-dataset-k06`](https://www.kaggle.com/datasets/daneshselwal/band372-dataset-k06) | [`daneshjangra/band372-suite-runner-k20`](https://www.kaggle.com/code/daneshjangra/band372-suite-runner-k20) | [`daneshselwal/band372-cpu-runner-k25`](https://www.kaggle.com/code/daneshselwal/band372-cpu-runner-k25) | [`Classification-372Band-Models`](https://huggingface.co/DaneshSelwal/Classification-372Band-Models) | `54` weights on HF (backbones, ensembles, CREDIT, MultiHead, DAPM Full); 210 result files extracted |
| **ETM 108** | [`multispectral_uncertainty_quantification/`](examples/multispectral_uncertainty_quantification/) | [`daneshselwal/etm108-dataset-k07`](https://www.kaggle.com/datasets/daneshselwal/etm108-dataset-k07) | [`daneshselwal/etm108-suite-runner-k21`](https://www.kaggle.com/code/daneshselwal/etm108-suite-runner-k21) (`RUNNING` Stg 0–11) | [`daneshselwal/etm108-cpu-runner-k26`](https://www.kaggle.com/code/daneshselwal/etm108-cpu-runner-k26) | [`Classification-ETM108-Models`](https://huggingface.co/DaneshSelwal/Classification-ETM108-Models) | `5/6` Stage 0 backbones live on HF (`ViT_UNet_best` added); `k21` GPU + `k26` CPU `RUNNING` |
| **Achla SAR** | [`achla_multi_temporal_uncertainty_quantification/`](examples/achla_multi_temporal_uncertainty_quantification/) | [`daneshselwal/achla-dataset-k08`](https://www.kaggle.com/datasets/daneshselwal/achla-dataset-k08) | [`daneshselwal/achla-suite-runner-k22`](https://www.kaggle.com/code/daneshselwal/achla-suite-runner-k22) (`QUEUED` Stg 0–11) | [`daneshjangra/achla-cpu-runner-k27`](https://www.kaggle.com/code/daneshjangra/achla-cpu-runner-k27) / [`daneshselwal1/...-s2`](https://www.kaggle.com/code/daneshselwal1/achla-cpu-runner-k27-s2) | [`Classification-AchlaSAR-Models`](https://huggingface.co/DaneshSelwal/Classification-AchlaSAR-Models) | `23` weights live on HF (`6/6` backbones, `8/15` ensembles, `9` MultiHead); `k22` GPU `QUEUED` + 2 CPU shards `RUNNING` |
| **Planet Hisar CSV** | [`planet_data_hisar_uncertainty_quantification/`](examples/planet_data_hisar_uncertainty_quantification/) | [`daneshselwal/planethisar-dataset-k09`](https://www.kaggle.com/datasets/daneshselwal/planethisar-dataset-k09) | [`daneshselwal/planethisar-suite-runner-k23`](https://www.kaggle.com/code/daneshselwal/planethisar-suite-runner-k23) | [`daneshjangra/planethisar-cpu-runner-k28`](https://www.kaggle.com/code/daneshjangra/planethisar-cpu-runner-k28) / [`daneshselwal1/...-s2`](https://www.kaggle.com/code/daneshselwal1/planethisar-cpu-runner-k28-s2) | [`Classification-PlanetHisar-Models`](https://huggingface.co/DaneshSelwal/Classification-PlanetHisar-Models) | `4/6` Stage 0 backbones live on HF (`AlexNet`, `GFNet_best`, `ViT_UNet_best`); 2 CPU shards `RUNNING` |

### 📊 Dataset-Wise Notebook Execution Progress (11 Benchmark Suites)

```text
Codebase, AST & Staging Checks [████████████████████] 100%  (All 11 suites verified — 449/449 suite + 932/932 staging)
Cloud Datasets & HF Provisioned[████████████████████] 100%  (11 / 11 suites provisioned — 9 Kaggle datasets + 271 HF weights)
Core Stage Completion          [████████████░░░░░░░░]  58%  (77 / 132 core stages complete; 4 GPU [1 RUN/3 Q] + 12 CPU RUNNING)
```

| Dataset Suite | Status | Execution Progress | Done / Total | Active / Latest Execution Runners | Model Checkpoints on HF Hub (`271` Total) |
|---|---|---|---|---|---|
| [**Indian Pines**](examples/indian_pines_uncertainty_quantification/) | ✅ Completed | `████████████████████` **100%** | `12 / 12` Done | [`k11`](https://www.kaggle.com/code/daneshselwal/ip-suite-runner-k11) (`COMPLETE`) / [`k15`](https://www.kaggle.com/code/daneshselwal/ip-cpu-runner-k15) | `21` (`6` backbones + `15` ensembles) |
| [**Pavia University**](examples/pavia_university_uncertainty_quantification/) | ✅ Completed | `████████████████████` **100%** | `12 / 12` Done | [`k12`](https://www.kaggle.com/code/daneshselwal/pavia-suite-runner-k12) (`COMPLETE`) / [`k16`](https://www.kaggle.com/code/daneshselwal/pavia-cpu-runner-k16) | `21` (`6` backbones + `15` ensembles) |
| [**DIAS (65-Band)**](examples/dias_uncertainty_quantification/) | ✅ Completed | `████████████████████` **100%** | `12 / 12` Done | Dedicated Kaggle Runners (`dias-*-k*`, 20 kernels) | `51` ([`Classification-DIAS-65Bands`](https://huggingface.co/DaneshSelwal/Classification-DIAS-65Bands)) |
| [**6-Band Multispectral**](examples/6_band_uncertainty_quantification/) | ✅ Completed | `████████████████████` **100%** | `12 / 12` Done (+ 4 Ext) | Dedicated Kaggle Runners (`cuq6-*-k*`, 17 kernels) | `6` ([`Classification-6Bands-Models`](https://huggingface.co/DaneshSelwal/Classification-6Bands-Models)) |
| [**372-Band**](examples/372_band_uncertainty_quantification/) | ✅ Completed | `████████████████████` **100%** | `12 / 12` Done (`ws3/5/7` SCMCP) | [`k20`](https://www.kaggle.com/code/daneshjangra/band372-suite-runner-k20) (extracted) / [`k25`](https://www.kaggle.com/code/daneshselwal/band372-cpu-runner-k25) | `54` (Backbones, Ensembles, CREDIT, MultiHead, DAPM) |
| [**Hissar-25**](examples/hissar_25_uncertainty_quantification/) | 🔄 CPU Running | `███████████████░░░░░` **75%** | `9 / 12` Done (`ws3/5` SCMCP) | [`daneshjangra/hissar25-cpu-runner-k18`](https://www.kaggle.com/code/daneshjangra/hissar25-cpu-runner-k18) (`RUNNING` Stg 4, 5, 11) | `45` (`21/21` + `6` MultiHead + `18` DAPM Full) |
| [**Pavia Centre**](examples/pavia_uncertainty_quantification/) | 🔄 GPU Q + CPU Running | `██████░░░░░░░░░░░░░░` **33%** | `4 / 12` Done (+ Stg 3/4/6 partial) | [`daneshjangra/k19`](https://www.kaggle.com/code/daneshjangra/pavia-suite-runner-k19) (GPU `QUEUED` `3..11`), [`daneshselwal/k24`](https://www.kaggle.com/code/daneshselwal/pavia-cpu-runner-k24) (`4..7`), [`geminicli/k24-s1`](https://www.kaggle.com/code/geminicli/pavia-cpu-runner-k24-s1) (`3`), [`geminicli/k24-s2`](https://www.kaggle.com/code/geminicli/pavia-cpu-runner-k24-s2) (`6`) | `19` (`6` backbones, `11/15` ens, `AlexNet` CREDIT & MultiHead) |
| [**Hisar-3**](examples/hisar_3mts_uncertainty_quantification/) | 🔄 GPU Q + CPU Running | `█████░░░░░░░░░░░░░░░` **25%** | `3 / 12` Done (+ Stg 3/6 partial) | [`daneshjangra/k13`](https://www.kaggle.com/code/daneshjangra/hisar3-suite-runner-k13) (GPU `QUEUED` `3..11`), [`daneshselwal/k17`](https://www.kaggle.com/code/daneshselwal/hisar3-cpu-runner-k17) (`4..7`), [`geminicli/k17-s1`](https://www.kaggle.com/code/geminicli/hisar3-cpu-runner-k17-s1) (`3`), [`daneshselwal1/k17-s2`](https://www.kaggle.com/code/daneshselwal1/hisar3-cpu-runner-k17-s2) (`6`) | `22` (`6` backbones, `12/15` ens, `4` AlexNet MultiHead) |
| [**Achla SAR**](examples/achla_multi_temporal_uncertainty_quantification/) | 🔄 GPU Q + CPU Running | `██░░░░░░░░░░░░░░░░░░` **8%** | `1 / 12` Done (`6/6` Stg 0, `8/15` ens) | [`daneshselwal/k22`](https://www.kaggle.com/code/daneshselwal/achla-suite-runner-k22) (GPU `QUEUED` `0..11`), [`daneshjangra/k27`](https://www.kaggle.com/code/daneshjangra/achla-cpu-runner-k27) (`0,1,2,7`), [`daneshselwal1/k27-s2`](https://www.kaggle.com/code/daneshselwal1/achla-cpu-runner-k27-s2) (`3`) | `23` (`6/6` backbones, `8/15` ens, `9` MultiHead) |
| [**ETM 108**](examples/multispectral_uncertainty_quantification/) | 🔄 GPU + CPU Running | `░░░░░░░░░░░░░░░░░░░░` **0%** | `0 / 12` Done (`5/6` Stg 0) | [`daneshselwal/k21`](https://www.kaggle.com/code/daneshselwal/etm108-suite-runner-k21) (GPU `RUNNING` `0..11`), [`daneshselwal/k26`](https://www.kaggle.com/code/daneshselwal/etm108-cpu-runner-k26) (`6..10`) | `5` (`AlexNet`, `GFNet`, `ViT_UNet_best`) |
| [**Planet Hisar CSV**](examples/planet_data_hisar_uncertainty_quantification/) | 🔄 CPU Running | `░░░░░░░░░░░░░░░░░░░░` **0%** | `0 / 12` Done (`4/6` Stg 0) | [`daneshjangra/k28`](https://www.kaggle.com/code/daneshjangra/planethisar-cpu-runner-k28) (`0,1,2`), [`daneshselwal1/k28-s2`](https://www.kaggle.com/code/daneshselwal1/planethisar-cpu-runner-k28-s2) (`6`) | `4` (`AlexNet`, `GFNet_best`, `ViT_UNet_best`) |

<details>
<summary>View 12-Stage × 11-Dataset Notebook Status Matrix (click to expand)</summary>

| Stage | Notebook | Indian Pines | Pavia Uni | DIAS 65B | 6-Band | Hisar-3 | Hissar-25 | Pavia Centre | 372-Band | ETM 108 | Achla SAR | Planet Hisar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | `baseline/model_training.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `5/6` (`k21` GPU) | ✅ `6/6` (`k22` / `k27`) | 🔄 `4/6` (`k28`) |
| **1** | `baseline/model_uncertainty_comparison.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k21` GPU | 🔄 `k22` / `k27` | 🔄 `k28` |
| **2** | `sacp/model_sacp_comparison.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k21` GPU | 🔄 `k22` / `k27` | 🔄 `k28` |
| **3** | `ensemble/model_training_ensembles.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `12/15` (`k13` Q / `k17-s1`) | ✅ Done | 🔄 `11/15` (`k19` Q / `k24-s1`) | ✅ Done | 🔄 `k21` GPU | 🔄 `8/15` (`k22` Q / `k27-s2`) | 📦 Staged |
| **4** | `credit/model_training_credit.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q / `k17` | 🔄 `k18` | 🔄 `1/3` (`k19` Q / `k24`) | ✅ Done | 🔄 `k21` GPU | 🔄 `k22` Q | 📦 Staged |
| **5** | `ensemble/model_uncertainty_credal_ensemble.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q / `k17` | 🔄 `k18` | 🔄 `k19` Q / `k24` | ✅ Done | 🔄 `k21` GPU | 🔄 `k22` Q | 📦 Staged |
| **6** | `multi_cp/model_training_multihead.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q / `k17` / `k17-s2` | ✅ Done | 🔄 `k19` Q / `k24` / `k24-s2` | ✅ Done | 🔄 `k21` GPU / `k26` | 🔄 `9` wts (`k22` Q) | 🔄 `k28-s2` |
| **7** | `multi_cp/model_uncertainty_multicp.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q / `k17` | ✅ Done | 🔄 `k19` Q / `k24` | ✅ Done | 🔄 `k21` GPU / `k26` | 🔄 `k22` Q / `k27` | 📦 Staged |
| **8** | `dapm/model_training_dapm_full.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q | ✅ Done | 🔄 `k19` Q | ✅ Done | 🔄 `k21` GPU / `k26` | 🔄 `k22` Q | 📦 Staged |
| **9** | `dapm/model_uncertainty_dapm_full.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q | ✅ Done | 🔄 `k19` Q | ✅ Done | 🔄 `k21` GPU / `k26` | 🔄 `k22` Q | 📦 Staged |
| **10** | `dapm/Model_uncertainty_dapm_cp.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q | ✅ Done | 🔄 `k19` Q | ✅ Done | 🔄 `k21` GPU / `k26` | 🔄 `k22` Q | 📦 Staged |
| **11** | `multicp_sacp/Model_uncertainty_multicp_sacp.ipynb` | ✅ Done | ✅ Done | ✅ Done | ✅ Done | 🔄 `k13` Q | 🔄 `ws3/5` (`k18`) | 🔄 `k19` Q | ✅ `ws3/5/7` | 🔄 `k21` GPU | 🔄 `k22` Q | 📦 Staged |

</details>

#### 🔬 6-Band Multispectral Extended Methods & Foundation Models

In addition to the 12 core uncertainty quantification stages, the **6-Band Multispectral** suite integrates advanced remote sensing foundation models and spatial representation learning:

| Extended Method | Directory | Description & Architecture | Kaggle Runner | Status |
|---|---|---|---|---|
| **DOFA** | [`examples/6_band_uncertainty_quantification/dofa/`](examples/6_band_uncertainty_quantification/dofa/) | Dynamic Optical-SAR Foundation Model feature extraction & classification | [`cuq6-dofa-k12`](https://www.kaggle.com/code/daneshselwal/cuq6-dofa-k12) / [`cuq6-dofa-uq-k13`](https://www.kaggle.com/code/daneshselwal/cuq6-dofa-uq-k13) | ✅ Complete |
| **GeoRSCLIP** | [`examples/6_band_uncertainty_quantification/georsclip/`](examples/6_band_uncertainty_quantification/georsclip/) | Vision-Language pre-trained remote sensing CLIP backbone & UQ | [`cuq6-georsclip-k14`](https://www.kaggle.com/code/daneshselwal/cuq6-georsclip-k14) / [`cuq6-georsclip-uq-k15`](https://www.kaggle.com/code/daneshselwal/cuq6-georsclip-uq-k15) | ✅ Complete |
| **Mamba-HSI** | [`examples/6_band_uncertainty_quantification/mambahsi/`](examples/6_band_uncertainty_quantification/mambahsi/) | State-Space Model (SSM) selective scanning for spectral-spatial tokens | [`cuq6-mambahsi-k10`](https://www.kaggle.com/code/daneshselwal/cuq6-mambahsi-k10) / [`cuq6-mambahsi-uq-k11`](https://www.kaggle.com/code/daneshselwal/cuq6-mambahsi-uq-k11) | ✅ Complete |
| **SpatialGCN** | [`examples/6_band_uncertainty_quantification/spatialgcn/`](examples/6_band_uncertainty_quantification/spatialgcn/) | Graph Convolutional Network modeling spatial adjacency & spectral topology | [`cuq6-spatialgcn-k16`](https://www.kaggle.com/code/daneshselwal/cuq6-spatialgcn-k16) / [`cuq6-spatialgcn-uq-k17`](https://www.kaggle.com/code/daneshselwal/cuq6-spatialgcn-uq-k17) | ✅ Complete |
| **Trials** | [`examples/6_band_uncertainty_quantification/trials/`](examples/6_band_uncertainty_quantification/trials/) | Hyperparameter tuning trials, architecture search, and calibration ablation logs | — | ✅ Documented |

### 📋 12-Stage Uncertainty Quantification Pipeline (Per Suite)

Baseline backbone weights (`AlexNet_CNN`, `GFNet`, `ViT_UNet`) trained by [`baseline/model_training.ipynb`](examples/indian_pines_uncertainty_quantification/baseline/model_training.ipynb) are hosted on Hugging Face Hub and automatically pulled at runner startup (or trained in Stage 0 and auto-uploaded on `k19`–`k23`), followed by the 11 downstream stages executed via Papermill (CPU runners skip Stage 3 and pull `models/ensembles/*.keras` from Hugging Face):

| Stage | Notebook Path (relative to suite dir) | Role & Method | Runner Scope |
|---|---|---|---|
| **0 (Pretrained)** | `baseline/model_training.ipynb` | Train & export `AlexNet_CNN`, `GFNet`, `ViT_UNet` backbones (hosted on HF) | GPU (`k19`–`k23`) / HF Pre-staged |
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
python scripts/verify_suites.py                  # Runs 449 structural, AST & regression checks across all 11 suites
python kaggle_staging/create_staged_runners.py   # Rebuilds all 18 Kaggle GPU/CPU runner notebooks (multi-account ready)
python kaggle_staging/verify_staging.py          # Runs 928 staging, packaging & token-hygiene checks
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
