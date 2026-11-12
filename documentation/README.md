# Classification & Uncertainty Quantification — Stage-Wise Walkthrough Guide
This `documentation/` directory mirrors the `Classification/` folder structure and explains every notebook stage-by-stage in simple, accessible language—focusing on the uncertainty quantification algorithms, mathematical intuition, worked numerical examples, and key results from basic Split Conformal Prediction up to DOFA.
## Chronological Notebook Execution Order
| Stage | Folder | Notebook Summary (`.md`) | Core Algorithm / Concept |
| :--- | :--- | :--- | :--- |
| **Stage 1** | `baseline/` | `Model_training.md`, `Model_uncertainty_comparison.md` | Base CNN/GFNet/ViT-UNet training, MC Dropout, Split CP (THR), APS, RAPS |
| **Stage 1** | `credit/` | `Model_training_credit.md` | CREDIT (Credal Ensemble Distillation & upper/lower probability bounds) |
| **Stage 1** | `sacp/` | `Model_sacp_comparison.md` | SACP (Spatially-Aware Conformal Prediction across 3x3 to 9x9 windows) |
| **Stage 1** | `multicp/` | `Model_training_multihead.md`, `Model_uncertainty_multicp.md` | Multi-Head Conformal Prediction & class-conditional calibration |
| **Stage 2** | `multicp_sacp/` | `Model_uncertainty_multicp_sacp.md` | Hybrid MultiCP + SACP (multi-head + spatial window + class-conditional quantiles) |
| **Stage 3** | `ensemble/` | `Model_training_ensembles.md`, `Model_uncertainty_CreDE.md` | 5-member Deep Ensembles & CreDE (Total = Aleatoric + Epistemic Mutual Information) |
| **Stage 3** | `dapm/` | `Model_training_dapm_full.md`, `Model_uncertainty_dapm_full.md`, `Model_uncertainty_dapm_cp.md` | DAPM full-scene chunked uncertainty mapping & conformal calibration |
| **Stage 4A** | `trials/` | `focal_loss/`, `cb_focal_loss/`, `cdl/`, `conformal_reg/`, `edl/`, `edl_v2/` | Focal/CB-Focal Loss, Temperature Scaling (CDL), Conformal Reg, Evidential Deep Learning (EDL v1 & v2) |
| **Stage 4B** | `mambahsi/` | `Model_training_mambahsi.md`, `Model_uncertainty_comparison_mambahsi.md` | Selective State Space Model (MambaHSI Small/Base/Large) + Conformal Prediction |
| **Stage 4B** | `georsclip/` | `Model_training.md`, `Model_uncertainty_comparison.md` | Vision-Language Contrastive TextHead (`GeoRSCLIP`) + Conformal Prediction |
| **Stage 4B** | `spatialgcn/` | `Model_training.md`, `Model_uncertainty_comparison.md` | Spatial Graph Convolutional Network (`SpatialGCN`) + Conformal Prediction |
| **Stage 4B** | `dofa/` | `Model_training.md`, `Model_uncertainty_comparison.md` | Dynamic Omnimodal Foundation Architecture (`DOFA`) wavelength hypernetwork + Conformal Benchmark |
