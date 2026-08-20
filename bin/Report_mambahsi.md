# MambaHSI for Hyperspectral Image Classification and Uncertainty Quantification

## Abstract
This report details the evaluation of MambaHSI, a novel architecture based on State Space Models (SSMs), for Hyperspectral Image (HSI) classification and Uncertainty Quantification (UQ). By leveraging the dynamic processing capabilities of Mamba, the proposed models (Small, Base, Large) achieve near-perfect classification accuracy while maintaining excellent calibration. Conformal prediction techniques are applied to provide statistically guaranteed uncertainty intervals.

## 1. Introduction
Hyperspectral imaging captures a wide spectrum of light for each pixel, providing rich spectral and spatial information. MambaHSI addresses the computational and memory bottlenecks of transformer-based architectures by utilizing structured State Space Models (SSMs). This report evaluates MambaHSI at three capacity levels—Small, Base, and Large—analyzing their predictive performance and uncertainty metrics.

## 2. Methodology

### 2.1 MambaHSI Architecture
MambaHSI adapts the Mamba architecture for sequence modeling in hyperspectral domains. Unlike standard CNNs or Vision Transformers, Mamba relies on structured SSMs that compute continuous-time systems discretized for fast processing. The selective state space mechanism allows the model to filter irrelevant spectral bands while focusing on discriminative features.
- **MambaHSI_Small**: A lightweight variant with reduced state dimensions and layers.
- **MambaHSI_Base**: A balanced architecture offering optimal trade-offs.
- **MambaHSI_Large**: High capacity for maximizing performance and exploring subtle spectral-spatial correlations.

### 2.2 Conformal Prediction for Uncertainty Quantification
To ensure reliable decision-making, we employ conformal prediction methods:
- **Split Conformal**: A standard baseline providing marginal coverage.
- **Class Conditional Conformal**: Guarantees coverage across each class separately.
- **Clustered Conformal**: Groups related pixels/classes for localized coverage.
- **RAPS (Regularized Adaptive Prediction Sets)**: Encourages smaller prediction sets for better interpretability.
- **RC3P**: Robust class-conditional conformal prediction.

## 3. Experiments and Results

### 3.1 Classification Performance
The test accuracy and key metrics for the MambaHSI variants are presented below. All models achieve extraordinary accuracy (> 99%).

| Model | Test Accuracy | Macro F1 | Weighted F1 | ECE (15 bins) |
|-------|---------------|----------|-------------|---------------|
| MambaHSI_Base | 0.9984 | 0.9987 | 0.9984 | 0.0205 |
| MambaHSI_Large | 0.9981 | 0.9985 | 0.9981 | 0.0193 |
| MambaHSI_Small | 0.9974 | 0.9974 | 0.9974 | 0.0174 |

#### Model Comparison Metrics
![Model Comparison](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/training_plots/model_comparison_metrics.png)

#### Training Curves
![MambaHSI Small Training](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/training_plots/MambaHSI_Small_training_curves.png)
![MambaHSI Base Training](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/training_plots/MambaHSI_Base_training_curves.png)
![MambaHSI Large Training](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/training_plots/MambaHSI_Large_training_curves.png)

### 3.2 Visual Analysis

#### Scene Overview and Ground Truth
| Scene RGB | Ground Truth |
|-----------|--------------|
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/scene_visualizations/scene_rgb.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/scene_visualizations/ground_truth_label_map.png) |

#### Initial Classification Maps
| MambaHSI Small | MambaHSI Base | MambaHSI Large |
|----------------|---------------|----------------|
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/scene_visualizations/MambaHSI_Small_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/scene_visualizations/MambaHSI_Base_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/scene_visualizations/MambaHSI_Large_initial_classification.png) |

#### Combined Overview
![Combined Classification Overview](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/scene_visualizations/combined_initial_classification_overview.png)

### 3.3 Confusion Matrices
![Confusion Matrices](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/training_plots/confusion_matrices_side_by_side.png)

### 3.4 Uncertainty Quantification Results
Target Coverage is set to 0.95.

| Model | Method | Empirical Coverage | Avg Set Size |
|-------|--------|-------------------|--------------|
| MambaBase | ClassConditional | 0.9657 | 0.9657 |
| MambaBase | Clustered | 0.9647 | 0.9647 |
| MambaBase | RAPS | 0.9991 | 1.0000 |
| MambaBase | RC3P | 0.9680 | 0.9680 |
| MambaBase | SplitConformal | 0.9541 | 0.9541 |
| MambaLarge | ClassConditional | 0.9675 | 0.9675 |
| MambaLarge | SplitConformal | 0.9536 | 0.9536 |
| MambaSmall | ClassConditional | 0.9615 | 0.9615 |
| MambaSmall | SplitConformal | 0.9564 | 0.9564 |

#### Uncertainty Proxy Metrics
![Uncertainty Proxies](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/mambahsi/results/training_plots/uncertainty_proxy_metrics.png)

## 4. Conclusion
The MambaHSI architecture demonstrates extraordinary classification capabilities on hyperspectral imaging datasets. While all variants achieve near-perfect metrics, the careful application of Conformal Prediction provides meaningful uncertainty guarantees, reinforcing the model's reliability in practical domains.

