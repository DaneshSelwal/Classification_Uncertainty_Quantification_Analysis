# Comprehensive Report: CDL Classification and Uncertainty Quantification

## Abstract
This report presents a comprehensive evaluation of classification performance and uncertainty quantification on the CDL dataset. Three deep learning architectures—AlexNet (CNN), GFNet, and ViT-UNet—were investigated. The models achieved exceptional accuracy, and various conformal prediction methods (Split Conformal, Class Conditional, RC3P, Clustered, and RAPS) were applied to quantify prediction uncertainty.

## 1. Introduction
The classification of the CDL dataset using advanced neural network architectures remains a critical task in remote sensing and precision agriculture. In this study, we benchmark the performance of CNNs (AlexNet), frequency-domain networks (GFNet), and vision transformers (ViT-UNet). Furthermore, we provide rigorous uncertainty quantification using Conformal Prediction, which provides distribution-free coverage guarantees.

## 2. Methodology and Theory

### 2.1. Model Architectures
- **AlexNet (CNN):** A classical convolutional neural network that extracts spatial features via hierarchical convolutions and max-pooling operations.
- **GFNet:** Global Filter Network replaces self-attention with operations in the frequency domain (via 2D FFT), providing global receptive fields while maintaining lower computational complexity than standard transformers.
- **ViT-UNet:** Integrates Vision Transformer blocks within a U-Net style encoder-decoder architecture, capturing both local spatial details and long-range dependencies through self-attention mechanisms.

### 2.2. Uncertainty Quantification via Conformal Prediction
We apply several conformal prediction methods to yield statistically valid prediction sets:
- **Split Conformal Prediction (SCP):** Uses a separate calibration set to compute non-conformity scores and define a threshold for a target coverage $(1 - \alpha)$.
- **Class-Conditional Conformal Prediction (CCCP):** Ensures coverage is satisfied independently for each class, critical for imbalanced datasets.
- **Regularized Adaptive Predictive Sets (RAPS):** Incorporates a regularization term to penalize overly large prediction sets.
- **RC3P & Clustered Conformal:** Advanced variants that adapt to local data density and cluster structures.

## 3. Experiments and Results

### 3.1. Classification Performance
The models demonstrated excellent classification capabilities. The table below summarizes test metrics:

| Model | Test Accuracy | Kappa | Macro F1 | Weighted F1 | Test NLL | Test Brier |
|-------|---------------|-------|----------|-------------|----------|------------|
| AlexNet | 0.9991 | 0.9987 | 0.9990 | 0.9991 | 0.0032 | 0.0018 |
| GFNet | 0.9979 | 0.9972 | 0.9976 | 0.9979 | 0.0112 | 0.0039 |
| ViT-UNet | 0.9954 | 0.9937 | 0.9940 | 0.9954 | 0.0218 | 0.0081 |

### 3.2. Training Dynamics
![Model Comparison Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/training_plots/model_comparison_metrics.png)

![Confusion Matrices](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/training_plots/confusion_matrices_side_by_side.png)

**AlexNet Training Curves**
![AlexNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/training_plots/AlexNet_CNN_training_curves.png)

**GFNet Training Curves**
![GFNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/training_plots/GFNet_training_curves.png)

**ViT-UNet Training Curves**
![ViT UNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/training_plots/ViT_UNet_training_curves.png)

### 3.3. Conformal Prediction Results
The table below summarizes the conformal prediction metrics at a target coverage of 95%:

| Model | Method | Empirical Coverage | Avg Set Size | Mean Per-Class Coverage |
|-------|--------|-------------------|--------------|-------------------------|
| AlexNet | Split Conformal | 0.9527 | 0.9527 | 0.9343 |
| AlexNet | ClassConditional | 0.9596 | 0.9596 | 0.9569 |
| GFNet | Split Conformal | 0.9466 | 0.9466 | 0.9213 |
| GFNet | ClassConditional | 0.9582 | 0.9582 | 0.9540 |
| ViT | Split Conformal | 0.9448 | 0.9452 | 0.9058 |
| ViT | ClassConditional | 0.9452 | 0.9476 | 0.9525 |

![Uncertainty Proxy Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/training_plots/uncertainty_proxy_metrics.png)

### 3.4. Scene Visualizations
The qualitative results for spatial classification map predictions are embedded below.

| Ground Truth | Scene RGB |
|--------------|-----------|
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/scene_visualizations/ground_truth_label_map.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/scene_visualizations/scene_rgb.png) |

| AlexNet Prediction | GFNet Prediction | ViT-UNet Prediction |
|--------------------|------------------|---------------------|
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/scene_visualizations/AlexNet_CNN_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/scene_visualizations/GFNet_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/scene_visualizations/ViT_UNet_initial_classification.png) |

![Combined Overview](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/cdl/results/scene_visualizations/combined_initial_classification_overview.png)

## 4. Conclusion
The findings demonstrate that while all models exhibit extremely high accuracy on the CDL dataset, AlexNet performs marginally better in terms of pure predictive performance. The integration of Conformal Prediction provides robust uncertainty bounds. Specifically, Class-Conditional methods successfully mitigated per-class coverage disparities compared to standard Split Conformal Prediction, which is vital for imbalanced remote sensing datasets.
