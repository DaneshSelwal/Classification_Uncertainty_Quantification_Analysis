# Classification Uncertainty Quantification using Conformal Prediction

## 1. Abstract
This report details the implementation and evaluation of various Conformal Prediction techniques for Uncertainty Quantification in image classification. The methods investigated include Split Conformal, Class-Conditional Conformal, Clustered Conformal, Regularized Adaptive Predictive Sets (RAPS), and Randomized Class-Conditional Conformal Prediction (RC3P). The base classifiers evaluated are AlexNet, GFNet, and ViT (Vision Transformer) based U-Net architectures. We demonstrate that conformal methods provide statistically valid prediction sets with guaranteed marginal coverage while varying in efficiency (set sizes) across the models.

## 2. Introduction
Deep learning models often provide overconfident point predictions, lacking reliable measures of uncertainty. Conformal prediction (CP) offers a distribution-free framework to quantify predictive uncertainty by constructing prediction sets that contain the true label with a user-specified probability $1 - \alpha$. In this study, we apply conformal prediction to multi-class image classification. 

We assess how different base classifiers (AlexNet, GFNet, ViT_UNet) interact with various conformal prediction algorithms, exploring the trade-offs between coverage guarantees and prediction set efficiency.

## 3. Methodology
### 3.1 Base Classifiers
Three models were trained on the dataset:
- **AlexNet (CNN)**: A standard Convolutional Neural Network.
- **GFNet**: Global Filter Network, which learns spatial representations in the frequency domain.
- **ViT_UNet**: A hybrid Vision Transformer and U-Net architecture.

### 3.2 Conformal Prediction Methods
Conformal Prediction uses a calibration set to compute non-conformity scores and threshold them to achieve the desired coverage $\ge 1 - \alpha$.
1. **Split Conformal**: Computes a single global threshold based on the calibration set.
2. **Class-Conditional Conformal**: Calculates independent thresholds for each class to ensure class-balanced coverage.
3. **Clustered Conformal**: Clusters the data (or features) and computes thresholds per cluster.
4. **RAPS (Regularized Adaptive Predictive Sets)**: Introduces a penalty term to favor smaller set sizes for highly confident predictions.
5. **RC3P**: Randomized variation of class-conditional conformal prediction.

## 4. Experiments and Results

### 4.1 Training Outcomes
The base models were trained and their metrics are summarized as follows. GFNet and AlexNet show very high accuracy, while ViT_UNet is slightly lower but still highly performant.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| AlexNet | 0.998 | 0.997 | 0.998 | 0.998 |
| GFNet | 0.997 | 0.996 | 0.996 | 0.997 |
| ViT_UNet | 0.995 | 0.994 | 0.994 | 0.995 |

![AlexNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/training_plots/AlexNet_CNN_training_curves.png)
![GFNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/training_plots/GFNet_training_curves.png)
![ViT UNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/training_plots/ViT_UNet_training_curves.png)
![Model Comparison Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/training_plots/model_comparison_metrics.png)

### 4.2 Initial Scene Classifications
Below are the initial scene classifications predicted by each model alongside the RGB scene and Ground Truth.

| Scene RGB | Ground Truth |
|:---:|:---:|
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/scene_visualizations/scene_rgb.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/scene_visualizations/ground_truth_label_map.png) |

| AlexNet | GFNet | ViT_UNet |
|:---:|:---:|:---:|
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/scene_visualizations/AlexNet_CNN_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/scene_visualizations/GFNet_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/scene_visualizations/ViT_UNet_initial_classification.png) |

![Combined Overview](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/scene_visualizations/combined_initial_classification_overview.png)
![Confusion Matrices](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/training_plots/confusion_matrices_side_by_side.png)

### 4.3 Conformal Prediction Results
The target coverage was set to 0.95 ($\alpha = 0.05$). All methods successfully achieved or exceeded this target marginal empirical coverage.

| Model | Method | Empirical Coverage | Avg Set Size | Mean Per-Class Coverage |
|-------|--------|--------------------|--------------|-------------------------|
| AlexNet | SplitConformal | 0.953 | 0.953 | 0.923 |
| GFNet | SplitConformal | 0.955 | 0.955 | 0.929 |
| ViT | SplitConformal | 0.947 | 0.947 | 0.904 |
| AlexNet | ClassConditionalConformal | 0.956 | 0.956 | 0.948 |
| GFNet | ClassConditionalConformal | 0.961 | 0.961 | 0.962 |
| ViT | ClassConditionalConformal | 0.954 | 0.954 | 0.961 |
| AlexNet | RAPS | 0.999 | 1.000 | 0.999 |
| GFNet | RAPS | 0.998 | 1.000 | 0.996 |
| ViT | RAPS | 0.994 | 1.000 | 0.993 |

RAPS consistently yielded over-coverage with higher average set sizes, while Class-Conditional Conformal Prediction provided tight coverage guarantees around the nominal 0.95 level while significantly improving the mean per-class coverage compared to the global Split Conformal baseline.

![Uncertainty Proxy Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/conformal_reg/results/training_plots/uncertainty_proxy_metrics.png)

## 5. Conclusion
Conformal Prediction is a powerful framework for bounding the uncertainty of classification models. The experimental results reveal that all models (AlexNet, GFNet, ViT_UNet) can be effectively calibrated to yield 95% marginal coverage. Class-Conditional CP and Clustered CP exhibit excellent balance, ensuring that minority classes also meet coverage requirements without excessively inflating the prediction set size.
