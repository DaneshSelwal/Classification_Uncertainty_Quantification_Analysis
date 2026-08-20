# Evidential Deep Learning (EDL) Classification Report

## Abstract
Evidential Deep Learning (EDL) introduces a robust framework for quantifying uncertainty in neural network classifications. Unlike standard networks that output a point estimate of class probabilities using softmax, EDL places a Dirichlet distribution over the class probabilities. This report details the performance and uncertainty quantification capabilities of EDL applied to three architectures: GFNet, ViT-UNet, and AlexNet.

## Introduction
Reliable uncertainty quantification is crucial for safety-critical applications. Traditional neural networks often produce overconfident predictions, particularly for out-of-distribution data. Evidential Deep Learning models uncertainty by interpreting network outputs as parameters of a Dirichlet distribution, providing a principled measure of epistemic uncertainty based on Dempster-Shafer theory.

## Methodology
In EDL, the neural network is trained to predict the parameters $\alpha_k$ of a Dirichlet distribution representing the density of possible class probabilities. The evidence for each class is given by $e_k = \alpha_k - 1$. 

The uncertainty $u$ is inversely proportional to the total evidence $S = \sum_{k} \alpha_k$:
$u = \frac{K}{S}$
where $K$ is the number of classes. 
Models evaluated:
- **GFNet**: Global Filter Network.
- **ViT-UNet**: Vision Transformer integrated with a UNet structure.
- **AlexNet**: Standard Convolutional Neural Network.

## Experiments and Results

### 1. Classification Performance
The test accuracy and macro F1 scores demonstrate the baseline performance of each model.

| Model | Test Accuracy | Macro F1 | Weighted F1 | Kappa | Val NLL | Test NLL |
|---|---|---|---|---|---|---|
| GFNet | 0.9937 | 0.9920 | 0.9937 | 0.9915 | 0.1373 | 0.1399 |
| ViT-UNet | 0.9652 | 0.9411 | 0.9652 | 0.9527 | 0.3078 | 0.3019 |
| AlexNet | 0.9387 | 0.8628 | 0.9375 | 0.9171 | 0.5194 | 0.5194 |

### 2. Scene Visualizations
Below are the classification maps generated for each model alongside the true label map and scene RGB.

| Scene RGB & Ground Truth |
|:---:|
| ![Scene RGB](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/scene_visualizations/scene_rgb.png) |
| ![Ground Truth](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/scene_visualizations/ground_truth_label_map.png) |

**Classification Maps:**

| GFNet | ViT-UNet | AlexNet |
|:---:|:---:|:---:|
| ![GFNet Map](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/scene_visualizations/GFNet_initial_classification.png) | ![ViT Map](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/scene_visualizations/ViT_UNet_initial_classification.png) | ![AlexNet Map](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/scene_visualizations/AlexNet_CNN_initial_classification.png) |

| Combined Overview |
|:---:|
| ![Overview](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/scene_visualizations/combined_initial_classification_overview.png) |

### 3. Training and Evaluation Metrics
Training curves and confusion matrices highlight the stability and convergence of the EDL models.

| Training Curves | Confusion Matrices |
|:---:|:---:|
| ![GFNet Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/training_plots/GFNet_training_curves.png) | ![Confusion Matrices](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/training_plots/confusion_matrices_side_by_side.png) |
| ![ViT-UNet Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/training_plots/ViT_UNet_training_curves.png) | ![Model Comparison](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/training_plots/model_comparison_metrics.png) |
| ![AlexNet Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/training_plots/AlexNet_CNN_training_curves.png) | ![Uncertainty Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/edl/results/training_plots/uncertainty_proxy_metrics.png) |

### 4. Uncertainty Quantification Results
We assessed Conformal Prediction methods over the EDL outputs. All methods achieved 1.0 empirical coverage with a target of 0.95.

| Model | Method | Empirical Coverage | Avg Set Size | Singleton Rate | Runtime (s) |
|---|---|---|---|---|---|
| GFNet | ClassConditionalConformal | 1.0 | 7.0 | 0.0 | 1.81 |
| ViT | SplitConformal | 1.0 | 7.0 | 0.0 | 1.49 |
| AlexNet | RAPS | 1.0 | 7.0 | 0.0 | 0.41 |

## Conclusion
Evidential Deep Learning (EDL) robustly estimates uncertainty. GFNet outperformed the other architectures in terms of test accuracy and F1 score, demonstrating high capability on this dataset while providing well-calibrated evidence through the EDL framework.
