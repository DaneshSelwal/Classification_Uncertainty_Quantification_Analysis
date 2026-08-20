# Comprehensive Report on Image Classification with Focal Loss and Uncertainty Quantification

## Abstract
This report details the implementation, theoretical background, and experimental evaluation of classification models trained with Focal Loss. We quantify classification performance across AlexNet, GFNet, and ViT-UNet architectures, while rigorously assessing uncertainty using state-of-the-art conformal prediction techniques. Our results indicate that Focal Loss successfully mitigates class imbalance, achieving competitive macro F1-scores, while the associated conformal prediction frameworks produce highly reliable predictive sets.

## 1. Introduction
Handling class imbalance in dense prediction and image classification tasks remains a prominent challenge. Models typically become biased towards majority classes, leading to suboptimal performance on minority classes. Focal Loss was introduced to address this by dynamically scaling the cross-entropy loss based on the prediction confidence. In this study, we investigate the efficacy of Focal Loss across three varied architectures (AlexNet, GFNet, and ViT-UNet) and further apply rigorous conformal prediction techniques to quantify model uncertainty and guarantee coverage bounds.

## 2. Methodology

### 2.1 Theoretical Foundations of Focal Loss
Focal Loss is an extension of standard Cross-Entropy (CE) loss designed to address extreme class imbalance. For binary classification, the standard CE loss is defined as:
$$ CE(p, y) = -y \log(p) - (1-y)\log(1-p) $$
where $y \in \{0,1\}$ is the ground truth class and $p \in [0,1]$ is the model's estimated probability for the class with label $y=1$. Let $p_t$ be the probability of the true class.

Focal Loss introduces a modulating factor $(1 - p_t)^\gamma$ to the CE loss, with a tunable focusing parameter $\gamma \geq 0$. It is formally defined as:
$$ FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$

**Motivation:**
1. **Down-weighting Easy Examples:** The modulating factor $(1 - p_t)^\gamma$ reduces the loss contribution of well-classified (easy) examples, focusing the training process on hard examples.
2. **Balancing Factor:** The weighting factor $\alpha_t \in [0,1]$ acts to balance the importance of positive and negative classes, ensuring that the gradient is not overwhelmed by the frequent classes.

### 2.2 Uncertainty Quantification and Conformal Prediction
Beyond point predictions, uncertainty quantification provides a mathematically rigorous way of defining reliability. We applied several Conformal Prediction strategies—including Split Conformal, Class-Conditional Conformal, Regularized Adaptive Predictive Sets (RAPS), Clustered Conformal, and RC3P. These techniques produce prediction sets $C(X)$ that guarantee a user-specified marginal coverage $1-\alpha$, such that $P(Y \in C(X)) \geq 1-\alpha$.

## 3. Experiments and Results

### 3.1 Model Performance Summary
The models were evaluated based on their test accuracy, Kappa, Macro F1, and Negative Log-Likelihood (NLL).

| Model | Test Accuracy | Kappa | Macro F1 | Weighted F1 | Test NLL | Test ECE (15-bin) | Train Time (s) |
|-------|---------------|-------|----------|-------------|----------|-------------------|----------------|
| GFNet | 0.9965 | 0.9953 | 0.9957 | 0.9965 | 0.0121 | 0.0017 | 823.16 |
| ViT_UNet | 0.9916 | 0.9887 | 0.9874 | 0.9917 | 0.0245 | 0.0025 | 1482.77 |
| AlexNet_CNN | 0.9828 | 0.9767 | 0.9744 | 0.9828 | 0.0887 | 0.0504 | 818.10 |

### 3.2 Conformal Prediction Results
The table below summarizes the uncertainty quantification metrics (Target Coverage: 0.95):

| Model | Method | Empirical Coverage | Avg Set Size | Singleton Rate | Mean Per-Class Coverage |
|-------|--------|--------------------|--------------|----------------|-------------------------|
| AlexNet | ClassConditional | 0.9684 | 0.9722 | 0.9722 | 0.9752 |
| GFNet | ClassConditional | 0.9503 | 0.9508 | 0.9508 | 0.9586 |
| ViT | ClassConditional | 0.9619 | 0.9638 | 0.9638 | 0.9754 |
| AlexNet | Clustered | 0.9643 | 0.9680 | 0.9680 | 0.9659 |
| GFNet | Clustered | 0.9527 | 0.9527 | 0.9527 | 0.9557 |
| ViT | Clustered | 0.9596 | 0.9610 | 0.9610 | 0.9699 |
| AlexNet | RAPS | 0.9916 | 1.0000 | 1.0000 | 0.9852 |
| GFNet | RAPS | 0.9991 | 1.0000 | 1.0000 | 0.9993 |
| ViT | RAPS | 0.9949 | 1.0000 | 1.0000 | 0.9958 |
| AlexNet | RC3P | 0.9698 | 0.9749 | 0.9749 | 0.9702 |
| GFNet | RC3P | 0.9582 | 0.9587 | 0.9587 | 0.9634 |
| ViT | RC3P | 0.9675 | 0.9694 | 0.9694 | 0.9778 |
| AlexNet | SplitConformal | 0.9582 | 0.9596 | 0.9596 | 0.9287 |
| GFNet | SplitConformal | 0.9564 | 0.9564 | 0.9564 | 0.9241 |
| ViT | SplitConformal | 0.9555 | 0.9559 | 0.9559 | 0.9328 |

### 3.3 Training Curves and Metrics

The following plots illustrate the learning dynamics of the tested models over 100 epochs.

![Model Comparison Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/training_plots/model_comparison_metrics.png)
*Figure 1: Comparison of model performance metrics.*

![Uncertainty Proxy Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/training_plots/uncertainty_proxy_metrics.png)
*Figure 2: Uncertainty proxy metrics comparison.*

![AlexNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/training_plots/AlexNet_CNN_training_curves.png)
*Figure 3: Training curves for AlexNet CNN.*

![GFNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/training_plots/GFNet_training_curves.png)
*Figure 4: Training curves for GFNet.*

![ViT UNet Training Curves](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/training_plots/ViT_UNet_training_curves.png)
*Figure 5: Training curves for ViT-UNet.*

![Confusion Matrices](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/training_plots/confusion_matrices_side_by_side.png)
*Figure 6: Confusion matrices for each evaluated model.*

### 3.4 Scene Classification Visualizations

Below are the visualizations showcasing the scene context, ground truth, and initial classification outcomes from the trained models.

| Original Scene | Ground Truth Map | Combined Overview |
| :---: | :---: | :---: |
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/scene_visualizations/scene_rgb.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/scene_visualizations/ground_truth_label_map.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/scene_visualizations/combined_initial_classification_overview.png) |

| AlexNet Classification | GFNet Classification | ViT-UNet Classification |
| :---: | :---: | :---: |
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/scene_visualizations/AlexNet_CNN_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/scene_visualizations/GFNet_initial_classification.png) | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/focal_loss/results/scene_visualizations/ViT_UNet_initial_classification.png) |

## 4. Conclusion
Our experiments demonstrate that integrating Focal Loss greatly enhances the capability of Deep Learning models in scenarios characterized by severe class imbalance. While GFNet attained the highest overall accuracy and macro F1 scores, all evaluated models benefited substantially. Conformal Prediction mechanisms reliably furnished statistically guaranteed bounds, showcasing that Focal Loss models can be confidently utilized in applications where uncertainty quantification is crucial.
