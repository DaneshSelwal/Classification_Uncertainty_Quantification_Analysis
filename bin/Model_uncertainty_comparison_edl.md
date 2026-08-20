# Conformal Prediction for Evidential Deep Learning

> **One-line description:** A pipeline applying post-hoc conformal prediction techniques (SplitCP, CcCP, RC3P, and RAPS) to the evidential output of the trained AlexNet, GFNet, and ViT models to construct statistically valid prediction sets.

---

## 1. Overview & Intuition

While the EDL models provide inherent uncertainty estimates via their Dirichlet strength $S$, they still require calibration to guarantee a specific coverage rate (e.g., 95%). This notebook applies standard conformal prediction techniques to the expected probabilities $\hat{\mathbf{p}} = \boldsymbol{\alpha} / S$ extracted from the EDL models. This provides a direct comparison of uncertainty quantification robustness across architectures using mathematically guaranteed prediction sets.

---

## 2. Methods Implemented

### 2.1 Split Conformal Prediction (SplitCP)
Constructs a single global threshold $\hat{q}$ across all classes by computing the $1-\alpha$ quantile of the nonconformity scores $1 - \hat{p}_{true}$ on the calibration set. Prediction sets include all classes with $\hat{p} \ge 1 - \hat{q}$.

### 2.2 Class-Conditional Conformal Prediction (CcCP)
Constructs class-specific thresholds $\hat{q}_c$ by computing quantiles independently for each class on the calibration set. This helps maintain validity across imbalanced classes.

### 2.3 Rank Calibrated Class-Conditional CP (RC3P)
Enhances CcCP by incorporating rank limits to reduce average prediction set sizes while maintaining conditional coverage guarantees.

### 2.4 Regularized Adaptive Prediction Sets (RAPS)
Sorts the predicted probabilities in descending order and greedily includes classes until the cumulative probability exceeds a threshold. A regularization penalty is applied to discourage excessively large prediction sets.

---

## 3. Implementation Details

1. **Model Loading:** The notebook strictly deserializes models using the registered `evidence_activation` function and other custom layers defined during training.
2. **Probability Extraction:** Model predictions ($\boldsymbol{\alpha}$ arrays) are converted to valid probability distributions via row-wise normalization.
3. **Conformal Searching:** Each method independently calculates its nonconformity scores, evaluates thresholds, and projects the sets onto the $H \times W$ spatial grid to generate intuitive uncertainty maps (where non-singleton sets denote uncertain regions).
4. **Outputs:** The pipeline generates visual maps, empirical coverage metrics, and average set sizes per method, saving the results systematically into Excel tables and `.png` plots for downstream comparison.
