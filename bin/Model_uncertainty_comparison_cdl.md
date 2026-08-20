# Uncertainty Quantification: Conformal Prediction Evaluation

> **One-line description:** A systematic evaluation of Split, Class-Conditional, Rank-Calibrated, Regularized Adaptive, and Cluster-Conditional Conformal Prediction methods applied to CDL-trained classification models.

---

## 1. Overview & Intuition

After training models to output well-calibrated probabilities via CDL, this notebook quantifies their uncertainty using various post-hoc Conformal Prediction (CP) algorithms. CP transforms point predictions into **Prediction Sets**—sets of classes that are mathematically guaranteed to contain the true class with a user-specified probability (e.g., $1 - \alpha = 0.95$).

The notebook investigates five CP variants to handle the multispectral classification outputs, observing how different thresholding and regularisation techniques adapt to the models' class imbalances and varying degrees of difficulty. 

---

## 2. Mathematical Framework

Let $D_{cal}$ be a calibration set. The goal is to construct a prediction set $C(x)$ for a new instance $x$ such that $\mathbb{P}(y \in C(x)) \geq 1 - \alpha$. 

### 2.1 Split Conformal Prediction (SplitCP)
Uses the true-class probability as a heuristic.
1. **Scores:** $s_i = 1 - \hat{p}_{y_i}(x_i)$ for $i \in D_{cal}$.
2. **Threshold:** $\hat{q} = \text{Quantile}\left(s_1, \dots, s_n; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$.
3. **Prediction Set:** $C(x) = \{ k \in \mathcal{K} : 1 - \hat{p}_k(x) \leq \hat{q} \}$.

### 2.2 Class-Conditional CP (CcCP)
Ensures coverage is maintained separately for *each* class, preventing minority classes from being under-covered.
- **Threshold:** Computes a unique $\hat{q}_c$ using only calibration samples where the true class is $c$.

### 2.3 Rank Calibrated Class-Conditional CP (RC3P)
Improves efficiency by restricting sets based on truncated rank limits.
- **Algorithm:** Determines the minimum rank $k_c$ for each class $c$ where top-$k_c$ error falls below $\alpha$, and calculates specific thresholds within this truncated space to minimise the average prediction set size.

### 2.4 Regularized Adaptive Prediction Sets (RAPS)
In penalises the inclusion of too many classes to prevent unnecessarily large prediction sets.
- **Score:** Sort probabilities descending. $s_i = \sum_{j=1}^{k} p_{(j)} + \lambda \max(0, k - k_{reg})$, where $k$ is the index of the true class.
- **Parameters:** $\lambda = 0.01$ (penalty factor) and $k_{reg} = 1$ (allowed set size before penalty).

### 2.5 Cluster-Conditional CP
Groups similar multispectral patches into $K$ clusters (using K-Means on flattened patches) and applies independent conformal thresholds $\hat{q}_{cluster}$ for each.

---

## 3. Algorithm & Implementation

1. **Model Loading:** Safely deserializes `AlexNet_CNN`, `GFNet`, and `ViT_UNet` via custom objects and trusted path validation.
2. **Data Splitting:** Uses the holdout test set and evenly splits it into a Calibration set ($50\%$) and Evaluation set ($50\%$).
3. **Conformal Modules:** 
   - Evaluates each of the 5 CP methods.
   - Calculates empirical coverage, average/median set size, singleton rate, and empty set rate.
4. **Spatial Mapping:** Generates `(H, W)` scene inferences showing prediction maps, pixel-wise prediction set sizes (certain vs. uncertain regions), and class-specific uncertainty masks.
5. **Reporting:** Aggregates metrics to pandas DataFrames and persists them alongside PNG buffers to an exhaustive `conformal_reports_all_models.xlsx` workbook.
