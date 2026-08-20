# Conformal Prediction Uncertainty Quantification: MambaHSI

> **One-line description:** Applies distribution-free uncertainty quantification (Conformal Prediction) to the softmax outputs of the pre-trained MambaHSI models to generate statistically rigorous prediction sets for multispectral patch classification.

---

## 1. Overview & Intuition

Standard softmax probabilities are often poorly calibrated and overconfident. Conformal Prediction (CP) addresses this by converting point predictions into *prediction sets* that are mathematically guaranteed to contain the true class label with a user-specified probability (e.g., $1 - \alpha = 95\%$).

This notebook takes the frozen probability outputs from the trained MambaHSI models (Small, Base, Large) and calibrates them using five distinct conformal methods. Instead of forcing a single answer, the pipeline outputs sets of classes; large set sizes indicate high uncertainty (borderline or out-of-distribution pixels).

---

## 2. Mathematical Framework

Let $\hat{\pi}(x)$ be the probability vector produced by MambaHSI for input patch $x$. We aim to form a prediction set $\mathcal{C}(x) \subseteq \{1, \dots, K\}$ such that $\mathbb{P}(Y \in \mathcal{C}(X)) \geq 1 - \alpha$.

### 2.1 Split Conformal Prediction (SplitCP)
Uses a single scalar threshold. The non-conformity score is $s_i = 1 - \hat{\pi}_{y_i}(x_i)$. The threshold $\hat{q}$ is the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of the calibration scores. For a test point, $\mathcal{C}(x) = \{ k : \hat{\pi}_k(x) \geq 1 - \hat{q} \}$.

### 2.2 Class-Conditional Conformal Prediction (CcCP)
Computes a separate threshold $\hat{q}_c$ for each class $c$ using only calibration samples belonging to class $c$. This ensures valid marginal coverage *per class*, mitigating the issue where minority classes are under-covered.

### 2.3 Rank Calibrated Class-Conditional CP (RC3P)
An advanced variant of CcCP that limits the search space for class-conditional thresholds using the top-$k$ ranking of predicted probabilities. By truncating the score distributions based on optimal rank cutoffs, RC3P improves statistical efficiency and reduces the average set size while maintaining strict class-conditional coverage.

### 2.4 Regularized Adaptive Prediction Sets (RAPS)
Sorts the predicted probabilities in descending order and includes classes until the cumulative sum exceeds a threshold, applying a regularization penalty $k_{reg}$ and $\lambda$ to discourage overly large sets for highly uncertain samples.

### 2.5 Cluster-Conditioned Conformal Prediction (ClusterCP)
Clusters the calibration inputs in the feature/probability space using KMeans (e.g., $N=4$ clusters). Computes a threshold $\hat{q}_m$ for each cluster $m$. For a test sample, it determines its cluster and applies the corresponding threshold. It adapts the threshold based on regional data difficulty.

---

## 3. Algorithm

1. **Load Models:** Load `MambaHSI_Small`, `Base`, and `Large` using custom layers.
2. **Data Splitting:** Split the test dataset into equal calibration and evaluation pools.
3. **Probability Extraction:** Predict raw softmax probabilities for all pools and the full dense scene.
4. **Calibration & Evaluation:** Loop through all 5 conformal methods:
   - Calculate non-conformity scores and empirical quantiles $\hat{q}$ on the calibration pool.
   - Construct prediction sets on the evaluation pool.
   - Calculate metrics: empirical coverage, average set size, singleton rate.
5. **Dense Uncertainty Maps:** Slide the conformal thresholds across the full scene to yield spatial prediction sets. Generate maps highlighting regions of high uncertainty.
6. **Export:** Consolidate metrics, per-class coverage DataFrames, and spatial map plots into a unified Excel workbook.

---

## 4. Implementation Walkthrough

- **`SpectralScanLayer` / `SelectiveSSMBlock`**: Re-registered to allow `keras.models.load_model` to successfully unpack the custom Mamba architectures.
- **`conformal_qhat`**: Helper function that computes the precise empirical quantile `(n+1)(1-alpha)/n` using the `higher` interpolation method for rigorous validity.
- **Method Builders (e.g., `build_classconditional_outputs_for_model`)**: Each method is encapsulated in a function that returns a standardised dictionary containing summary metrics, plotting buffers, and tabular results.
- **Visualisations**: Generates comprehensive plots (Certain vs Uncertain Maps, Class Maps with Uncertain Masks) showing exactly where the model outputs non-singleton sets (i.e. is uncertain).
