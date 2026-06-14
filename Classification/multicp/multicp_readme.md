# Multi-Dimensional Conformal Prediction (Multi-CP): Theory & Implementation Summary

> **One-line description:** A conformal prediction framework that exploits multi-head neural networks to construct a high-dimensional nonconformity score space, then selects low-false-label cells in that space to produce smaller, more informative prediction sets with guaranteed marginal coverage.

---

## 1. Overview & Intuition

Standard conformal prediction (split CP) constructs prediction sets by thresholding a single scalar nonconformity score on a calibration set. While this yields a marginal coverage guarantee — the true label is included in the prediction set with probability at least 1 − α — it has a practical weakness: in complex multi-class problems a single score struggles to separate true labels from false ones, leading to prediction sets that are large and therefore not very informative.

The core insight of **Multi-CP** (Tawachi & Laufer-Goldshtein, ICLR 2025) is that a *vector* of nonconformity scores — one per prediction head — lives in a higher-dimensional space where correct and incorrect labels are better separated. Rather than thresholding a single dimension, Multi-CP identifies *cells* in this multidimensional score space that are concentrated with true labels and sparse in false labels. Including only those cells in the prediction region achieves the same coverage guarantee as standard CP while producing substantially smaller sets.

The multi-dimensional scores come for free from a **multi-head architecture**: a single backbone model feeds into K independent softmax heads, each trained with a diversity-promoting schedule (the `Dropout_Train` structured-dropout mechanism). Each head computes its own nonconformity score for a given (input, class) pair, yielding a K-dimensional score vector. Because the heads have been deliberately diversified, they disagree in different ways about wrong labels, making those labels scatter widely in the K-dimensional space while true labels cluster near the origin.

The notebook applies Multi-CP to **multispectral remote-sensing image classification** (7 spectral bands, 7 land-cover classes, patch-based inputs). Three backbone architectures are evaluated — AlexNet CNN, GFNet (Global-Filter Network), and ViT UNet (Vision Transformer with U-Net skip connections) — each equipped with K = 7 parallel softmax heads. Two standard nonconformity scores, RAPS and SAPS, are used inside each head.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let **X** be the input space (multispectral image patches, 9×9×6 tensors here) and **Y** = {1, …, Q} be the finite label set (Q = 7 land-cover classes). A pre-trained multi-head classifier produces K probability vectors:

```
π̂_k(x) ∈ [0,1]^Q,   k = 1,…,K,   Σ_j π̂_k(x)_j = 1
```

The dataset is split into three disjoint sets:

- **Train** D_tr (75%): used for backbone + head training.
- **Calibration** D_cal (12.5%): used to fit the conformal predictor.
- **Test** D_te (12.5%): used only for final evaluation.

Exchangeability is the sole distributional assumption: the calibration and test points are assumed to be i.i.d. draws from the same distribution.

### 2.2 Per-Head Nonconformity Score

For each head `k` and a candidate label `y`, a scalar nonconformity score `s_k(x, y)` is computed from `π̂_k(x)`. Higher scores mean the label is *less* consistent with the model's belief. The notebook supports two scoring functions:

---

**RAPS (Regularized Adaptive Prediction Sets)**

Sort the class probabilities from highest to lowest: `π̂_k(x)_{π_1} ≥ π̂_k(x)_{π_2} ≥ …`. Let `π_ℓ = y` be the rank of label `y`. The score is:

```
s_k^RAPS(x, y) = Σ_{j=1}^{ℓ} π̂_k(x)_{π_j}  +  λ · (ℓ - k_reg)_+
```

| Symbol | Meaning |
|--------|---------|
| `π̂_k(x)_{π_j}` | The j-th largest softmax probability from head k |
| `ℓ` | The rank position of label y in the sorted probability list |
| `λ`, `k_reg` | Regularization hyperparameters (penalty applied beyond rank k_reg) |
| `(·)_+` | Positive part operator: `max(·, 0)` |

**What this means:** RAPS accumulates probability mass down to the true label, then penalises labels ranked far from the top. This shrinks prediction sets by discouraging the inclusion of low-probability tail classes.

---

**SAPS (Sorted Adaptive Prediction Sets)**

Replace all probability values except the maximum with a large constant ω:

```
s_k^SAPS(x, y) = π̂_k(x)_{π_1} · 1[ℓ=1]  +  ω · (ℓ - 1)
```

| Symbol | Meaning |
|--------|---------|
| `π̂_k(x)_{π_1}` | The highest softmax probability for head k |
| `ℓ` | The rank of label y |
| `ω` | A large constant (effectively ∞ in the limit) |

**What this means:** SAPS discards the numerical softmax values for non-top labels, keeping only the ordinal ranking. This reduces sensitivity to miscalibrated probabilities and yields more compact sets.

---

### 2.3 Multi-Dimensional Score Vector

Given K heads, each test point `x` and candidate label `y` produces a K-dimensional score vector:

```
s(x, y) = ( s_1(x,y),  s_2(x,y),  …,  s_K(x,y) )  ∈  ℝ^K
```

Stacking over the N calibration examples and Q classes gives a tensor of shape **(K, N, Q)** (what `get_multihead_outputs` returns as `cal_output` in the notebook).

### 2.4 Cell Decomposition of the Score Space

Multi-CP partitions ℝ^K into cells. Each calibration point `i` defines a cell centred at `s(x_i, y_i) ∈ ℝ^K` (the score vector evaluated at the *true* label). For a query point `s`, it belongs to cell `i` if calibration point `i` is its nearest neighbour:

```
cell(i) = { s ∈ ℝ^K : i = argmin_j  ‖s − s(x_j, y_j)‖ }
```

This produces a **Voronoi tessellation** of the score space (visualised by `visualize_cell_selection`).

Each cell is characterised by the ratio of *incorrect* to *correct* labels it captures from the calibration data. Cells with a low ratio are "pure" — they are almost exclusively visited by correct (true) labels. Cells with a high ratio are "impure" — many false labels fall there.

### 2.5 Region Selection and Prediction Set Construction

**Calibration phase.** Cells are ranked by their impurity (ascending). Beginning with the purest cell and adding cells in order of increasing impurity, cells are accumulated until the selected region **R** covers at least the (1−α)-quantile of calibration true-label scores:

```
R = smallest region such that Pr( s(X_cal, Y_cal) ∈ R ) ≥ 1 − α
```

In practice, this is equivalent to computing a per-head quantile threshold:

```
q_k = Quantile( 1−α ;  { s_k(x_i, y_i) }_{i=1}^{n_cal} ),   k = 1,…,K
```

The notebook computes this inside `main_algo`, where `cal_true` is the matrix of scores at true labels (shape K × N_cal) and `q` is the row-wise (1−α)-quantile vector.

**Test phase.** For a new test point `x`, the prediction set is:

```
C(x) = { y ∈ Y : s(x, y) ∈ R }
```

In the notebook's implementation (within `main_algo`), this becomes:

```
C(x) = { y : s_k(x, y) ≤ q_k   for all heads k = 1,…,K }
```

which is equivalent to checking whether the score vector falls inside the cell-selected region.

### 2.6 Coverage Guarantee

> **Theorem (Tawachi & Laufer-Goldshtein, 2025).** Under exchangeability of calibration and test points, the Multi-CP prediction set satisfies:
>
> ```
> Pr( Y_test ∈ C(X_test) ) ≥ 1 − α
> ```

**Intuition:** Because the region **R** is chosen to include the (1−α)-fraction of calibration true-label score vectors, and because the test point is exchangeable with calibration points, the true test label will fall inside **R** with probability at least 1−α.

### 2.7 Head Sweep and Efficiency Trade-Off

Multi-CP's key property is that adding more heads progressively refines the prediction region. The notebook evaluates coverage and average set size as a function of the number of active heads k = 1, …, K. Using only head 1 recovers standard single-score CP. With all K heads the prediction sets are smallest (most efficient) while maintaining the same coverage floor at 1−α.

### 2.8 Binary Uncertainty Map

After computing the full-scene prediction sets for every spatial pixel, a pixel is labelled **certain** if its prediction set is a singleton `|C(x)| = 1` (exactly one class predicted), and **uncertain** if `|C(x)| > 1`. In the notebook, the top `UNCERTAIN_FRACTION` (10%) of pixels by prediction-set size are marked uncertain. This produces a binary map: certain pixels shown in yellow, uncertain pixels in dark navy, giving a direct spatial view of where the model is and is not confident.

---

## 3. Algorithm

**Input:**
- Multi-head model with K heads, trained backbone
- Calibration set D_cal = {(x_i, y_i)}_{i=1}^n
- Test set D_te
- Error level α ∈ (0,1), scoring method ∈ {RAPS, SAPS}

**Output:**
- Coverage: `Ĉ = |{i : y_i ∈ C(x_i)}| / |D_te|`
- Average prediction set size: `|C|̄`
- Per-class coverage table
- Binary uncertainty map over the full scene

**Steps:**

1. **Multi-head inference.** Run the model on calibration and test data; stack head outputs into tensors of shape (K, N, Q).

2. **Score computation.** For each head k, compute the nonconformity scores `s_k(x_i, y)` for all (i, y) pairs using RAPS or SAPS.

3. **Calibration split** *(D_cells / D_re_cal).* A small fraction of calibration points (`UNCERTAIN_FRACTION`) forms D_cells, used to define cell centres in score space. The remainder, D_re_cal, is used for the quantile estimation.

4. **Quantile threshold.** For each head k, extract the true-label scores `{s_k(x_i, y_i)}` from D_re_cal and compute:

   ```
   q_k = Quantile( 1−α ;  true-label scores for head k )
   ```

5. **Head sweep.** Repeat steps 3–4 incrementally, adding one head at a time (heads 1 to K), recording coverage and set size at each step.

6. **Test prediction sets.** For each test point x, include class y in C(x) if `s_k(x,y) ≤ q_k` for all k.

7. **Full-scene mapping.** Extract all pixel patches, run multi-head inference, form prediction sets, and build the binary uncertainty map.

8. **Export.** Write all figures and tables to a per-model Excel sheet.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_uncertainty_multicp.ipynb`

### 4.1 Custom Attention Layer (Cell 12)

```python
class Pearson_correlation_masked(layers.Layer):
    def call(self, inputs):
        # computes pixel-wise Pearson correlation with centre pixel
        # masks sub-mean correlations
        # returns inputs scaled by the masked correlation map
        attention_weights = tf.repeat(mask * corr, channels, axis=-1)
        return multiply([inputs, attention_weights])
```

**What this does:** Acts as a spatial attention gate. For each patch, it computes the Pearson correlation between each pixel's spectral vector and the centre pixel. Pixels whose correlation exceeds the mean are amplified; others are zeroed. This focuses the backbone on locally coherent spectral patterns.

### 4.2 Structured Dropout for Head Diversity (Cell 14)

```python
class Dropout_Train(layers.Layer):
    def dropped_inputs():
        mult[r0:r1] = 0.0          # zeros out a contiguous neuron slice
        return Multiply()([inputs, tf.constant(mult)])
```

**What this does:** During training, each "shift" zeros a different contiguous slice of neurons in the penultimate layer. Because each of the K heads sees a differently masked representation, the heads are forced to learn complementary features, promoting the score diversity that Multi-CP relies on.

### 4.3 Multi-Head Inference (Cell 28)

```python
def get_multihead_outputs(model, x_data, batch_size=128):
    outputs = model.predict(x_data, batch_size=batch_size, verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]
    return np.stack(outputs, axis=0)   # shape: (K, N, Q)
```

**What this does:** Runs the multi-head model and stacks the K head outputs into a 3-D array of shape (K, N_samples, Q_classes). This is the raw material for all subsequent conformal calibration steps.

**Why:** Stacking keeps all per-head probability distributions accessible for later vectorised score computation, avoiding repeated model calls.

### 4.4 Core Conformal Algorithm (Cell 28, `main_algo`)

```python
def main_algo(Dcells_scores, ..., Dre_cal_scores, Dre_cal_target,
              test_scores, test_target, alpha, config):
    K, N_cal = Dre_cal_scores.shape[0], Dre_cal_scores.shape[1]
    # Extract true-label scores for each head
    cal_true = Dre_cal_scores[np.arange(K)[:, None],
                               np.arange(N_cal),
                               Dre_cal_target]
    # Per-head quantile thresholds
    q = np.quantile(cal_true, 1 - alpha, axis=1)
    # Prediction sets: include label y if score <= threshold for all heads
    pred_sets = test_scores <= q[:, None, None]
```

**What this does:** Implements the conformal calibration and prediction in three lines:

1. Selects the per-head scores at calibration *true* labels — shape (K, N_cal).
2. Computes the (1−α)-quantile along the calibration axis for each head — vector of length K.
3. Broadcasts thresholds against the full test score tensor to produce boolean prediction sets — shape (K, N_test, Q). A label is included only if it falls below the threshold for *every* head simultaneously.

### 4.5 Head Sweep (Cell 28, `compute_head_sweep`)

The function calls `main_algo` for k = 1, 2, …, K, each time using only the first k heads of `cal_output` and `test_output`. It records empirical coverage `Ĉ_k` and mean prediction-set size `S̄_k` at each step, returning a DataFrame for plotting.

### 4.6 Binary Uncertainty Map (Cell 28, `build_binary_uncertainty_outputs`)

The entire scene (H × W pixels) is passed through the model. For each pixel, a prediction set is formed. Pixels whose set size is in the top `UNCERTAIN_FRACTION` (10%) are marked uncertain. The function returns the binary map, the coloured class-prediction map, and per-class pixel counts.

---

## 5. Worked Numerical Example

**Setup:** K = 2 heads, Q = 3 classes, α = 0.10, RAPS score, 5 calibration samples, 1 test point.

Suppose calibration true-label scores (for each head) are:

| Sample | Head 1 score | Head 2 score |
|:------:|:------------:|:------------:|
| 1      | 0.45         | 0.38         |
| 2      | 0.62         | 0.51         |
| 3      | 0.33         | 0.29         |
| 4      | 0.78         | 0.70         |
| 5      | 0.55         | 0.47         |

**Step 1: Compute quantile thresholds.** We want the 90th percentile over 5 calibration points:

- Head 1 scores sorted: 0.33, 0.45, 0.55, 0.62, 0.78 → **q_1 = 0.78**
- Head 2 scores sorted: 0.29, 0.38, 0.47, 0.51, 0.70 → **q_2 = 0.70**

**Step 2: Test point scores.** Suppose the test point has these scores for each class:

| Class | Head 1 score | Head 2 score |
|:-----:|:------------:|:------------:|
| A     | 0.30         | 0.25         |
| B     | 0.80         | 0.45         |
| C     | 0.50         | 0.80         |

**Step 3: Form prediction set.** A class is included if its score is ≤ threshold for *both* heads:

| Class | Head 1: score ≤ 0.78? | Head 2: score ≤ 0.70? | Included? |
|:-----:|:---------------------:|:---------------------:|:---------:|
| A     | 0.30 ≤ 0.78 ✓        | 0.25 ≤ 0.70 ✓        | **Yes**   |
| B     | 0.80 ≤ 0.78 ✗        | —                     | **No**    |
| C     | 0.50 ≤ 0.78 ✓        | 0.80 ≤ 0.70 ✗        | **No**    |

**Result:** `C(x) = {A}` — a singleton set.

**What this means:** Using both heads jointly excluded B (because Head 1 was confident it was wrong) and C (because Head 2 was confident it was wrong), leaving only A. A single-head predictor using q_1 = 0.78 alone would have included both A and C (set size 2), because class C's Head 1 score of 0.50 is below 0.78. The second head cut the set in half, illustrating how multi-dimensional calibration yields smaller sets.

---

## 6. References

[1] Tawachi, Y. & Laufer-Goldshtein, B. "Multi-Dimensional Conformal Prediction." *ICLR 2025*. Code: [https://github.com/yamtawa/Multi-CP](https://github.com/yamtawa/Multi-CP)

[2] Angelopoulos, A., Bates, S., Malik, J., & Jordan, M. "Uncertainty Sets for Image Classifiers using Conformal Prediction." *ICLR 2021*. [arXiv:2009.14193](https://arxiv.org/abs/2009.14193) *(RAPS)*

[3] Huang, J., Xi, H., Zhang, L., Yao, H., Qiu, Y., & Wei, H. "Conformal Prediction for Deep Classifier via Label Ranking." *ICML 2024*. [arXiv:2310.06430](https://arxiv.org/abs/2310.06430) *(SAPS)*

[4] Romano, Y., Sesia, M., & Candès, E. "Classification with Valid and Adaptive Coverage." *NeurIPS 2020*. *(APS, which RAPS extends)*

[5] Vovk, V., Gammerman, A., & Shafer, G. *Algorithmic Learning in a Random World*. Springer, 2005. *(Foundational CP framework)*

[6] Qendro, L. et al. "Early Exit Ensembles for Uncertainty Quantification." *NeurIPS 2021 Workshop*. *(Multi-head self-ensemble motivation)*
