# CREDIT (Credal Ensemble Distillation): Theory & Implementation Summary

> **One-line description:** CREDIT distills a deep ensemble of classifiers into a single dual-head student that predicts class-wise probability *intervals* — a credal set — enabling simultaneous, inference-efficient quantification of aleatoric and epistemic uncertainty.

---

## 1. Overview & Intuition

### The Problem with Standard Neural Classifiers

A standard deep neural network outputs a single probability vector `p̂ ∈ ℝ^C` (via softmax), treating the model's knowledge of every class as perfectly precise. In practice, the model is uncertain about this distribution in two fundamentally different ways: **aleatoric uncertainty** (AU) — irreducible randomness in the data itself, such as overlapping class signatures in ambiguous pixels — and **epistemic uncertainty** (EU) — reducible model uncertainty arising from limited or unrepresentative training data. A single softmax distribution conflates both, making them impossible to disentangle.

### Why Deep Ensembles Help (and Where They Fall Short)

Deep ensembles (DEs) address this by training M independent networks from different random initializations. Because each member reaches a distinct loss basin, the spread of their predictions reflects epistemic disagreement. Averaging the ensemble outputs recovers a well-calibrated point estimate; the inter-member variance signals EU. The critical limitation is cost: running M networks at inference time multiplies latency and memory by M, which is prohibitive for large-scale or edge-deployed models.

Standard ensemble distillation compresses the M models into a single student that approximates the ensemble *mean*, but in doing so, it collapses the epistemic spread back into a single distribution — losing the very information that made the ensemble valuable for uncertainty quantification.

### The Credal Ensemble Distillation (CED) Idea

CED, and its resulting student model CREDIT, solves this by making the student predict not a single probability vector but a **probability interval** `[p̲_k, p̄_k]` for each class k. These intervals together define a **credal set** — a convex set of probability distributions consistent with the model's current state of knowledge. The interval *centre* encodes aleatoric probability structure (the most probable outcome) while the interval *width* `Δp_k = p̄_k − p̲_k` directly encodes epistemic uncertainty: wide intervals indicate classes the model is confused about; narrow intervals indicate confident knowledge.

In this notebook, a simplified yet faithful variant of CED is implemented: the ensemble's **per-class minimum** prediction `p*` serves as the aleatoric proxy (the intersection probability) and the **per-class prediction spread** `Δp` serves as the epistemic proxy (the interval length). A dual-head student is then trained to predict both quantities simultaneously via knowledge distillation.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let **X** be the input space (here: 9×9×6 multispectral patches) and **Y** = {1, …, C} the set of C land-cover classes. A deep ensemble consists of M independently trained softmax classifiers `{f^(m)}`, each producing a probability vector:

```
p^(m)(x) = [ p_1^(m),  p_2^(m),  …,  p_C^(m) ]  ∈  Δ^{C-1}
```

where `Δ^{C-1}` denotes the probability simplex. The training set for the student is `D_train = {x_i}_{i=1}^N`.

### 2.2 Credal Target Construction from the Ensemble

Given the M teacher predictions for input x, the class-wise probability bounds are extracted:

```
p̄_k(x) = max_{m ∈ {1,…,M}}  p_k^(m)(x)        (upper bound)
p̲_k(x) = min_{m ∈ {1,…,M}}  p_k^(m)(x)        (lower bound)
```

| Symbol | Meaning |
|--------|---------|
| `p̄_k` | Upper probability bound for class k (best-case prediction across the ensemble) |
| `p̲_k` | Lower probability bound for class k (worst-case prediction across the ensemble) |

**What this means:** The interval `[p̲_k, p̄_k]` contains all class-k probability values that at least one ensemble member would assign; it represents the range of *reasonable beliefs* about class k.

From these bounds, two distillation targets are derived.

---

**Aleatoric target — Normalized lower bound (intersection probability):**

```
p*_k(x) = p̲_k(x) / ( Σ_j p̲_j(x) + ε ),    ε = 1e-12
```

| Symbol | Meaning |
|--------|---------|
| `p*_k` | Normalized per-class minimum, forming a valid probability vector on `Δ^{C-1}` |
| `ε` | Numerical stability constant |

**What this means:** `p*(x)` is the point in the credal set that every ensemble member agrees is *at least this probable* for each class. It is the conservative, consensus-based probability estimate and serves as the aleatoric proxy: it is stable (low EU) when all members agree, and shifts with true data ambiguity (high AU).

---

**Epistemic target — Per-class interval length:**

```
Δp_k(x) = p̄_k(x) − p̲_k(x)
```

**What this means:** `Δp(x)` measures how much the ensemble members *disagree* about class k. Large disagreement signals high epistemic uncertainty — the model has not seen enough data to commit to a belief about this class.

### 2.3 Dual-Head Student Architecture

The CREDIT student is constructed from any single-head base network by replacing its output layer with two parallel heads attached to the penultimate feature tensor `φ(x) ∈ ℝ^d`:

```
p̂*(x)  = softmax( W_AL · φ(x) + b_AL )  ∈  Δ^{C-1}     [aleatoric head]
Δp̂(x)  = sigmoid( W_EP · φ(x) + b_EP )  ∈  (0,1)^C      [epistemic head]
```

| Symbol | Meaning |
|--------|---------|
| `W_AL`, `b_AL` | Weights and biases of the aleatoric head (Dense → softmax) |
| `W_EP`, `b_EP` | Weights and biases of the epistemic head (Dense → sigmoid) |
| `sigmoid` | Ensures each predicted interval width `Δp̂_k ∈ (0,1)` |

### 2.4 Distillation Loss

The student is trained to match both targets jointly via a weighted composite loss:

```
L = L_KL( p*, p̂* )  +  λ · L_MSE( Δp, Δp̂ )
    |_______________|     |___________________|
     aleatoric head          epistemic head
```

| Symbol | Meaning |
|--------|---------|
| `L_KL(p, q) = Σ_k p_k · log(p_k / q_k)` | KL divergence — appropriate because `p*` is a probability vector |
| `L_MSE(a, b) = (1/C) · Σ_k (a_k − b_k)²` | Mean squared error — appropriate because `Δp` is a regression target |
| `λ = 0.5` | Loss weight balancing the two heads |

### 2.5 Uncertainty Decomposition at Inference

Once the student is trained, three scalar uncertainty estimates are produced for any test input x:

**Aleatoric Uncertainty (AU):** Shannon entropy of the aleatoric head output

```
AU(x) = −Σ_{k=1}^{C}  p̂*_k(x) · log( p̂*_k(x) )
```

**Epistemic Uncertainty (EU):** Mean predicted interval length across classes

```
EU(x) = (1/C) · Σ_{k=1}^{C}  Δp̂_k(x)
```

**Total Uncertainty (TU):**

```
TU(x) = AU(x) + EU(x)
```

**What this means:** AU is high when the aleatoric distribution is spread across many classes (genuine label ambiguity). EU is high when the predicted interval widths are large (the model disagrees with itself about this input). TU combines both sources.

---

## 3. Algorithm

**Input:** Ensemble of M trained teacher models, training set D_train, test set D_test, base architecture builder, number of epochs T, loss weight λ

**Output:** Trained CREDIT student with AU and EU heads; per-pixel uncertainty maps

1. **Generate soft targets.** For each teacher `f^(m)`, run inference over D_train to obtain `p^(m)`. Compute `p̲`, `p̄`, then derive `p*` (normalized lower bound) and `Δp` (interval widths).
2. **Repeat for test set.** Generate `p*_test` and `Δp_test` for validation monitoring.
3. **Build CREDIT student.** Instantiate the base network, extract the penultimate feature layer, attach softmax (AU) and sigmoid (EU) heads.
4. **Compile.** Use KL divergence for the AU head and MSE for the EU head, weighted 1 : 0.5.
5. **Train for T epochs** on `(x, (p*, Δp))` pairs with best-checkpoint saving on validation loss.
6. **Evaluate.** Load best weights; predict `(p̂*, Δp̂)`; compute AU, EU, TU per test pixel; compute classification metrics from `argmax p̂*`.
7. **Spatial mapping.** Extract a patch for every pixel in the full scene; run student inference; threshold AU, EU, TU maps to produce certain/uncertain spatial masks.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training_credit.ipynb`

### 4.1 Soft Target Generation (`generate_credit_targets`)

```python
def generate_credit_targets(ensemble_paths, x_data, batch_size=128):
    all_preds = []
    for path in ensemble_paths:
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        all_preds.append(model.predict(x_data, batch_size=batch_size, verbose=1))
        del model
        tf.keras.backend.clear_session()

    stacked = tf.stack(all_preds, axis=0)             # (M, N, C)
    p_min   = tf.reduce_min(stacked, axis=0)          # (N, C)
    p_max   = tf.reduce_max(stacked, axis=0)          # (N, C)

    delta_p_true = p_max - p_min
    p_star_true  = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
    return p_star_true, delta_p_true
```

**What this does:** Loads each teacher model sequentially (to avoid holding all M models in GPU memory simultaneously), stacks their predictions along axis 0 to form a tensor of shape (M, N, C), then computes element-wise minimum and maximum across the M axis to obtain the lower and upper bounds per class. `Δp` is the raw spread; `p*` is the normalized lower bound.

**Why:** Clearing the session after each teacher load keeps memory bounded regardless of ensemble size. The normalization of `p*` is essential to produce a valid probability distribution from the raw (unnormalized) per-class minima.

### 4.2 CREDIT Student Construction (`build_credit_student`)

```python
def build_credit_student(base_builder_func, num_classes):
    base_model = base_builder_func()
    features   = base_model.layers[-2].output   # penultimate feature tensor

    p_star  = layers.Dense(num_classes, activation='softmax', name='p_star' )(features)
    delta_p = layers.Dense(num_classes, activation='sigmoid', name='delta_p')(features)

    return tf.keras.Model(
        inputs=base_model.input,
        outputs=[p_star, delta_p],
        name='CREDIT_Student'
    )
```

**What this does:** Takes any pre-defined single-head architecture (AlexNet CNN, GFNet, or ViT-UNet), taps into its second-to-last layer (the rich feature representation before the original classification head), and branches two new linear heads off that shared representation.

**Why:** Using the penultimate layer means the two heads share all feature-extraction computation; the base network only needs to be run once per input, keeping inference cost identical to a standard single classifier. The softmax activation on `p_star` guarantees a valid probability simplex output; sigmoid on `delta_p` bounds each interval width to (0, 1).

### 4.3 Compilation and Training Loop (Cell 38)

```python
student.compile(
    optimizer=optimizer,
    loss={
        'p_star':  tf.keras.losses.KLDivergence(),
        'delta_p': tf.keras.losses.MeanSquaredError(),
    },
    loss_weights={'p_star': 1.0, 'delta_p': 0.5},
)

student.fit(
    train_ds, validation_data=test_ds,
    epochs=EPOCHS, callbacks=callbacks, verbose=1
)
```

**What this does:** Compiles the dual-output student with two separate named loss functions matched by the output layer names. `ModelCheckpoint` saves the weights at the epoch of minimum validation loss.

**Why:** KL divergence is the natural loss for training a softmax output to match a soft probability target; it penalises overconfidence in the wrong classes more strongly than cross-entropy would. MSE is appropriate for the interval-width regression task as it is bounded and smooth. The 0.5 weight on the EU head reflects that the primary task is class prediction, with uncertainty calibration as a secondary objective.

### 4.4 Uncertainty Computation at Evaluation (Cell 41)

```python
p_star_pred, delta_p_pred = student.predict(x_te, batch_size=BATCH_SIZE)

au = -np.sum(p_star_pred * np.log(p_star_pred + 1e-12), axis=-1)
eu =  np.mean(delta_p_pred, axis=-1)
tu =  au + eu
```

**What this does:** Runs the CREDIT student on the test set, obtaining both heads simultaneously. Computes per-sample AU (Shannon entropy of `p̂*`), EU (mean of `Δp̂`), and TU (their sum).

**Why:** The `1e-12` additive constant guards against `log(0)` for any class with near-zero predicted probability. The mean of `Δp̂` across classes gives a single scalar EU per sample that is directly comparable to AU on the same scale.

### 4.5 Full-Scene Spatial Uncertainty Mapping (Cell 48)

```python
au_scene = -np.sum(p_star_scene * np.log(p_star_scene), axis=-1)
eu_scene =  np.mean(delta_p_scene, axis=-1)

au_mask = (au_scene.reshape(H, W) > au_thresh).astype(int)
eu_mask = (eu_scene.reshape(H, W) > eu_thresh).astype(int)
```

**What this does:** Extracts a 9×9×6 patch centred on every pixel of the full 330×307 scene, runs batch inference, and reshapes the resulting per-pixel AU and EU scores into the original spatial layout. Hard thresholds (AU > 0.5, EU > 0.2, TU > 0.7) produce binary certain/uncertain masks that are overlaid on the class prediction map for visual diagnosis.

**Why:** Spatially localised uncertainty maps are essential for remote sensing applications — they directly indicate which regions of the image the model should not be trusted on, guiding targeted field validation campaigns.

---

## 5. Worked Numerical Example

**Setup:** 3 classes, ensemble of M = 3 teachers, 1 test pixel.

**Step 1 — Collect teacher predictions:**

| Teacher | p_1 | p_2 | p_3 |
|:-------:|:---:|:---:|:---:|
| f^(1) | 0.70 | 0.20 | 0.10 |
| f^(2) | 0.60 | 0.30 | 0.10 |
| f^(3) | 0.65 | 0.10 | 0.25 |

**Step 2 — Compute per-class bounds:**

```
p̲ = [ 0.60,  0.10,  0.10 ]
p̄ = [ 0.70,  0.30,  0.25 ]

Δp = p̄ − p̲ = [ 0.10,  0.20,  0.15 ]
```

**Step 3 — Normalise lower bound to get `p*`:**

```
Σ_k p̲_k = 0.60 + 0.10 + 0.10 = 0.80

p* = [ 0.60/0.80,  0.10/0.80,  0.10/0.80 ]
   = [ 0.750,      0.125,      0.125      ]
```

**Step 4 — Compute uncertainty** (assuming the student has learned to predict `p̂* ≈ p*` and `Δp̂ ≈ Δp`):

```
AU = −( 0.750·ln(0.750) + 0.125·ln(0.125) + 0.125·ln(0.125) )
   = −( −0.2164 + −0.2599 + −0.2599 )
   ≈ 0.736

EU = ( 0.10 + 0.20 + 0.15 ) / 3
   = 0.45 / 3
   = 0.150

TU = AU + EU = 0.736 + 0.150 = 0.886
```

**Interpretation:** The model is fairly confident in class 1 (the dominant class in `p*`), so AU is modest. EU of 0.15 is moderate — the three teachers disagree most on class 2 (range 0.20), indicating some epistemic uncertainty about that class. With spatial thresholds AU > 0.5 and EU > 0.2, this pixel would be flagged as *aleatorically uncertain* (AU 0.736 > 0.5) but *epistemically borderline* (EU 0.15 < 0.2), suggesting that more training data for this pixel type would not substantially reduce uncertainty — the ambiguity is intrinsic to the data.

---

## 6. References

[1] K. Wang, F. Cuzzolin, D. Moens, and H. Hallez. "Credal Ensemble Distillation for Uncertainty Quantification." arXiv preprint, 2025. [arXiv:2511.13766](https://arxiv.org/abs/2511.13766)

[2] B. Lakshminarayanan, A. Pritzel, and C. Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)

[3] K. Wang, F. Cuzzolin, K. Shariatmadar, D. Moens, and H. Hallez. "Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification." *International Conference on Learning Representations (ICLR)*, 2025.

[4] G. Hinton, O. Vinyals, and J. Dean. "Distilling the Knowledge in a Neural Network." *NeurIPS Deep Learning Workshop*, 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
