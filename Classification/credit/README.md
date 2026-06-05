# CREDIT (Credal Ensemble Distillation): Theory & Implementation Summary

> **One-line description:** CREDIT distils a deep ensemble of classifiers into a single dual-head student that predicts class-wise probability *intervals* — a credal set — enabling simultaneous, inference-efficient quantification of aleatoric and epistemic uncertainty.

---

## 1. Overview & Intuition

### The Problem with Standard Neural Classifiers

A standard deep neural network outputs a single probability vector $\hat{p} \in \mathbb{R}^C$ (via softmax), treating the model's knowledge of every class as perfectly precise. In practice, the model is uncertain about this distribution in two fundamentally different ways: **aleatoric uncertainty** (AU) — irreducible randomness in the data itself, such as overlapping class signatures in ambiguous pixels — and **epistemic uncertainty** (EU) — reducible model uncertainty arising from limited or unrepresentative training data. A single softmax distribution conflates both, making them impossible to disentangle.

### Why Deep Ensembles Help (and Where They Fall Short)

Deep ensembles (DEs) address this by training $M$ independent networks from different random initialisations. Because each member reaches a distinct loss basin, the spread of their predictions reflects epistemic disagreement. Averaging the ensemble outputs recovers a well-calibrated point estimate; the inter-member variance signals EU. The critical limitation is cost: running $M$ networks at inference time multiplies latency and memory by $M$, which is prohibitive for large-scale or edge-deployed models.

Standard ensemble distillation compresses the $M$ models into a single student that approximates the ensemble *mean*, but in doing so, it collapses the epistemic spread back into a single distribution — losing the very information that made the ensemble valuable for uncertainty quantification.

### The Credal Ensemble Distillation (CED) Idea

CED, and its resulting student model CREDIT, solves this by making the student predict not a single probability vector but a **probability interval** $[\underline{p}_k, \overline{p}_k]$ for each class $k$. These intervals together define a **credal set** — a convex set of probability distributions consistent with the model's current state of knowledge. The interval *centre* encodes aleatoric probability structure (the most probable outcome) while the interval *width* $\Delta p_k = \overline{p}_k - \underline{p}_k$ directly encodes epistemic uncertainty: wide intervals indicate classes the model is confused about; narrow intervals indicate confident knowledge.

In this notebook, a simplified yet faithful variant of CED is implemented: the ensemble's **per-class minimum** prediction $p^*$ serves as the aleatoric proxy (the intersection probability) and the **per-class prediction spread** $\Delta p$ serves as the epistemic proxy (the interval length). A dual-head student is then trained to predict both quantities simultaneously via knowledge distillation.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X}$ be the input space (here: 9×9×6 multispectral patches) and $\mathcal{Y} = \{1, \ldots, C\}$ the set of $C$ land-cover classes. A deep ensemble consists of $M$ independently trained softmax classifiers $\{f^{(m)}\}_{m=1}^{M}$, each producing a probability vector

$$
\mathbf{p}^{(m)}(x) = \left[p_1^{(m)}, p_2^{(m)}, \ldots, p_C^{(m)}\right] \in \Delta^{C-1}
$$

where $\Delta^{C-1}$ denotes the probability simplex. The training set for the student is denoted $\mathcal{D}_\mathrm{train} = \{x_i\}_{i=1}^{N}$.

### 2.2 Credal Target Construction from the Ensemble

Given the $M$ teacher predictions for input $x$, the class-wise probability bounds are extracted:

$$
\overline{p}_k(x) = \max_{m \in \{1,\ldots,M\}} p_k^{(m)}(x), \qquad \underline{p}_k(x) = \min_{m \in \{1,\ldots,M\}} p_k^{(m)}(x)
$$

**Where:**
- $\overline{p}_k$ — the upper probability bound for class $k$ (best-case prediction across the ensemble)
- $\underline{p}_k$ — the lower probability bound for class $k$ (worst-case prediction across the ensemble)

**What this means:** The interval $[\underline{p}_k, \overline{p}_k]$ contains all class-$k$ probability values that at least one ensemble member would assign; it represents the range of *reasonable beliefs* about class $k$.

From these bounds, two distillation targets are derived.

**Aleatoric target — Normalised lower bound (intersection probability):**

$$
p^*_k(x) = \frac{\underline{p}_k(x)}{\sum_{j=1}^{C} \underline{p}_j(x) + \epsilon}
$$

**Where:**
- $p^*_k$ — the normalised per-class minimum, forming a valid probability vector on $\Delta^{C-1}$
- $\epsilon = 10^{-12}$ — numerical stability constant

**What this means:** $\mathbf{p}^*(x)$ is the point in the credal set that every ensemble member agrees is *at least this probable* for each class. It is the conservative, consensus-based probability estimate and serves as the aleatoric proxy: it is stable (low EU) when all members agree, and shifts with true data ambiguity (high AU).

**Epistemic target — Per-class interval length:**

$$
\Delta p_k(x) = \overline{p}_k(x) - \underline{p}_k(x)
$$

**What this means:** $\Delta \mathbf{p}(x)$ measures how much the ensemble members *disagree* about class $k$. Large disagreement signals high epistemic uncertainty — the model has not seen enough data to commit to a belief about this class.

### 2.3 Dual-Head Student Architecture

The CREDIT student is constructed from any single-head base network by replacing its output layer with two parallel heads attached to the penultimate feature tensor $\phi(x) \in \mathbb{R}^d$:

$$
\hat{\mathbf{p}}^*(x) = \mathrm{softmax}\!\left(W_\mathrm{AL} \cdot \phi(x) + b_\mathrm{AL}\right) \in \Delta^{C-1}
$$

$$
\widehat{\Delta \mathbf{p}}(x) = \sigma\!\left(W_\mathrm{EP} \cdot \phi(x) + b_\mathrm{EP}\right) \in [0,1]^C
$$

**Where:**
- $W_\mathrm{AL}, b_\mathrm{AL}$ — weights and biases of the aleatoric head (Dense → softmax)
- $W_\mathrm{EP}, b_\mathrm{EP}$ — weights and biases of the epistemic head (Dense → sigmoid)
- $\sigma$ — sigmoid activation, ensuring $\widehat{\Delta p}_k \in (0,1)$ as required for an interval length

### 2.4 Distillation Loss

The student is trained to match both targets jointly via a weighted composite loss:

$$
\mathcal{L} = \underbrace{\mathcal{L}_\mathrm{KL}\!\left(\mathbf{p}^*, \hat{\mathbf{p}}^*\right)}_{\text{aleatoric head}} + \lambda \cdot \underbrace{\mathcal{L}_\mathrm{MSE}\!\left(\Delta\mathbf{p}, \widehat{\Delta\mathbf{p}}\right)}_{\text{epistemic head}}
$$

**Where:**
- $\mathcal{L}_\mathrm{KL}(p, q) = \sum_k p_k \log\frac{p_k}{q_k}$ — KL divergence; appropriate because $\mathbf{p}^*$ is a probability vector that the student must reproduce faithfully
- $\mathcal{L}_\mathrm{MSE}(a, b) = \frac{1}{C}\sum_k (a_k - b_k)^2$ — mean squared error; appropriate because $\Delta\mathbf{p}$ is a regression target (interval width)
- $\lambda = 0.5$ — loss weight balancing the two heads

### 2.5 Uncertainty Decomposition at Inference

Once the student is trained, three scalar uncertainty estimates are produced for any test input $x$:

**Aleatoric Uncertainty (AU):** Shannon entropy of the aleatoric head output

$$
\mathrm{AU}(x) = -\sum_{k=1}^{C} \hat{p}^*_k(x) \log \hat{p}^*_k(x)
$$

**Epistemic Uncertainty (EU):** Mean predicted interval length across classes

$$
\mathrm{EU}(x) = \frac{1}{C} \sum_{k=1}^{C} \widehat{\Delta p}_k(x)
$$

**Total Uncertainty (TU):**

$$
\mathrm{TU}(x) = \mathrm{AU}(x) + \mathrm{EU}(x)
$$

**What this means:** AU is high when the aleatoric distribution is spread across many classes (genuine label ambiguity). EU is high when the predicted interval widths are large (the model disagrees with itself about this input). TU combines both sources.

---

## 3. Algorithm

**Input:** Ensemble of $M$ trained teacher models, training set $\mathcal{D}_\mathrm{train}$, test set $\mathcal{D}_\mathrm{test}$, base architecture builder, number of epochs $T$, loss weight $\lambda$

**Output:** Trained CREDIT student with AU and EU heads; per-pixel uncertainty maps

1. **Generate soft targets:** For each teacher $f^{(m)}$, run inference over $\mathcal{D}_\mathrm{train}$ to obtain $\mathbf{p}^{(m)}$. Compute $\underline{\mathbf{p}}$, $\overline{\mathbf{p}}$, then derive $\mathbf{p}^*$ (normalised lower bound) and $\Delta\mathbf{p}$ (interval widths).
2. **Repeat for test set:** Generate $\mathbf{p}^*_\mathrm{test}$ and $\Delta\mathbf{p}_\mathrm{test}$ for validation monitoring.
3. **Build CREDIT student:** Instantiate the base network, extract the penultimate feature layer, attach softmax (AU) and sigmoid (EU) heads.
4. **Compile:** Use KL divergence for the AU head and MSE for the EU head, weighted $1{:}0.5$.
5. **Train for $T$ epochs:** On $(x, (\mathbf{p}^*, \Delta\mathbf{p}))$ pairs with best-checkpoint saving on validation loss.
6. **Evaluate:** Load best weights; predict $(\hat{\mathbf{p}}^*, \widehat{\Delta\mathbf{p}})$; compute AU, EU, TU per test pixel; compute classification metrics from $\argmax \hat{\mathbf{p}}^*$.
7. **Spatial mapping:** Extract a patch for every pixel in the full scene; run student inference; threshold AU, EU, TU maps to produce certain/uncertain spatial masks.

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

    stacked = tf.stack(all_preds, axis=0)   # (M, N, C)
    p_min   = tf.reduce_min(stacked, axis=0)  # (N, C)
    p_max   = tf.reduce_max(stacked, axis=0)  # (N, C)

    delta_p_true = p_max - p_min
    p_star_true  = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
    return p_star_true, delta_p_true
```

**What this does:** Loads each teacher model sequentially (to avoid holding all $M$ models in GPU memory simultaneously), stacks their predictions along axis 0 to form a tensor of shape $(M, N, C)$, then computes element-wise minimum and maximum across the $M$ axis to obtain the lower and upper bounds per class. $\Delta\mathbf{p}$ is the raw spread; $\mathbf{p}^*$ is the normalised lower bound.

**Why:** Clearing the session after each teacher load keeps memory bounded regardless of ensemble size. The normalisation of $\mathbf{p}^*$ is essential to produce a valid probability distribution from the raw (unnormalised) per-class minima.

### 4.2 CREDIT Student Construction (`build_credit_student`)

```python
def build_credit_student(base_builder_func, num_classes):
    base_model = base_builder_func()
    features   = base_model.layers[-2].output   # penultimate feature tensor

    p_star  = layers.Dense(num_classes, activation='softmax',  name='p_star' )(features)
    delta_p = layers.Dense(num_classes, activation='sigmoid',  name='delta_p')(features)

    return tf.keras.Model(
        inputs=base_model.input,
        outputs=[p_star, delta_p],
        name='CREDIT_Student'
    )
```

**What this does:** Takes any pre-defined single-head architecture (AlexNet CNN, GFNet, or ViT-UNet), taps into its second-to-last layer (the rich feature representation before the original classification head), and branches two new linear heads off that shared representation.

**Why:** Using the penultimate layer means the two heads share all feature-extraction computation; the base network only needs to be run once per input, keeping inference cost identical to a standard single classifier. The softmax activation on `p_star` guarantees a valid probability simplex output; sigmoid on `delta_p` bounds each interval width to $(0, 1)$.

### 4.3 Compilation and Training Loop (`Cell 38`)

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

**Why:** KL divergence is the natural loss for training a softmax output to match a soft probability target; it penalises overconfidence in the wrong classes more strongly than cross-entropy would. MSE is appropriate for the interval-width regression task as it is bounded and smooth. The $0.5$ weight on the EU head reflects that the primary task is class prediction, with uncertainty calibration as a secondary objective.

### 4.4 Uncertainty Computation at Evaluation (`Cell 41`)

```python
p_star_pred, delta_p_pred = student.predict(x_te, batch_size=BATCH_SIZE)

au = -np.sum(p_star_pred * np.log(p_star_pred + 1e-12), axis=-1)
eu =  np.mean(delta_p_pred, axis=-1)
tu =  au + eu
```

**What this does:** Runs the CREDIT student on the test set, obtaining both heads simultaneously. Computes per-sample AU (Shannon entropy of $\hat{\mathbf{p}}^*$), EU (mean of $\widehat{\Delta\mathbf{p}}$), and TU (their sum).

**Why:** The $10^{-12}$ additive constant guards against $\log(0)$ for any class with near-zero predicted probability. The mean of $\widehat{\Delta\mathbf{p}}$ across classes gives a single scalar EU per sample that is directly comparable to AU on the same scale.

### 4.5 Full-Scene Spatial Uncertainty Mapping (`Cell 48`)

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

**Setup:** 3 classes, ensemble of $M = 3$ teachers, 1 test pixel.

**Step 1 — Collect teacher predictions:**

| Teacher | $p_1$ | $p_2$ | $p_3$ |
|---------|--------|--------|--------|
| $f^{(1)}$ | 0.70 | 0.20 | 0.10 |
| $f^{(2)}$ | 0.60 | 0.30 | 0.10 |
| $f^{(3)}$ | 0.65 | 0.10 | 0.25 |

**Step 2 — Compute per-class bounds:**

$$\underline{\mathbf{p}} = [0.60,\ 0.10,\ 0.10], \qquad \overline{\mathbf{p}} = [0.70,\ 0.30,\ 0.25]$$

$$\Delta\mathbf{p} = \overline{\mathbf{p}} - \underline{\mathbf{p}} = [0.10,\ 0.20,\ 0.15]$$

**Step 3 — Normalise lower bound to get $\mathbf{p}^*$:**

$$\sum_k \underline{p}_k = 0.60 + 0.10 + 0.10 = 0.80$$

$$\mathbf{p}^* = [0.60/0.80,\ 0.10/0.80,\ 0.10/0.80] = [0.75,\ 0.125,\ 0.125]$$

**Step 4 — Suppose the CREDIT student has learned these targets well and predicts $\hat{\mathbf{p}}^* \approx \mathbf{p}^*$ and $\widehat{\Delta\mathbf{p}} \approx \Delta\mathbf{p}$. Compute uncertainty:**

$$\mathrm{AU} = -(0.75 \ln 0.75 + 0.125 \ln 0.125 + 0.125 \ln 0.125) \approx 0.75 \times 0.288 + 2 \times 0.125 \times 2.079 \approx 0.736$$

$$\mathrm{EU} = \frac{0.10 + 0.20 + 0.15}{3} = 0.15$$

$$\mathrm{TU} = 0.736 + 0.15 = 0.886$$

**Interpretation:** The model is fairly confident in class 1 (the dominant class in $\mathbf{p}^*$), so AU is modest. The EU of 0.15 is moderate — the three teachers disagree most on class 2 (range 0.20), indicating some epistemic uncertainty about that class. With spatial thresholds AU > 0.5 and EU > 0.2, this pixel would be flagged as *aleatorically uncertain* (AU 0.736 > 0.5) but *epistemically borderline* (EU 0.15 < 0.2), suggesting that more training data for this pixel type would not substantially reduce uncertainty — the ambiguity is intrinsic to the data.

---

## 6. References

[1] K. Wang, F. Cuzzolin, D. Moens, and H. Hallez. "Credal Ensemble Distillation for Uncertainty Quantification." arXiv preprint, 2025. [arXiv:2511.13766](https://arxiv.org/abs/2511.13766)

[2] B. Lakshminarayanan, A. Pritzel, and C. Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." Advances in Neural Information Processing Systems (NeurIPS), 2017. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)

[3] K. Wang, F. Cuzzolin, K. Shariatmadar, D. Moens, and H. Hallez. "Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification." International Conference on Learning Representations (ICLR), 2025.

[4] G. Hinton, O. Vinyals, and J. Dean. "Distilling the Knowledge in a Neural Network." NeurIPS Deep Learning Workshop, 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
