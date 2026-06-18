# CREDIT: Theory & Implementation Summary

> **One-line description:** CREDIT distils a 5-model deep ensemble into a single dual-head student network that predicts, for every input, a class-wise probability interval (a credal set) instead of a single softmax vector, letting one forward pass recover both aleatoric and epistemic uncertainty that previously required running the whole ensemble.

---

## 1. Overview & Intuition

A deep ensemble of independently trained networks is one of the most reliable ways to separate "the data is ambiguous" (aleatoric uncertainty) from "the model hasn't seen enough like this" (epistemic uncertainty): each member proposes its own softmax vector, and the *agreement* or *disagreement* across members carries the epistemic signal, while the *spread* within any one member's softmax carries the aleatoric signal. The catch is cost — at inference time you must run every ensemble member, store every set of weights, and aggregate the results, which is exactly the overhead practitioners are trying to avoid when they reach for a single deployable model in the first place.

CREDIT (Credal Ensemble Distillation, CED) addresses this by training a single "student" network that mimics the *interval* spanned by the ensemble's predictions rather than a single point estimate. For every class, the student predicts a lower bound and an upper bound on the probability the ensemble assigns that class. Together, these bounds describe a credal set: a convex region of probability distributions that the true predictive distribution is believed to lie within. This is a strictly richer object than a single softmax vector, because its *width* per class encodes how much the ensemble members disagreed (epistemic uncertainty), while the *shape* of the lower-bound vector itself behaves like a calibrated softmax and encodes the data's inherent ambiguity (aleatoric uncertainty).

The key architectural trick is that the student does not need a second forward pass or a second network: it reuses the backbone's penultimate feature representation and attaches two small output heads on top — one softmax head for the lower-probability vector and one sigmoid head for the per-class interval width. Both heads are trained simultaneously by regression against teacher-derived targets, so the whole apparatus collapses to ordinary supervised training once the targets have been computed offline from the ensemble.

What makes this attractive relative to simply running the ensemble is that, after training, only the single student network is needed at inference time. What makes it attractive relative to ad-hoc single-model uncertainty heuristics (e.g. softmax entropy alone) is that the interval width is trained explicitly against ensemble disagreement, so it inherits some of the ensemble's epistemic sensitivity rather than relying solely on softmax flatness, which conflates aleatoric and epistemic effects.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let the input space be $\mathcal{X}$ (here, $9\times 9\times 6$ multispectral image patches) and the label space $\mathcal{Y} = \{1, \dots, C\}$ with $C$ classes. A deep ensemble of $M$ independently trained teacher networks $\{f_{\theta_1}, \dots, f_{\theta_M}\}$ is assumed already trained; each teacher outputs a softmax vector $\pi_m(x) \in \Delta^{C-1}$ (the probability simplex) for input $x$. The notebook uses $M = 5$ teachers per architecture.

The goal is to train a single student network $g_\phi$ with two output heads,
$$
g_\phi(x) = \big(\hat p^{\star}(x), \widehat{\Delta p}(x)\big), \qquad \hat p^{\star}(x) \in \Delta^{C-1}, \;\; \widehat{\Delta p}(x) \in [0,1]^C,
$$
such that $\hat p^{\star}(x)$ approximates a class-wise *lower* probability bound derived from the ensemble, and $\widehat{\Delta p}(x)$ approximates the *width* of the per-class probability interval. Together they define an *upper* bound $\hat p^{\star}(x) + \widehat{\Delta p}(x)$ (informally), and the pair $(\hat p^{\star}, \widehat{\Delta p})$ characterises a credal set: the convex hull of all probability vectors that lie within these per-class bounds.

### 2.2 Teacher-Derived Training Targets

For a given input $x$, stack the $M$ teacher softmax vectors into a $M \times C$ matrix and take the per-class minimum and maximum across the ensemble:

**Equation:**
$$
p_{\min,c}(x) = \min_{m=1,\dots,M} \pi_m(x)_c, \qquad p_{\max,c}(x) = \max_{m=1,\dots,M} \pi_m(x)_c, \qquad c = 1,\dots,C.
$$

**Where:**
- $\pi_m(x)_c$ — the probability teacher $m$ assigns to class $c$ for input $x$.
- $p_{\min,c}(x)$ — the smallest probability any teacher assigned to class $c$; the raw lower bound.
- $p_{\max,c}(x)$ — the largest probability any teacher assigned to class $c$; the raw upper bound.

**What this means:** for each class independently, the ensemble's disagreement is captured by how far apart its members' opinions are; classes where every teacher agrees will have $p_{\min,c} \approx p_{\max,c}$, while classes the ensemble is unsure about will show a wide gap.

The raw interval width is then defined and the lower bound is renormalised into a valid probability vector to serve as the aleatoric-style training target:

**Equation:**
$$
\Delta p_{\text{true},c}(x) = p_{\max,c}(x) - p_{\min,c}(x), \qquad p^{\star}_{\text{true},c}(x) = \frac{p_{\min,c}(x)}{\sum_{j=1}^{C} p_{\min,j}(x) + \varepsilon},
$$
with a small constant $\varepsilon$ (the notebook uses $10^{-12}$) added purely to avoid division by zero.

**Where:**
- $\Delta p_{\text{true},c}(x)$ — the epistemic-proxy training target for class $c$: how wide the ensemble's interval is for that class.
- $p^{\star}_{\text{true},c}(x)$ — the aleatoric-proxy training target for class $c$: the renormalised "worst-case" probability lower bound, rescaled to sum to 1 across classes so it behaves like a proper categorical distribution.

**What this means:** $p^{\star}_{\text{true}}$ is not simply the ensemble's mean softmax; it is built only from the most conservative (lowest) per-class vote across teachers, then rescaled. It represents the most cautious belief the ensemble is willing to commit to, while $\Delta p_{\text{true}}$ is a separate vector tracking exactly how much headroom exists above that cautious belief for each class.

### 2.3 Student Architecture and Loss

The student reuses an existing single-head backbone $h_\phi$ (any of AlexNet-style CNN, GFNet, or ViT in the notebook) up to its penultimate feature layer, and attaches two new dense heads on top of those features:

**Equation:**
$$
\hat p^{\star}(x) = \mathrm{softmax}\big(W_1 \, z(x) + b_1\big), \qquad \widehat{\Delta p}(x) = \sigma\big(W_2 \, z(x) + b_2\big),
$$
where $z(x)$ is the shared penultimate feature vector and $\sigma$ is the elementwise sigmoid.

**Where:**
- $z(x)$ — the feature representation produced by the backbone immediately before its original single output layer.
- $W_1, b_1$ — the softmax head's weights, producing the aleatoric-proxy output $\hat p^{\star}(x)$.
- $W_2, b_2$ — the sigmoid head's weights, producing the epistemic-proxy output $\widehat{\Delta p}(x)$; sigmoid (rather than softmax) is used here because interval widths are independent per class and need not sum to 1.

**What this means:** the backbone does the same feature extraction it always did; only the final layer is duplicated and repurposed, so the student is barely more expensive than the original single-head model.

Training minimises a weighted sum of two losses, one per head:

**Equation:**
$$
\mathcal{L}(\phi) = \mathrm{KL}\big(p^{\star}_{\text{true}} \,\Vert\, \hat p^{\star}_\phi\big) \;+\; \lambda \cdot \mathrm{MSE}\big(\Delta p_{\text{true}}, \widehat{\Delta p}_\phi\big),
$$
with $\lambda = 0.5$ in the notebook's configuration.

**Where:**
- $\mathrm{KL}(\cdot\Vert\cdot)$ — the Kullback–Leibler divergence between the true and predicted aleatoric-proxy distributions, the natural loss for matching one categorical distribution to another.
- $\mathrm{MSE}(\cdot,\cdot)$ — mean squared error between the true and predicted per-class interval widths, an ordinary regression loss since $\Delta p$ is not a probability distribution.
- $\lambda$ — a fixed scalar balancing how much the epistemic-width regression loss contributes relative to the aleatoric KL loss.

**What this means:** the two heads are trained jointly but with losses suited to what each head represents — a distributional match for the belief vector, and a plain numeric match for the uncertainty widths.

### 2.4 Inference-Time Uncertainty Decomposition

Once trained, the student's two outputs are converted into scalar uncertainty quantities for any new input $x$:

**Equation:**
$$
\mathrm{AU}(x) = -\sum_{c=1}^{C} \hat p^{\star}_c(x) \,\log\!\big(\hat p^{\star}_c(x) + \varepsilon\big), \qquad \mathrm{EU}(x) = \frac{1}{C}\sum_{c=1}^{C} \widehat{\Delta p}_c(x), \qquad \mathrm{TU}(x) = \mathrm{AU}(x) + \mathrm{EU}(x).
$$

**Where:**
- $\mathrm{AU}(x)$ — aleatoric uncertainty, the Shannon entropy of the predicted aleatoric-proxy distribution; high when the belief vector is spread across many classes.
- $\mathrm{EU}(x)$ — epistemic uncertainty, the mean predicted interval width across classes; high when the student (mimicking ensemble disagreement) is unsure how wide the credal set should be.
- $\mathrm{TU}(x)$ — total uncertainty, simply the sum of the two components.

**What this means:** a single forward pass through the student now yields both an entropy-style aleatoric score and a disagreement-style epistemic score, the same decomposition a full ensemble would have required five forward passes (one per teacher) to estimate directly.

---

## 3. Algorithm

**Input:** a pre-trained ensemble of $M$ teacher networks per architecture; a backbone architecture to convert into a student; training and test data splits; the loss weight $\lambda$.
**Output:** a trained dual-head CREDIT student, plus its per-sample $(\mathrm{AU}, \mathrm{EU}, \mathrm{TU})$ uncertainty estimates on held-out data.

1. For the chosen architecture, locate all $M$ saved teacher checkpoints from the previously trained ensemble.
2. Run every teacher over the training inputs (and separately over the test inputs) to collect their softmax vectors.
3. Stack the per-input softmax vectors across teachers and compute the per-class minimum and maximum, then derive $p^{\star}_{\text{true}}$ (renormalised minimum) and $\Delta p_{\text{true}}$ (max − min) as in §2.2.
4. Build a `tf.data` pipeline pairing each input with its two derived targets.
5. Construct the dual-head student by reusing the backbone's penultimate features and attaching a softmax head and a sigmoid head, as in §2.3.
6. Compile the student with the KL-divergence loss on the softmax head, the MSE loss on the sigmoid head, weighted by $\lambda$, and an architecture-specific optimiser.
7. Train for a fixed number of epochs, checkpointing the weights that minimise validation loss.
8. Reload the best checkpoint and run inference on held-out data to obtain $\hat p^{\star}$ and $\widehat{\Delta p}$ for every test sample.
9. Compute $\mathrm{AU}$, $\mathrm{EU}$, and $\mathrm{TU}$ per sample as in §2.4, and aggregate (e.g. mean) for reporting.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training_credit.ipynb`

### 4.1 Locating Ensemble Teachers

```python
def get_ensemble_paths(model_name):
    pattern = str(ENSEMBLE_DIR / f'{model_name}_ens_*_final.keras')
    paths   = glob.glob(pattern)
    if not paths:
        pattern = str(ENSEMBLE_DIR / f'ensembles_{model_name}' / f'{model_name}_ens_*_final.keras')
        paths = glob.glob(pattern)
    if not paths:
        pattern = str(MODEL_DIR / f'{model_name}_ens_*_final.keras')
        paths   = glob.glob(pattern)
    return sorted(paths)
```
**What this does:** searches three candidate directory layouts in order until it finds the saved `.keras` checkpoints for an architecture's ensemble teachers.
**Why:** the teachers were trained and saved in an earlier stage of the project (not shown in this notebook), and this helper makes the distillation step robust to different folder conventions without hard-coding a single path.

### 4.2 Deriving the Aleatoric and Epistemic Targets

```python
def generate_credit_targets(ensemble_paths, x_data, batch_size=128):
    all_preds = []
    for path in ensemble_paths:
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        all_preds.append(model.predict(x_data, batch_size=batch_size, verbose=1))
        del model
        tf.keras.backend.clear_session()

    stacked = tf.stack(all_preds, axis=0)   # (M, N, C)
    p_min   = tf.reduce_min(stacked, axis=0)
    p_max   = tf.reduce_max(stacked, axis=0)

    delta_p_true = p_max - p_min
    p_star_true  = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
    return p_star_true, delta_p_true
```
**What this does:** loads each teacher one at a time (freeing it from memory immediately afterward), collects all $M$ softmax predictions, then computes the per-class min/max across teachers exactly as defined in §2.2.
**Why:** this is the direct, literal implementation of the teacher-derived training targets; loading teachers one-by-one and clearing the Keras session avoids holding all $M$ models in memory simultaneously, which matters since some of these backbones (especially the ViT) are large.

### 4.3 Attaching the Dual CREDIT Heads

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
**What this does:** rebuilds the single-head base architecture, then severs its original output layer by reaching one layer back (`layers[-2]`) to grab the shared feature tensor, on top of which two new heads are attached.
**Why:** this is the architectural mechanism from §2.3 — both CREDIT heads share the backbone's learned representation, so the student is only marginally more expensive than the original single-head model it is built from.

### 4.4 Compiling with the Weighted Dual Loss

```python
student.compile(
    optimizer=optimizer,
    loss={
        'p_star':  tf.keras.losses.KLDivergence(),
        'delta_p': tf.keras.losses.MeanSquaredError(),
    },
    loss_weights={'p_star': 1.0, 'delta_p': 0.5},   # lambda = 0.5
)
```
**What this does:** assigns the KL-divergence loss to the `p_star` head and MSE to the `delta_p` head, then weights the latter by $\lambda = 0.5$ before summing.
**Why:** this is the direct implementation of the combined loss in §2.3; the $0.5$ weighting downweights the epistemic regression term relative to the aleatoric distributional term, reflecting that the two heads are not equally important to the overall objective in this configuration.

### 4.5 Post-Training Uncertainty Decomposition

```python
p_star_pred, delta_p_pred = student.predict(x_te, batch_size=BATCH_SIZE)

au = -np.sum(p_star_pred * np.log(p_star_pred + 1e-12), axis=-1)
eu =  np.mean(delta_p_pred, axis=-1)
tu =  au + eu
```
**What this does:** runs the trained student once on held-out data, then computes the entropy of the predicted aleatoric-proxy vector and the mean of the predicted epistemic-width vector for every sample.
**Why:** this is the literal implementation of the AU/EU/TU decomposition in §2.4 — note that only one model and one forward pass is needed here, in contrast to the five teacher forward passes that were needed to build the *training* targets in §4.2.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

To keep the arithmetic concrete, consider a toy version of the notebook's $C = 7$ class problem, but with $K = 4$ classes (a representative subset) and an ensemble of $M = 3$ teachers (instead of 7 and 5, to keep the example legible while preserving the same mechanics). Each teacher produces a 4-class softmax vector for every input.

Three test inputs are designed to show how the CREDIT decomposition behaves under different conditions:

- **Easy sample (A):** all three teachers agree strongly that class 1 is correct (each teacher's top-class probability $\geq 0.60$).
- **Borderline sample (B):** the teachers' top-class probability for the true class sits close to a 0.5 threshold, with some disagreement on which class is most likely.
- **Ambiguous sample (C):** no class dominates clearly for any teacher (top probability $\leq 0.40$ everywhere) and teachers disagree substantially with each other.

The three teachers' softmax vectors for each test sample (rows sum to 1.00, true class in **bold**):

**Sample A (true class = 1):**

| Teacher | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|
| 1 | **0.70** | 0.15 | 0.10 | 0.05 |
| 2 | **0.65** | 0.20 | 0.10 | 0.05 |
| 3 | **0.75** | 0.10 | 0.10 | 0.05 |

**Sample B (true class = 2):**

| Teacher | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|
| 1 | 0.20 | **0.50** | 0.20 | 0.10 |
| 2 | 0.35 | **0.40** | 0.15 | 0.10 |
| 3 | 0.15 | **0.55** | 0.20 | 0.10 |

**Sample C (true class = 3):**

| Teacher | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|
| 1 | 0.30 | 0.30 | **0.30** | 0.10 |
| 2 | 0.15 | 0.40 | **0.35** | 0.10 |
| 3 | 0.35 | 0.20 | **0.25** | 0.20 |

### 5.1 Step A — Per-Class Min/Max Across Teachers

For each test sample, take the per-class minimum and maximum across the three teacher rows above.

**Sample A:**

| Class | $p_{\min}$ | $p_{\max}$ |
|---|---|---|
| 1 | 0.65 | 0.75 |
| 2 | 0.10 | 0.20 |
| 3 | 0.10 | 0.10 |
| 4 | 0.05 | 0.05 |

**Sample B:**

| Class | $p_{\min}$ | $p_{\max}$ |
|---|---|---|
| 1 | 0.15 | 0.35 |
| 2 | 0.40 | 0.55 |
| 3 | 0.15 | 0.20 |
| 4 | 0.10 | 0.10 |

**Sample C:**

| Class | $p_{\min}$ | $p_{\max}$ |
|---|---|---|
| 1 | 0.15 | 0.35 |
| 2 | 0.20 | 0.40 |
| 3 | 0.25 | 0.35 |
| 4 | 0.10 | 0.20 |

### 5.2 Step B — Deriving $\Delta p_{\text{true}}$ and $p^{\star}_{\text{true}}$

The interval width is $\Delta p_{\text{true},c} = p_{\max,c} - p_{\min,c}$, computed per class:

| Sample | Class 1 $\Delta p$ | Class 2 $\Delta p$ | Class 3 $\Delta p$ | Class 4 $\Delta p$ |
|---|---|---|---|---|
| A | 0.10 | 0.10 | 0.00 | 0.00 |
| B | 0.20 | 0.15 | 0.05 | 0.00 |
| C | 0.20 | 0.20 | 0.10 | 0.10 |

The aleatoric-proxy target is the renormalised $p_{\min}$ vector, $p^{\star}_{\text{true},c} = p_{\min,c} / \sum_j p_{\min,j}$ (the $\varepsilon$ term is negligible here and omitted from the arithmetic):

**Sample A:** $\sum p_{\min} = 0.65 + 0.10 + 0.10 + 0.05 = 0.90$.
$$
p^{\star}_{\text{true}} = \left(\frac{0.65}{0.90}, \frac{0.10}{0.90}, \frac{0.10}{0.90}, \frac{0.05}{0.90}\right) = (0.722,\ 0.111,\ 0.111,\ 0.056).
$$

**Sample B:** $\sum p_{\min} = 0.15 + 0.40 + 0.15 + 0.10 = 0.80$.
$$
p^{\star}_{\text{true}} = \left(\frac{0.15}{0.80}, \frac{0.40}{0.80}, \frac{0.15}{0.80}, \frac{0.10}{0.80}\right) = (0.1875,\ 0.500,\ 0.1875,\ 0.125).
$$

**Sample C:** $\sum p_{\min} = 0.15 + 0.20 + 0.25 + 0.10 = 0.70$.
$$
p^{\star}_{\text{true}} = \left(\frac{0.15}{0.70}, \frac{0.20}{0.70}, \frac{0.25}{0.70}, \frac{0.10}{0.70}\right) = (0.214,\ 0.286,\ 0.357,\ 0.143).
$$

These vectors and the $\Delta p_{\text{true}}$ rows above are exactly what a real training pipeline would pair with each input $x$ as regression/KL targets for the two student heads, batched together as `(x, (p_star_true, delta_p_true))` exactly as the notebook's `tf.data.Dataset.from_tensor_slices` call does.

### 5.3 Step C — Hypothetical Trained-Student Predictions and Loss

Suppose that after training, the student's two heads produce the following predictions on these same three samples (a plausible, partially-but-not-perfectly fit student):

**Predicted $\hat p^{\star}$ (softmax head):**

| Sample | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|
| A | 0.69 | 0.14 | 0.12 | 0.05 |
| B | 0.20 | 0.47 | 0.21 | 0.12 |
| C | 0.24 | 0.27 | 0.32 | 0.17 |

**Predicted $\widehat{\Delta p}$ (sigmoid head):**

| Sample | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|
| A | 0.09 | 0.08 | 0.02 | 0.01 |
| B | 0.18 | 0.13 | 0.07 | 0.02 |
| C | 0.17 | 0.18 | 0.09 | 0.12 |

**KL-divergence loss on the $p^{\star}$ head for Sample B** (true vs. predicted, using natural log):
$$
\mathrm{KL}(p^{\star}_{\text{true}} \Vert \hat p^{\star}) = \sum_c p^{\star}_{\text{true},c} \log\frac{p^{\star}_{\text{true},c}}{\hat p^{\star}_c}.
$$
Plugging in Sample B's values $(0.1875, 0.500, 0.1875, 0.125)$ against $(0.20, 0.47, 0.21, 0.12)$:
$$
= 0.1875\log\frac{0.1875}{0.20} + 0.500\log\frac{0.500}{0.47} + 0.1875\log\frac{0.1875}{0.21} + 0.125\log\frac{0.125}{0.12}
$$
$$
\approx 0.1875(-0.0645) + 0.500(0.0619) + 0.1875(-0.1142) + 0.125(0.0408)
$$
$$
\approx -0.0121 + 0.0310 - 0.0214 + 0.0051 \approx 0.0027.
$$
This small positive value reflects that the predicted belief vector is already fairly close to the teacher-derived target for this sample.

**MSE loss on the $\Delta p$ head for Sample B:**
$$
\mathrm{MSE} = \frac{1}{4}\sum_c (\Delta p_{\text{true},c} - \widehat{\Delta p}_c)^2 = \frac{1}{4}\big[(0.20-0.18)^2 + (0.15-0.13)^2 + (0.05-0.07)^2 + (0.00-0.02)^2\big]
$$
$$
= \frac{1}{4}\big[0.0004 + 0.0004 + 0.0004 + 0.0004\big] = \frac{0.0016}{4} = 0.0004.
$$

**Combined loss for Sample B** with $\lambda = 0.5$:
$$
\mathcal{L} = \mathrm{KL} + \lambda\cdot\mathrm{MSE} = 0.0027 + 0.5(0.0004) = 0.0029.
$$

### 5.4 Step D — Inference-Time AU / EU / TU for All Three Samples

Using the predicted $\hat p^{\star}$ and $\widehat{\Delta p}$ from §5.3, compute the three uncertainty scalars per sample.

**Sample A** (easy): $\hat p^{\star} = (0.69, 0.14, 0.12, 0.05)$, $\widehat{\Delta p} = (0.09, 0.08, 0.02, 0.01)$.
$$
\mathrm{AU} = -\big[0.69\log 0.69 + 0.14\log 0.14 + 0.12\log 0.12 + 0.05\log 0.05\big]
$$
$$
= -\big[0.69(-0.3711) + 0.14(-1.9661) + 0.12(-2.1203) + 0.05(-2.9957)\big]
$$
$$
= -\big[-0.2561 - 0.2753 - 0.2544 - 0.1498\big] = 0.9356.
$$
$$
\mathrm{EU} = \frac{0.09+0.08+0.02+0.01}{4} = \frac{0.20}{4} = 0.0500. \qquad \mathrm{TU} = 0.9356 + 0.0500 = 0.9856.
$$

**Sample B** (borderline): $\hat p^{\star} = (0.20, 0.47, 0.21, 0.12)$, $\widehat{\Delta p} = (0.18, 0.13, 0.07, 0.02)$.
$$
\mathrm{AU} = -\big[0.20\log 0.20 + 0.47\log 0.47 + 0.21\log 0.21 + 0.12\log 0.12\big]
$$
$$
= -\big[0.20(-1.6094) + 0.47(-0.7550) + 0.21(-1.5606) + 0.12(-2.1203)\big]
$$
$$
= -\big[-0.3219 - 0.3549 - 0.3277 - 0.2544\big] = 1.2589.
$$
$$
\mathrm{EU} = \frac{0.18+0.13+0.07+0.02}{4} = \frac{0.40}{4} = 0.1000. \qquad \mathrm{TU} = 1.2589 + 0.1000 = 1.3589.
$$

**Sample C** (ambiguous): $\hat p^{\star} = (0.24, 0.27, 0.32, 0.17)$, $\widehat{\Delta p} = (0.17, 0.18, 0.09, 0.12)$.
$$
\mathrm{AU} = -\big[0.24\log 0.24 + 0.27\log 0.27 + 0.32\log 0.32 + 0.17\log 0.17\big]
$$
$$
= -\big[0.24(-1.4271) + 0.27(-1.3093) + 0.32(-1.1394) + 0.17(-1.7720)\big]
$$
$$
= -\big[-0.3425 - 0.3535 - 0.3646 - 0.3012\big] = 1.3618.
$$
$$
\mathrm{EU} = \frac{0.17+0.18+0.09+0.12}{4} = \frac{0.56}{4} = 0.1400. \qquad \mathrm{TU} = 1.3618 + 0.1400 = 1.5018.
$$

### 5.5 Summary Table

| Sample | Description | AU | EU | TU | Predicted class | True class | Correct? |
|---|---|---|---|---|---|---|---|
| A | Easy (high agreement) | 0.9356 | 0.0500 | 0.9856 | 1 | 1 | ✓ |
| B | Borderline | 1.2589 | 0.1000 | 1.3589 | 2 | 2 | ✓ |
| C | Ambiguous (no clear winner) | 1.3618 | 0.1400 | 1.5018 | 3 | 3 | ✓ |

The progression from Sample A to Sample C shows the intended behaviour of the CREDIT decomposition end to end: as the underlying teacher ensemble's votes become flatter and more spread out (visible already in the raw teacher tables in §5.0, and quantified in the wider $\Delta p_{\text{true}}$ values computed in §5.2), both the aleatoric proxy (entropy of $\hat p^{\star}$) and the epistemic proxy (mean of $\widehat{\Delta p}$) rise in tandem, and so does their sum, the total uncertainty $\mathrm{TU}$. This mirrors the directional pattern reported in the notebook's own output, where the three trained architectures' mean $\mathrm{TU}$ values track their mean $\mathrm{AU}$ values closely, with $\mathrm{EU}$ contributing a smaller but architecture-dependent share of the total.

---

## 6. References

[1] Wang, K.; Cuzzolin, F.; Moens, D.; Hallez, H. "Credal Ensemble Distillation for Uncertainty Quantification." Extended version accepted at AAAI 2026, arXiv:2511.13766, 2025. [Link](https://arxiv.org/abs/2511.13766)

[2] Lakshminarayanan, B.; Pritzel, A.; Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS, 2017. [Link](https://arxiv.org/abs/1612.01474)
