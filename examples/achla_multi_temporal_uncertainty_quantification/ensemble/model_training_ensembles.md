# Deep Ensembles for Patch-Based Multispectral Land-Cover Classification: Theory & Implementation Summary

> **One-line description:** Trains M independently-seeded copies of three backbone architectures (a legacy AlexNet-style CNN, a Global Filter Network, and a Vision Transformer with U-Net skip connections) on multispectral image patches, and uses per-member calibration diagnostics (Brier score, log-loss, Expected Calibration Error) as the substrate for a downstream Deep-Ensemble predictive-uncertainty analysis.

---

## 1. Overview & Intuition

A single trained neural network outputs a softmax vector that *looks* like a probability distribution, but there is no guarantee it behaves like one: modern networks are routinely overconfident, and the same architecture trained twice from different random initialisations can converge to different functions that agree almost everywhere on the training distribution and disagree sharply elsewhere — typically near class boundaries, on noisy pixels, or on inputs unlike anything seen during training. A single softmax score cannot tell these two situations apart: "the input is genuinely ambiguous" and "the model itself is unsure which function to trust" look identical from one network's point of view.

**Deep Ensembles** address this by training several independently-initialised copies of a model and treating the *spread* across copies as a proxy for the model's own uncertainty, on top of whatever uncertainty is already visible in any single member's softmax output. The idea, introduced by Lakshminarayanan et al. (2017), is deliberately simple compared to Bayesian neural networks: instead of learning a distribution over weights, train M ordinary networks with different random seeds (and, optionally, different data shuffling or bootstrap resampling), and combine their predictive distributions by averaging. No change to the loss function or training loop is required, and the approach has repeatedly been shown to produce predictive uncertainty estimates that are competitive with — and often better calibrated than — far more expensive Bayesian approximations.

The uploaded notebook implements the *training and diagnostic* half of this pipeline for a patch-based multispectral land-cover classification task. Square patches are cut out of a six-band image around every labelled pixel, and three architecturally distinct backbones are considered as candidate ensemble members: a legacy AlexNet-style convolutional network (purely local receptive fields), a Global Filter Network (GFNet, which mixes information across the whole patch in the frequency domain via a 2-D FFT), and a Vision Transformer with U-Net-style skip connections (which mixes information globally via self-attention, with explicit links between early and late transformer blocks). For whichever architecture is selected, the notebook trains M = 5 independently-seeded members, evaluates each one's accuracy *and* its calibration quality (multiclass Brier score, log-loss, and a 15-bin Expected Calibration Error), and stores enough artefacts (per-member checkpoints, summary tables, training curves, confusion matrices, full-scene classification maps) to support exactly the kind of cross-member combination that Deep Ensembles theory prescribes.

What makes this notebook a genuine ensemble-uncertainty pipeline rather than just "train several models and pick the best one" is the explicit collection of calibration proxy metrics per member (Section 7.4 is literally titled *"Uncertainty Proxy Metrics"*) and the closing utility cell, which reloads a saved checkpoint specifically "for downstream uncertainty analysis without retraining." The architecture comparison itself is also theoretically motivated: GFNet and the ViT-UNet both mix information across the *entire* patch (in the frequency domain, or via attention, respectively), which is exactly the kind of long-range context that a plain convolutional stack with only local 3×3 receptive fields lacks — and that lack of context is precisely where spectrally mixed, boundary-straddling pixels become hardest to classify confidently and correctly.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let the multispectral scene be a tensor $X \in \mathbb{R}^{H \times W \times B}$ with $H=330$, $W=307$ spatial rows/columns and $B=6$ spectral bands, each band independently min–max normalised to $[0,1]$. For every labelled pixel at spatial coordinate $(r,c)$ with $y_{(r,c)} \in \{1,\dots,K\}$, a fixed-size patch of side $P=9$ centred on that pixel is extracted (with edge-padding at the scene boundary), giving an input tensor $x \in \mathbb{R}^{P \times P \times B}$. The set of all such patches and labels forms the classification dataset $\{(x_i, y_i)\}_{i=1}^N$, split into train / validation / test partitions (and, for the legacy AlexNet recipe, a separate train/test split using a different seed and split ratio, to reproduce the behaviour of the original single-head script).

A backbone architecture $f$ (AlexNet-CNN, GFNet, or ViT-UNet) maps a patch to a softmax distribution over the $K$ classes, $f(x;\theta): x \mapsto p(y\mid x,\theta) \in \Delta^{K-1}$, where $\theta$ denotes the trainable weights and $\Delta^{K-1}$ is the probability simplex. A **deep ensemble** of size $M$ for a chosen backbone is a set of $M$ independently-trained weight vectors $\{\theta_m\}_{m=1}^M$, obtained by re-seeding the global random state (Python, NumPy, and TensorFlow) before each member is built and trained — so that weight initialisation, dropout masks, and (for the non-legacy architectures) the train/validation shuffling order all differ from member to member, while the training data and architecture family stay fixed.

### 2.2 Predictive Distribution and the Ensemble Combination Rule

Each member produces its own predictive distribution $p_m(y\mid x) \equiv f(x;\theta_m)$. Deep Ensembles theory treats the ensemble's predictive distribution as a uniformly-weighted mixture of its members:

$$\bar{p}(y \mid x) \;=\; \frac{1}{M}\sum_{m=1}^{M} p_m(y \mid x)$$

**Where:**
- $p_m(y\mid x)$ — the softmax probability vector produced by ensemble member $m$ for input $x$
- $M$ — the number of independently-seeded members in the ensemble (the notebook sets $M=5$)
- $\bar{p}(y\mid x)$ — the ensemble's combined predictive distribution, itself a valid probability vector over the $K$ classes

**What this means:** rather than trusting any single network's softmax output, the ensemble's belief about $x$ is the *average* of what every member believes. The predicted class is then $\hat{y} = \arg\max_y \bar{p}(y\mid x)$. Averaging probabilities (rather than, say, majority-voting hard labels) is the choice that lets the ensemble exploit information about *how confident* each member is, not just *which* class it picked.

### 2.3 Calibration Proxy Metrics

The notebook does not yet form $\bar{p}$ explicitly (see Section 3, Step 7, for why) — but it computes three calibration-quality metrics for every individual member, which are the quantities a downstream combination step would actually compare against the combined $\bar{p}$.

**Multiclass Brier score** (computed by `multiclass_brier_score` in the code):

$$\mathrm{BS} \;=\; \frac{1}{N}\sum_{i=1}^{N} \lVert p_i - \mathbf{y}_i \rVert_2^2 \;=\; \frac{1}{N}\sum_{i=1}^N \sum_{k=1}^{K} \big(p_{i,k} - \mathbf{y}_{i,k}\big)^2$$

**Where:**
- $p_i$ — the model's predicted probability vector for sample $i$
- $\mathbf{y}_i$ — the one-hot encoding of the true label for sample $i$
- $N$ — the number of evaluation samples

**What this means:** the Brier score is the mean squared distance between the predicted probability vector and the "ideal" one-hot vector. It penalises both wrong predictions *and* overconfident-but-correct ones less than miscalibrated ones, and it is a strictly proper scoring rule — a property used directly in Section 5.

**Negative log-likelihood / log-loss** (computed via `sklearn.metrics.log_loss` in the code):

$$\mathrm{NLL} \;=\; -\frac{1}{N}\sum_{i=1}^{N} \log p_i(y_i \mid x_i)$$

**Where:**
- $p_i(y_i\mid x_i)$ — the probability the model assigned to the *true* class of sample $i$

**What this means:** NLL only looks at the probability mass placed on the correct answer and penalises low confidence in the right class heavily (via the logarithm) — it is also a strictly proper scoring rule, and, importantly, a *convex* one (see Section 5.1, Step B, for why this matters for ensembling).

**Expected Calibration Error**, 15-bin version (computed by `expected_calibration_error` in the code):

$$\mathrm{ECE} \;=\; \sum_{b=1}^{n_{\text{bins}}} \frac{|B_b|}{N}\, \big|\, \mathrm{acc}(B_b) - \mathrm{conf}(B_b)\,\big|$$

**Where:**
- $B_b$ — the subset of evaluation samples whose top-class confidence $\max_k p_{i,k}$ falls in equal-width confidence bin $b$ (the code uses 15 bins over $[0,1]$)
- $\mathrm{acc}(B_b)$ — the fraction of samples in bin $b$ that were classified correctly
- $\mathrm{conf}(B_b)$ — the mean top-class confidence of samples in bin $b$
- $|B_b|/N$ — the bin's weight, proportional to how many evaluation samples fall in it

**What this means:** ECE asks, "among the times the model said it was, say, 80% confident, was it actually right 80% of the time?" A perfectly calibrated model has $\mathrm{acc}(B_b)=\mathrm{conf}(B_b)$ in every bin and ECE $=0$. Unlike Brier score and NLL, ECE is a *binning-based diagnostic*, not a proper scoring rule — a distinction that turns out to matter directly in the worked example below.

### 2.4 Uncertainty Decomposition: Total, Aleatoric, and Epistemic

The calibration metrics above are computed per individual member and are exactly what the notebook reports. The theoretical payoff of having $M$ members — and the "downstream uncertainty analysis" the final notebook cell explicitly defers to — is the ability to split the ensemble's *total* predictive uncertainty into two interpretable components, using a standard information-theoretic identity (Depeweg et al., 2018).

$$\underbrace{\mathbb{H}\!\big[\bar p(y\mid x)\big]}_{\text{total uncertainty}} \;=\; \underbrace{\frac{1}{M}\sum_{m=1}^{M}\mathbb{H}\!\big[p_m(y\mid x)\big]}_{\text{aleatoric (expected member entropy)}} \;+\; \underbrace{\Big(\mathbb{H}\big[\bar p\big] - \frac{1}{M}\sum_{m=1}^{M}\mathbb{H}[p_m]\Big)}_{\text{epistemic (mutual information)}}$$

where $\mathbb{H}[p] = -\sum_{k=1}^K p_k \log p_k$ is the Shannon entropy of a distribution.

**Where:**
- $\mathbb{H}[\bar p]$ — **total uncertainty**: the entropy of the ensemble-averaged distribution. High when the *combined* belief is spread across multiple classes.
- $\frac{1}{M}\sum_m \mathbb{H}[p_m]$ — **aleatoric uncertainty**: the average entropy each individual member exhibits on its own. High when every member, individually, is unsure — typically because the input itself is genuinely ambiguous (e.g. a spectrally mixed pixel straddling two land-cover classes), a property no amount of additional training data or extra ensemble members can remove.
- $\mathbb{H}[\bar p] - \frac{1}{M}\sum_m \mathbb{H}[p_m]$ — **epistemic uncertainty**, also called the mutual information between the predicted label and the choice of ensemble member: how much the members *disagree* with each other, over and above their average individual uncertainty. High when different random seeds led to genuinely different learned functions for this particular input — uncertainty that, in principle, more training data or more members could reduce.

**What this means:** this decomposition is what makes ensembles more informative than a single softmax score. Two inputs can have identical total uncertainty $\mathbb{H}[\bar p]$ for completely different reasons — one because every member individually hedges (aleatoric), the other because the members confidently disagree with each other (epistemic) — and only by inspecting the individual $p_m$'s, not just $\bar p$, can the two be told apart.

---

## 3. Algorithm

**Input:** a multispectral scene `data.csv` / `ref.csv` ($H{\times}W{\times}B$ pixels with per-pixel class labels), an architecture preset, an ensemble size $M$
**Output:** $M$ trained checkpoints per selected architecture, a metrics summary table, training/comparison/calibration plots, confusion matrices, full-scene classification maps, and (downstream) an ensemble-combined uncertainty map

1. **Load and preprocess.** Read the band-stacked CSV and label CSV, reshape to $(H,W,B)$, min–max normalise every band independently.
2. **Extract patches.** For every labelled pixel, cut a $9\times9\times6$ patch (edge-padded at scene borders); record the patch, the zero-indexed label, and the pixel's coordinates.
3. **Split the data.** Build a stratified train/validation/test split for the GFNet/ViT pipeline, plus a separately-seeded legacy train/test split that reproduces the original single-head AlexNet script's behaviour.
4. **Define architectures.** Build three single-head Keras models sharing a common patch size and number of classes: a configurable-depth AlexNet-style CNN, a GFNet built from stacked frequency-domain global-filter blocks, and a ViT with symmetric encoder/decoder skip connections and a CLS-token classification head.
5. **For the selected architecture(s), repeat $M$ times** (the notebook sets $M=5$):
 a. Re-seed Python, NumPy, and TensorFlow's global RNGs with a per-member seed.
 b. Build the model from its primary config; on a `ResourceExhaustedError` (Colab OOM), rebuild from a smaller deterministic fallback config instead.
 c. Compile with an architecture-specific optimiser/schedule (plain Adagrad with a hand-tuned cosine learning-rate schedule for the legacy AlexNet recipe; AdamW with cosine-decay and label smoothing for GFNet/ViT).
 d. Train with early-checkpointing on the best validation metric; save the best weights and the final full model, first to a local scratch directory, then copied to persistent storage.
 e. Run inference on the held-out evaluation split; compute accuracy, Cohen's kappa, macro/weighted F1, the validation and test multiclass Brier score, validation and test log-loss, and the 15-bin test ECE.
 f. Append this member's row to a running results table; for the final member ($i=M$), additionally retain the classification report, confusion matrix, and training history for plotting.
6. **Aggregate and report.** Save the per-member summary table to CSV; for each architecture, plot accuracy/loss training curves, cross-architecture bar charts of test accuracy / macro-F1 / kappa / training time, a calibration-proxy bar chart (NLL, Brier, ECE side by side, "lower is better"), and confusion-matrix heatmaps.
7. **Visualise the full scene.** Reload each architecture's best checkpoint, run dense patch-by-patch inference over every pixel of the scene (not just the labelled ones) to produce a full classification map, save individual and combined-overview PNGs, and embed everything into an Excel workbook.
8. **(Downstream, not yet implemented in this notebook)** Combine the $M$ members' predictive distributions per architecture via Equation 2.2 to obtain $\bar p$, and apply the decomposition of Section 2.4 to produce per-pixel total / aleatoric / epistemic uncertainty maps — the explicit purpose for which Step 5's checkpoints and Step 6's calibration diagnostics were collected.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training_ensembles.md`

### 4.1 Patch Extraction and Splits (Section 3)

```python
def extract_labeled_patches(x, y, patch_size=9):
    pad   = patch_size // 2
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    coords  = np.argwhere(y > 0)
    patches = np.empty((coords.shape[0], patch_size, patch_size, x.shape[-1]), dtype=np.float32)
    labels  = np.empty((coords.shape[0],), dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        labels[i]  = int(y[r, c]) - 1
    return patches, labels, coords
```
**What this does:** pads the scene by half the patch width on every side, then, for every pixel with a non-zero (i.e. labelled) ground-truth class, slices out the surrounding $9\times9\times6$ window and stores its zero-indexed class. **Why:** every ensemble member ultimately classifies *patches*, not raw pixels, so the spatial context within a $9\times9$ neighbourhood is what each architecture actually has access to — this is also exactly the scope an AlexNet-style local convolution can "see" directly, versus what GFNet's FFT or ViT's attention can mix across.

The two separate `train_test_split` calls — one stratified split shared by GFNet/ViT, and a second one with a different seed and ratio reserved for AlexNet — exist purely so the legacy AlexNet results remain numerically reproducible against an older single-head script, even though it means AlexNet members are trained and evaluated on a different partition of the data than GFNet/ViT members.

### 4.2 Shared Frequency- and Attention-Domain Layers (Section 4.1)

```python
class GlobalFilterLayer(layers.Layer):
    def call(self, x):
        x_2d       = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
        x_fft      = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        w_complex  = tf.complex(self.w_real, self.w_imag)
        x_filtered = x_fft * w_complex
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
        return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])
```
**What this does:** reshapes the flattened patch tokens back into a 2-D grid, takes a 2-D FFT, multiplies every frequency component by a *learned* complex weight (separately, per channel), inverts the FFT, and keeps only the real part. **Why:** this is GFNet's substitute for self-attention — multiplying in the frequency domain is mathematically a circular convolution in the spatial domain with a learned, globally-supported kernel, so every output token can depend on every input token (long-range mixing) at a fraction of self-attention's compute cost.

`PatchEncoderWithCLS` differs from the plain `PatchPositionEncoder` only by prepending a trainable CLS token before adding positional embeddings — the CLS token is what the ViT head eventually reads out as the patch's summary representation, in place of GFNet's global-average-pooling.

### 4.3 The Three Backbones (Sections 4.2–4.4)

The **AlexNet-CNN** (`build_alexnet`) is a straight stack of five same-padded $3\times3$ convolutions (no pooling between them, reflecting the legacy script's exact layer order), a single max-pool, then four dense layers with dropout between each, ending in a softmax. Every layer's receptive field stays local to the $9\times9$ patch.

The **GFNet** (`build_gfnet`) first slices the patch into non-overlapping $3\times3$ inner-patches (`INNER_PATCH=3`) via `PatchExtractor`, projects and positionally encodes them, then stacks five `gf_block`s — each a pre-norm `GlobalFilterLayer` followed by a gated-MLP with a residual connection — before global-average-pooling and a final dense softmax head.

The **ViT-UNet** (`build_vit_unet_singlehead`) tokenises the patch the same way, but encodes with a CLS token, then stacks twelve standard pre-norm transformer blocks (multi-head self-attention + FFN, each with its own residual). The "U-Net" naming reflects this code block:
```python
if i <= transformer_layers // 2:
    block_list.append(x)
else:
    x = layers.Add(name=f'vit_skip_add_{i+1}')([x, block_list[transformer_layers - i - 1]])
```
**What this does:** for the first half of the transformer stack, every block's output is cached; for the second half, each block's output is added back to the *mirror-image* cached output from the first half (block 12 added to block 1's output, block 11 to block 2's, and so on). **Why:** this gives the deeper, more abstract later blocks a direct residual path back to earlier, more local representations — the same encoder-decoder skip-connection idea popularised by U-Net for segmentation, repurposed here inside a transformer classification head rather than across a literal down/up-sampling encoder-decoder.

### 4.4 Calibration Helper Functions (Section 5.0)

```python
def expected_calibration_error(y_true, y_prob, n_bins=15):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct     = (predictions == y_true).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(correct[in_bin]) - np.mean(confidences[in_bin])) * prop
    return float(ece)
```
**What this does:** implements Section 2.3's ECE formula directly — bin every sample by its top-class confidence, and within each non-empty bin, weight the gap between empirical accuracy and mean confidence by that bin's share of the data. **Why:** this is computed identically for every ensemble member and for the validation/test splits, which is exactly the per-member input the worked example below feeds into an ensemble-level comparison.

### 4.5 The Ensemble Training Loop (Section 6.2)

```python
for model_name, builder in model_builders.items():
    if model_name != "GFNet":   # change this to the one you want
        continue
    for i in range(1, M + 1):
        seed_val = 42 + i
        tf.random.set_seed(seed_val); np.random.seed(seed_val); random.seed(seed_val)
        tf.keras.backend.clear_session()
        try:
            model = builder()
        except tf.errors.ResourceExhaustedError:
            model = build_gfnet_with_cfg(GFNET_FALLBACK_CFG)  # or ViT fallback
        ...
        row = {'model': f'{model_name}_ens_{i}', 'test_accuracy': ..., 'test_brier': ..., 'test_ece_15bin': ..., ...}
        results_rows.append(row)
```
**What this does:** for the architecture(s) not skipped by the `continue` guard, re-seeds all RNGs per member (`seed_val = 42 + i`, so members differ only in initialisation/shuffling, never in architecture or training data), clears the Keras session to avoid memory creep across members, builds the model with an automatic fallback to a smaller config on out-of-memory errors, trains with architecture-appropriate compile settings, and records every metric from Section 2.3 into one row per member of a growing results table. **Why this matters for the worked example:** as written, the loop's `continue` guard means *only GFNet* is actually trained in this run — AlexNet and ViT-UNet are defined and ready, but skipped — so a single execution of this notebook produces a 5-row summary table (`GFNet_ens_1` … `GFNet_ens_5`) rather than the full 15-row, three-architecture table that Section 7's plotting code is written to expect. Generating the full cross-architecture comparison requires re-running this cell with the guard changed, once per architecture.

### 4.6 Results, Comparison, and Calibration Plots (Section 7)

`summary_df` is sorted by descending test accuracy and saved to CSV; per-model classification reports are dumped as JSON. The training-curve plot draws a dual-axis accuracy/loss figure per architecture present in `model_artifacts` (i.e. only for the *last* member, $i=M$, of each architecture that was actually trained). The cross-model bar charts plot test accuracy, macro-F1, kappa, and training time side by side across whichever architectures' rows are present in `summary_df`. The calibration-proxy chart specifically groups `test_nll`, `test_brier`, and `test_ece_15bin` into one figure titled "Uncertainty Proxy Metrics (Lower is Better)" — the clearest signal in the notebook that these three metrics are intended as stand-ins for ensemble-level predictive-uncertainty quality, not just generic evaluation metrics.

### 4.7 Full-Scene Visualisation (Section 8)

`predict_full_scene_labels` pads the *entire* scene and runs the model row-by-row over every pixel (not just labelled ones), producing a dense $H\times W$ class-label map. This is run once per successfully-loaded architecture checkpoint, and `save_combined_overview` lays the approximate-RGB scene, the ground-truth label map, and each architecture's classification map side by side with a shared legend — the most direct visual comparison of how the three backbones disagree spatially, which is exactly where an epistemic-uncertainty map (Section 2.4) would be the natural next panel to add.

### 4.8 Reload Utility (Section 9)

```python
loaded_model = keras.models.load_model(example_path, custom_objects=VIS_CUSTOM_OBJECTS, compile=False, safe_mode=False)
```
**What this does:** a minimal convenience cell to reload any single saved `.keras` checkpoint without retraining. **Why it is included:** the comment explicitly states this is "for downstream uncertainty analysis" — confirming that everything upstream (per-member checkpoints, per-member calibration metrics) exists specifically to be picked back up and combined across members at a later stage, which is exactly the worked example performed in Section 5 below.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

To keep every number traceable by hand, this example uses $K=3$ classes (matching the notebook's generic 1-indexed class convention) and **reduces the ensemble size to $M=3$ members** (the notebook itself uses $M=5$; the combination logic is identical, just with three terms in every average instead of five). A toy evaluation set of $n=9$ patches is used, three per class, drawn from the GFNet architecture — the one architecture the loop in Section 4.5 actually trains in this run. Three of the nine patches are singled out as the test patches whose per-sample uncertainty decomposition (Section 5.2, Step C) is traced in full:

- **Easy patch ($S_1$, true class = Water):** every member assigns Water a high, mutually consistent probability.
- **Borderline patch ($S_6$, true class = Vegetation):** every member individually hedges between Vegetation and Urban — a case of high *aleatoric* uncertainty (genuine spectral mixing) rather than member disagreement.
- **Ambiguous patch ($S_9$, true class = Urban):** two of three members favour Urban while one favours Vegetation — a case of elevated *epistemic* uncertainty (the members themselves disagree).

**Calibration/test probability matrix — Member 1 (GFNet_ens_1):**

| Sample | True class | $p(\text{Water})$ | $p(\text{Vegetation})$ | $p(\text{Urban})$ |
|---|---|---|---|---|
| $S_1$ | Water | **0.86** | 0.09 | 0.05 |
| $S_2$ | Water | **0.78** | 0.15 | 0.07 |
| $S_3$ | Water | **0.55** | 0.35 | 0.10 |
| $S_4$ | Vegetation | 0.10 | **0.80** | 0.10 |
| $S_5$ | Vegetation | 0.08 | **0.85** | 0.07 |
| $S_6$ | Vegetation | 0.20 | **0.50** | 0.30 |
| $S_7$ | Urban | 0.05 | 0.10 | **0.85** |
| $S_8$ | Urban | 0.07 | 0.13 | **0.80** |
| $S_9$ | Urban | 0.10 | 0.25 | **0.65** |

**Calibration/test probability matrix — Member 2 (GFNet_ens_2):**

| Sample | True class | $p(\text{Water})$ | $p(\text{Vegetation})$ | $p(\text{Urban})$ |
|---|---|---|---|---|
| $S_1$ | Water | **0.88** | 0.08 | 0.04 |
| $S_2$ | Water | **0.82** | 0.12 | 0.06 |
| $S_3$ | Water | **0.48** | 0.42 | 0.10 |
| $S_4$ | Vegetation | 0.08 | **0.85** | 0.07 |
| $S_5$ | Vegetation | 0.10 | **0.82** | 0.08 |
| $S_6$ | Vegetation | 0.25 | **0.45** | 0.30 |
| $S_7$ | Urban | 0.04 | 0.08 | **0.88** |
| $S_8$ | Urban | 0.06 | 0.10 | **0.84** |
| $S_9$ | Urban | 0.20 | 0.35 | **0.45** |

**Calibration/test probability matrix — Member 3 (GFNet_ens_3):**

| Sample | True class | $p(\text{Water})$ | $p(\text{Vegetation})$ | $p(\text{Urban})$ |
|---|---|---|---|---|
| $S_1$ | Water | **0.84** | 0.11 | 0.05 |
| $S_2$ | Water | **0.70** | 0.20 | 0.10 |
| $S_3$ | Water | **0.40** | 0.45 | 0.15 |
| $S_4$ | Vegetation | 0.15 | **0.75** | 0.10 |
| $S_5$ | Vegetation | 0.12 | **0.78** | 0.10 |
| $S_6$ | Vegetation | 0.30 | **0.40** | 0.30 |
| $S_7$ | Urban | 0.10 | 0.15 | **0.75** |
| $S_8$ | Urban | 0.12 | 0.18 | **0.70** |
| $S_9$ | Urban | 0.30 | **0.45** | 0.25 |

(All rows sum to 1.00; bold marks the probability assigned to the true class — except $S_9$ under Member 3, where the bold mark is on the class Member 3 actually predicts, Vegetation, to flag the disagreement traced in Step C.)

---

### 5.1 Method: GFNet Deep Ensemble (M = 3)

#### Step A — Per-member calibration metrics on the toy evaluation set

For Member 1, the Brier-score contribution of sample $i$ is $(p_{i,\text{true}}-1)^2 + \sum_{k\neq\text{true}} p_{i,k}^2$ and the NLL contribution is $-\log p_{i,\text{true}}$ (natural log, in nats). Evaluated for every sample:

| Sample | True class | $p_\text{true}$ | Brier contribution | $-\log p_\text{true}$ |
|---|---|---|---|---|
| $S_1$ | Water | 0.86 | 0.0302 | 0.1508 |
| $S_2$ | Water | 0.78 | 0.0758 | 0.2485 |
| $S_3$ | Water | 0.55 | 0.3350 | 0.5978 |
| $S_4$ | Vegetation | 0.80 | 0.0600 | 0.2231 |
| $S_5$ | Vegetation | 0.85 | 0.0338 | 0.1625 |
| $S_6$ | Vegetation | 0.50 | 0.3800 | 0.6931 |
| $S_7$ | Urban | 0.85 | 0.0350 | 0.1625 |
| $S_8$ | Urban | 0.80 | 0.0618 | 0.2231 |
| $S_9$ | Urban | 0.65 | 0.1950 | 0.4308 |
| **Sum** | | | **1.2066** | **2.8922** |

$\Rightarrow$ Member 1: $\mathrm{BS}=1.2066/9=0.1341$, $\mathrm{NLL}=2.8922/9=0.3214$.

Repeating the identical arithmetic for Members 2 and 3 (every sample's contribution computed the same way from their probability matrices above) gives:

| Sample | True class | M2 $p_\text{true}$ | M2 Brier | M2 $-\log p_\text{true}$ | M3 $p_\text{true}$ | M3 Brier | M3 $-\log p_\text{true}$ |
|---|---|---|---|---|---|---|---|
| $S_1$ | Water | 0.88 | 0.0224 | 0.1278 | 0.84 | 0.0402 | 0.1744 |
| $S_2$ | Water | 0.82 | 0.0504 | 0.1985 | 0.70 | 0.1400 | 0.3567 |
| $S_3$ | Water | 0.48 | 0.4568 | 0.7340 | 0.40 | 0.5850 | 0.9163 |
| $S_4$ | Vegetation | 0.85 | 0.0338 | 0.1625 | 0.75 | 0.0950 | 0.2877 |
| $S_5$ | Vegetation | 0.82 | 0.0488 | 0.1985 | 0.78 | 0.0728 | 0.2485 |
| $S_6$ | Vegetation | 0.45 | 0.4550 | 0.7985 | 0.40 | 0.5400 | 0.9163 |
| $S_7$ | Urban | 0.88 | 0.0224 | 0.1278 | 0.75 | 0.0950 | 0.2877 |
| $S_8$ | Urban | 0.84 | 0.0392 | 0.1744 | 0.70 | 0.1368 | 0.3567 |
| $S_9$ | Urban | 0.45 | 0.4650 | 0.7985 | 0.25 | 0.8550 | 1.3863 |
| **Sum** | | | **1.5938** | **3.3205** | | **2.5598** | **4.9306** |

$\Rightarrow$ Member 2: $\mathrm{BS}=0.1771$, $\mathrm{NLL}=0.3689$. Member 3: $\mathrm{BS}=0.2844$, $\mathrm{NLL}=0.5478$.

Note that $S_3$ and $S_9$ under Member 3 are the two samples where Member 3's own argmax is *wrong*: at $S_3$, Member 3's highest probability (0.45) sits on Vegetation rather than the true class Water; at $S_9$, its highest probability (0.45) sits on Vegetation rather than the true class Urban. This is exactly the kind of individual-member error that ensemble averaging is positioned to correct (see Step B).

**ECE (4 equal-width bins, for legibility — the notebook uses 15):** binning Member 1's nine confidences $\{0.86,0.78,0.55,0.80,0.85,0.50,0.85,0.80,0.65\}$ by $\max_k p_k$ gives three samples ($S_3,S_6,S_9$) in $[0.5,0.75)$, all correct, mean confidence $0.5667 \Rightarrow$ contributes $\frac{3}{9}\,|1-0.5667|=0.1444$; the remaining six samples land in $[0.75,1.0]$, all correct, mean confidence $0.8233\Rightarrow$ contributes $\frac{6}{9}\,|1-0.8233|=0.1178$. $\mathrm{ECE}(M_1)=0.1444+0.1178=0.2622$. The identical binning procedure gives $\mathrm{ECE}(M_2)=0.2811$ and $\mathrm{ECE}(M_3)=0.1978$.

**Full score list, Member 1:** Brier $= \{0.0302,0.0758,0.3350,0.0600,0.0338,0.3800,0.0350,0.0618,0.1950\}$; mean $=0.1341$.

#### Step B — Forming the ensemble combination $\bar p$

For every sample, $\bar p = \frac{1}{3}(p_{M1}+p_{M2}+p_{M3})$. Worked explicitly for the three highlighted test patches:

$$\bar p(S_1) = \tfrac{1}{3}\big([0.86,0.09,0.05]+[0.88,0.08,0.04]+[0.84,0.11,0.05]\big) = [0.8600,\,0.0933,\,0.0467]$$

$$\bar p(S_6) = \tfrac{1}{3}\big([0.20,0.50,0.30]+[0.25,0.45,0.30]+[0.30,0.40,0.30]\big) = [0.2500,\,0.4500,\,0.3000]$$

$$\bar p(S_9) = \tfrac{1}{3}\big([0.10,0.25,0.65]+[0.20,0.35,0.45]+[0.30,0.45,0.25]\big) = [0.2000,\,0.3500,\,0.4500]$$

Applying the same averaging to all nine samples and recomputing the calibration metrics *on $\bar p$* gives ensemble-level $\mathrm{BS}=0.1910$ and $\mathrm{NLL}=0.4021$. Both are lower than the *average of the three members' individual scores* ($\overline{\mathrm{BS}}=(0.1341+0.1771+0.2844)/3=0.1985$; $\overline{\mathrm{NLL}}=(0.3214+0.3689+0.5478)/3=0.4127$) — exactly as Jensen's inequality guarantees for any convex (proper) scoring rule: averaging probabilities before scoring can never do worse, on average, than the average of the members' individual scores. Concretely, the ensemble also *fixes* both of Member 3's individual misclassifications: at $S_3$, $\bar p=[0.4767,0.4067,0.1167]$ correctly puts Water first; at $S_9$, $\bar p=[0.20,0.35,0.45]$ correctly puts Urban first.

The ensemble's ECE, however, comes out to $0.3081$ — *worse* than the average of the individual members' ECEs ($0.2470$). This is not a contradiction: ECE is a binning diagnostic, not a proper scoring rule, so it enjoys no Jensen's-inequality guarantee. Here, averaging softens the members' confident-but-slightly-wrong-direction probabilities toward the middle of the simplex; accuracy stays at 100% on this toy set, but mean confidence in the affected bins drops, *widening* the accuracy–confidence gap that ECE measures even as the strictly-proper Brier/NLL scores improve. This is a genuine, well-documented quirk of ECE and a good reason calibration work increasingly favours proper scoring rules over ECE alone.

#### Step C — Per-test-sample uncertainty decomposition

**Easy patch $S_1$ (true = Water).** Individual member entropies: $\mathbb{H}(M_1)=0.4962$, $\mathbb{H}(M_2)=0.4433$, $\mathbb{H}(M_3)=0.5391$ nats; mean (aleatoric) $=0.4929$. Ensemble entropy $\mathbb{H}[\bar p]=0.4940$.

| Class | $p_{M1}$ | $p_{M2}$ | $p_{M3}$ | $\bar p$ | In predicted set? |
|---|---|---|---|---|---|
| Water | 0.86 | 0.88 | 0.84 | **0.8600** | ✓ (argmax) |
| Vegetation | 0.09 | 0.08 | 0.11 | 0.0933 | |
| Urban | 0.05 | 0.04 | 0.05 | 0.0467 | |

Predicted class: **Water**. True class covered: **✓**. Epistemic uncertainty $=0.4940-0.4929=\mathbf{0.0011}$ — essentially zero: all three members agree closely both on the predicted class and on how confident to be.

**Borderline patch $S_6$ (true = Vegetation).** Individual entropies: $\mathbb{H}(M_1)=1.0297$, $\mathbb{H}(M_2)=1.0671$, $\mathbb{H}(M_3)=1.0889$; mean (aleatoric) $=1.0619$. Ensemble entropy $\mathbb{H}[\bar p]=1.0671$.

| Class | $p_{M1}$ | $p_{M2}$ | $p_{M3}$ | $\bar p$ | In predicted set? |
|---|---|---|---|---|---|
| Water | 0.20 | 0.25 | 0.30 | 0.2500 | |
| Vegetation | 0.50 | 0.45 | 0.40 | **0.4500** | ✓ (argmax) |
| Urban | 0.30 | 0.30 | 0.30 | 0.3000 | |

Predicted class: **Vegetation**. True class covered: **✓**, but by a thin margin over Urban (0.45 vs 0.30). Epistemic uncertainty $=1.0671-1.0619=\mathbf{0.0052}$ — still low: every member individually hedges between Vegetation and Urban in almost the same proportions, so the high *total* uncertainty here ($1.0671$, the highest of the three test patches) is almost entirely **aleatoric** — a property of the patch itself (genuine spectral mixing at a class boundary), not of model disagreement.

**Ambiguous patch $S_9$ (true = Urban).** Individual entropies: $\mathbb{H}(M_1)=0.8569$, $\mathbb{H}(M_2)=1.0486$, $\mathbb{H}(M_3)=1.0671$; mean (aleatoric) $=0.9909$. Ensemble entropy $\mathbb{H}[\bar p]=1.0486$.

| Class | $p_{M1}$ | $p_{M2}$ | $p_{M3}$ | $\bar p$ | In predicted set? |
|---|---|---|---|---|---|
| Water | 0.10 | 0.20 | 0.30 | 0.2000 | |
| Vegetation | 0.25 | 0.35 | **0.45** | 0.3500 | |
| Urban | **0.65** | **0.45** | 0.25 | **0.4500** | ✓ (argmax) |

Predicted class: **Urban** (ensemble), but Members 1 and 2 individually pick Urban while Member 3 individually picks Vegetation (its own argmax, 0.45, sits on Vegetation rather than Urban). True class covered: **✓** by the ensemble, despite one of three members disagreeing on the label. Epistemic uncertainty $=1.0486-0.9909=\mathbf{0.0577}$ — roughly five times $S_1$'s and ten times $S_6$'s, correctly flagging that this is the one test patch where the members' learned functions genuinely diverge, not just hedge similarly.

#### Step D — Summary table for the GFNet ensemble

| Test patch | Predicted (ensemble) | True class | Total $\mathbb{H}[\bar p]$ | Aleatoric | Epistemic | Correct? |
|---|---|---|---|---|---|---|
| $S_1$ (Easy) | Water | Water | 0.4940 | 0.4929 | 0.0011 | ✓ |
| $S_6$ (Borderline) | Vegetation | Vegetation | 1.0671 | 1.0619 | 0.0052 | ✓ |
| $S_9$ (Ambiguous) | Urban | Urban | 1.0486 | 0.9909 | 0.0577 | ✓ |

---

### 5.2 Cross-Architecture Comparison

The notebook defines AlexNet-CNN and ViT-UNet alongside GFNet, but the `continue` guard in Section 4.5 trains only GFNet in this particular run. To illustrate why the architecture choice matters for the same uncertainty pipeline, the table below applies the identical $M=3$-member ensembling and entropy decomposition to small, illustrative toy ensembles for the other two backbones, on the **same three test patches** used above.

**AlexNet-CNN** (purely local $3\times3$ receptive fields, no cross-patch mixing mechanism): on $S_1$ all three members agree confidently and correctly (mean $\bar p_\text{Water}=0.9233$). On $S_6$, all three members confidently agree — but on the *wrong* class: $\bar p = [0.1767, 0.3233, 0.5000]$, predicting Urban when the true class is Vegetation. On $S_9$, the same pattern repeats: $\bar p=[0.2767,0.4233,0.3000]$, confidently predicting Vegetation when the true class is Urban. In both failure cases the epistemic component is tiny (members agree with each other), illustrating an important limitation: **low epistemic uncertainty signals member *agreement*, not correctness** — when an architecture's inductive bias (here, a receptive field too local to see the class-boundary context) causes every randomly-seeded member to make the *same* systematic mistake, the ensemble is confidently wrong with no internal disagreement to flag it.

**ViT-UNet** (global self-attention plus encoder–decoder skip connections): on $S_1$, members agree and are correct ($\bar p_\text{Water}=0.8367$). On $S_6$, members agree *and* are correct ($\bar p=[0.20,0.5233,0.2767]$, correctly favouring Vegetation) — the global context the U-Net-style skip connections provide is enough to resolve the boundary ambiguity that defeated AlexNet. On $S_9$, members again agree and are correct, confidently ($\bar p=[0.10,0.20,0.70]$).

| Test patch | True class | AlexNet-CNN ensemble | GFNet ensemble | ViT-UNet ensemble |
|---|---|---|---|---|
| $S_1$ (Easy) | Water | Water, $p=0.9233$ ✓ | Water, $p=0.8600$ ✓ | Water, $p=0.8367$ ✓ |
| $S_6$ (Borderline) | Vegetation | **Urban**, $p=0.5000$ ✗ | Vegetation, $p=0.4500$ ✓ | Vegetation, $p=0.5233$ ✓ |
| $S_9$ (Ambiguous) | Urban | **Vegetation**, $p=0.4233$ ✗ | Urban, $p=0.4500$ ✓ | Urban, $p=0.7000$ ✓ |

**Effective confidence-in-true-class comparison** (the natural analogue, for this method, of a per-class decision threshold — the ensemble-averaged probability mass each architecture actually places on the correct answer):

| Class (as it appears in test patches) | AlexNet-CNN | GFNet | ViT-UNet |
|---|---|---|---|
| Water ($S_1$) | 0.9233 | 0.8600 | 0.8367 |
| Vegetation ($S_6$) | 0.3233 (true class, not predicted) | 0.4500 | 0.5233 |
| Urban ($S_9$) | 0.3000 (true class, not predicted) | 0.4500 | 0.7000 |

**Why the architectures differ:** AlexNet-CNN's convolutions only ever see a $9\times9$ window with no explicit long-range mixing operator, so on patches where the discriminating signal lies in the broader spatial/spectral *pattern* around a boundary pixel rather than in the immediate neighbourhood, every member converges to the same locally-plausible but globally-wrong answer — a systematic, low-epistemic, high-aleatoric-looking failure that is actually architectural bias in disguise. GFNet's frequency-domain global filter and ViT-UNet's self-attention (further reinforced by the encoder–decoder skip connections, which let late, abstract representations draw directly on early, more local features) both give every member access to whole-patch context, which is enough to correctly resolve the same two patches that defeated AlexNet — with ViT-UNet doing so the most confidently of the three on this toy set, plausibly because its skip connections combine both the local detail AlexNet relies on *and* the global context GFNet relies on.

---

## 6. References

[1] Lakshminarayanan, B., Pritzel, A., and Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*, 2017. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)

[2] Rao, Y., Zhao, W., Zhu, Z., Lu, J., and Zhou, J. "Global Filter Networks for Image Classification." *NeurIPS*, 2021. [arXiv:2107.00645](https://arxiv.org/abs/2107.00645)

[3] Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

[4] Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015.

[5] Krizhevsky, A., Sutskever, I., and Hinton, G. E. "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*, 2012.

[6] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. "On Calibration of Modern Neural Networks." *ICML*, 2017. [PMLR v70](https://proceedings.mlr.press/v70/guo17a.html)

[7] Brier, G. W. "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*, 78(1), 1950.

[8] Depeweg, S., Hernández-Lobato, J. M., Doshi-Velez, F., and Udluft, S. "Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning." *ICML*, 2018. [PMLR v80](https://proceedings.mlr.press/v80/depeweg18a.html)
