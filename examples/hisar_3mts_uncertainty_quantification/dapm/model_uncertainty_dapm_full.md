# DAPM (Diffusion-Augmented Probabilistic Model): Theory & Implementation Summary

> **One-line description:** DAPM combines a VAE encoder with a latent-space conditional diffusion denoiser to generate stochastic class-probability samples, then uses a Welch t-test on the top-two class distributions to flag spatially uncertain pixels in remote-sensing classification.

---

## 1. Overview & Intuition

### The problem: deterministic classifiers are overconfident

Standard deep classifiers — even powerful architectures like AlexNet-CNN, GFNet, or ViT-UNet — are fundamentally deterministic: given a patch, they produce a single probability vector. In spatially complex scenes such as multispectral remote-sensing images, many pixels sit on class boundaries, in shadow, or under mixed-cover conditions where no single label is truly correct. A softmax output of `[0.51, 0.49]` looks nearly the same as `[0.99, 0.01]` once the argmax is taken, yet the two predictions carry vastly different reliability.

Conventional uncertainty proxies (entropy, max-softmax probability) use a single forward pass and therefore reflect only the sharpness of that one prediction. They cannot tell whether the variability across different random draws from the input's distribution would be similarly concentrated or wildly spread.

### The insight: model the *distribution* over predictions

DAPM addresses this by treating the mapping from an input patch to a class probability vector as a *probabilistic* process. Instead of producing one softmax vector, DAPM produces **N independent stochastic draws** of that vector (N = 30 in the notebook). Uncertainty is then measured by comparing those 30 vectors statistically: if the top-1 class consistently dominates (tight distribution), the pixel is "certain"; if the top-1 and top-2 classes are statistically indistinguishable, the pixel is "uncertain."

### What makes it different

DAPM achieves stochasticity through two interacting components:

1. **VAE Encoder** — maps the pre-trained model's penultimate-layer features to a Gaussian latent distribution `q(z | x)`. Each draw samples a different latent code `z`.
2. **Conditional Diffusion Denoiser** — takes the sampled latent code `z` plus a soft "guidance" prediction from a latent classifier, then runs a full **T-step reverse diffusion chain** in *label space* (over `K` class probabilities), each time starting from Gaussian noise. The output of the denoiser is a fresh class-probability vector.

Crucially, the denoiser starts from different noise realisations on every call, producing genuinely diverse probability samples even for the same input patch. By running it N times, DAPM obtains an empirical distribution over class probabilities for every pixel, which is then subjected to a Welch two-sample t-test to decide whether the pixel is certain or uncertain.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

| Symbol | Meaning |
|--------|---------|
| $x \in \mathbb{R}^{P \times P \times B}$ | Input patch of size $P \times P$ with $B$ spectral bands ($P=9$, $B=6$) |
| $K$ | Number of classes ($K=7$) |
| $f_\theta$ | Frozen pre-trained base model (AlexNet-CNN / GFNet / ViT-UNet) |
| $\phi(x) \in \mathbb{R}^{d}$ | Penultimate-layer feature vector from $f_\theta$ |
| $z \in \mathbb{R}^{L}$ | Latent code ($L=64$) |
| $\hat{y} \in \Delta^{K-1}$ | Softmax probability vector (lives on the $(K-1)$-simplex) |
| $T$ | Number of diffusion time steps (configured per model, typically 100) |
| $N$ | Number of stochastic samples drawn per pixel ($N = 30$) |
| $\alpha_{\mathrm{ttest}}$ | P-value threshold for the Welch t-test ($\alpha_{\mathrm{ttest}} = 0.05$) |

---

### 2.2 VAE Encoder — Latent Code Sampling

The VAE encoder $\text{Enc}_\psi$ maps the feature vector $\phi(x)$ to a Gaussian posterior over the latent space:

$$q(z \mid x) = \mathcal{N}(\mu_z, \, \text{diag}(\sigma_z^2))$$

The encoder outputs two vectors:

$$(\mu_z,\, \log \sigma_z^2) = \text{Enc}_\psi(\phi(x))$$

A latent sample is drawn via the **reparameterisation trick**:

$$z = \mu_z + \sigma_z \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_L)$$

**Where:**
- $\mu_z \in \mathbb{R}^L$ — posterior mean vector
- $\sigma_z^2 \in \mathbb{R}^L$ — posterior variance (element-wise)
- $\odot$ — element-wise multiplication
- $\varepsilon$ — standard Gaussian noise drawn fresh for every sample

**What this means:** The reparameterisation trick allows gradients to flow through the sampling operation during training. At inference time it is the source of stochasticity: each of the $N$ draws uses a different $\varepsilon$, giving a different latent code $z$ even for the same input.

---

### 2.3 Latent Classifier — Soft Guidance Signal

A small classifier network $\text{Clf}_\omega$ maps the **posterior mean** $\mu_z$ (not the sampled $z$) to a soft class-probability vector:

$$g = \text{Clf}_\omega(\mu_z) = \text{softmax}(W_\omega \cdot \text{ReLU}(V_\omega \cdot \mu_z))$$

$g \in \Delta^{K-1}$ serves as a *guidance signal* injected into the diffusion denoiser at every reverse step. Its purpose is to bias the diffusion chain toward the correct label territory, similar to how classifier guidance works in image-generation diffusion models.

---

### 2.4 Linear (Cosine-inspired) Beta Schedule

Before running diffusion, a noise schedule is pre-computed:

$$\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1}(\beta_{\text{end}} - \beta_{\text{start}}), \quad t = 1, \ldots, T$$

$$\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

**Where:**
- $\beta_t$ — the noise variance added at step $t$ (linearly interpolated from $\beta_{\text{start}}$ to $\beta_{\text{end}}$)
- $\alpha_t$ — the signal retention factor at step $t$
- $\bar{\alpha}_t$ — the *cumulative* signal retention factor (how much of the original signal survives through step $t$)

This is a standard **linear beta schedule**. It governs how fast the forward process corrupts the label signal and therefore how the reverse (denoising) process must reconstruct it.

---

### 2.5 Conditional Diffusion Denoiser

#### Forward process (conceptual only — not run at inference)

In training, the forward process gradually corrupts a clean label vector $y_0$ over $T$ steps:

$$y_t = \sqrt{\bar{\alpha}_t}\, y_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_K)$$

At step $T$ the label has been fully destroyed into noise.

#### Reverse denoising (run at inference)

The denoiser network $D_\xi(z, y_t, g, t)$ predicts the noise $\varepsilon$ that was added at step $t$. Given the prediction, the reverse update rule recovers $y_{t-1}$ from $y_t$:

$$y_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( y_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot D_\xi(z, y_t, g, t) \right) + \sqrt{\beta_t} \cdot \varepsilon', \quad \varepsilon' \sim \mathcal{N}(0, I_K)$$

For the final step ($t=1$), the stochastic noise term $\sqrt{\beta_1}\varepsilon'$ is omitted.

**Where:**
- $y_t \in \mathbb{R}^K$ — the noisy label vector at step $t$ (starts from pure Gaussian noise $y_T \sim \mathcal{N}(0, I_K)$ at inference)
- $z$ — the latent code sampled from the VAE encoder
- $g$ — the soft guidance vector from the latent classifier
- $t$ — the current time step (embedded as a learned embedding)
- $D_\xi$ — the denoiser (MLP with two 256-unit hidden layers plus a learnable time embedding)

**What this means:** Starting from random noise in label space, the denoiser iteratively refines $y_t$ over $T$ steps, guided by both the latent code $z$ (which encodes *what kind of object* was seen) and the guidance $g$ (which biases toward the most likely label). After $T$ steps, $y_0$ is recovered and passed through a softmax to produce a valid probability vector.

---

### 2.6 Stochastic Sampling for Uncertainty Estimation

To obtain $N$ independent predictions for a single pixel, DAPM tiles the latent codes and guidance vectors, then runs the full reverse diffusion chain once for all $N$ copies simultaneously:

$$\{z^{(n)}\}_{n=1}^N = \mu_z + \sigma_z \odot E, \quad E \in \mathbb{R}^{N \times L}, \; E_{ij} \sim \mathcal{N}(0,1)$$

Each $z^{(n)}$ starts its own independent diffusion chain from its own noise draw $y_T^{(n)} \sim \mathcal{N}(0, I_K)$, producing N probability vectors:

$$\{\hat{y}^{(n)}\}_{n=1}^N = \text{softmax}(y_0^{(n)})$$

This gives an **empirical distribution** over class predictions for every pixel.

---

### 2.7 Welch T-Test Uncertainty Criterion

Given the $N$ probability samples, the mean probability vector is computed and the two classes with the highest mean probability are identified:

$$\bar{y}_k = \frac{1}{N} \sum_{n=1}^N \hat{y}_k^{(n)}, \quad k = 1, \ldots, K$$

$$c_1 = \arg\max_k \bar{y}_k, \quad c_2 = \arg\max_{k \neq c_1} \bar{y}_k$$

The **Welch two-sample t-test** is then applied to compare the top-1 and top-2 class probability distributions across the $N$ samples:

$$t = \frac{\bar{g}_1 - \bar{g}_2}{\sqrt{\frac{s_1^2}{N} + \frac{s_2^2}{N}}}$$

$$\bar{g}_k = \frac{1}{N}\sum_{n=1}^N \hat{y}_{c_k}^{(n)}, \quad s_k^2 = \frac{1}{N-1}\sum_{n=1}^N (\hat{y}_{c_k}^{(n)} - \bar{g}_k)^2$$

The **p-value** $p$ of this t-statistic is then compared to a threshold:

$$\text{uncertain} = \begin{cases} \text{True} & \text{if } p > \alpha_{\mathrm{ttest}} \\ \text{False} & \text{if } p \leq \alpha_{\mathrm{ttest}} \end{cases}$$

**What this means:** A low p-value means the top-1 class is *significantly* more probable than the top-2 across all $N$ samples — the pixel is confidently classified. A high p-value means the two classes are statistically indistinguishable — the pixel is flagged as uncertain.

The Welch variant (rather than Student's t) is used because it does not assume equal variance between the two groups, which is appropriate since the class probability distributions for top-1 and top-2 can have different spreads.

---

## 3. Algorithm

**Input:** Multispectral image $X \in \mathbb{R}^{H \times W \times B}$; DAPM bundle (feature extractor, VAE encoder, latent classifier, diffusion denoiser, noise schedule); $N$, $T$, $\alpha_{\mathrm{ttest}}$

**Output:** Per-pixel maps of: predicted class, uncertain mask, p-values, top-1/top-2 gap

1. **Patch extraction** — For every pixel $(r, c)$ extract a $P \times P$ spatial neighbourhood centred on that pixel (edge-padded via reflection).
2. **Feature extraction** — Pass the batch of patches through the frozen base model to obtain feature vectors $\phi(x) \in \mathbb{R}^d$.
3. **VAE encoding** — Pass $\phi(x)$ through the VAE encoder to obtain $(\mu_z, \log \sigma_z^2)$.
4. **Soft guidance** — Pass $\mu_z$ through the latent classifier to obtain the guidance vector $g$.
5. **Latent tiling** — Tile $\mu_z$, $\sigma_z$, and $g$ by $N$ copies; draw noise $E \sim \mathcal{N}(0, I)$ and compute $z^{(n)} = \mu_z + \sigma_z \odot \varepsilon^{(n)}$.
6. **Reverse diffusion** — Initialise $y_T \sim \mathcal{N}(0, I_K)$ for all $N$ copies. Loop from $t = T$ down to $t = 1$:
   - Compute $\hat{\varepsilon} = D_\xi(z, y_t, g, t)$
   - Apply the DDPM reverse update to obtain $y_{t-1}$
   - Add stochastic noise for $t > 1$
7. **Softmax** — Apply element-wise softmax to $y_0$ to obtain probability vectors $\hat{y}^{(n)} \in \Delta^{K-1}$.
8. **Reshape** — Rearrange the flat $(N \cdot P)$ samples back to shape $(N, P, K)$.
9. **Welch t-test** — For each pixel compute mean across $N$ samples, identify top-2 classes, run Welch t-test, record p-value and uncertainty flag.
10. **Output** — Assemble per-pixel arrays of predicted class, uncertain mask, p-values, and probability gap. Process in chunks to manage GPU memory.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_uncertainty_dapm_full.ipynb`

### 4.1 Custom Keras Layers (Section 5)

```python
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mu, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mu))
        return z_mu + tf.exp(0.5 * z_logvar) * eps
```

**What this does:** Implements the reparameterisation trick. Given the posterior mean `z_mu` and log-variance `z_logvar`, it draws $z = \mu_z + \exp(0.5 \cdot \log\sigma^2) \cdot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0,I)$.

**Why:** The `@register_keras_serializable` decorator ensures the layer can be saved and restored from the `.keras` model file, since it is a custom class not part of the standard Keras API. The other custom layers (`PatchExtractor`, `PatchPositionEncoder`, `GlobalFilterLayer`, `PatchEncoderWithCLS`) are likewise registered so the frozen base models can be deserialised.

---

### 4.2 DAPM Sub-Network Builders (Section 6)

```python
def get_feature_extractor(base_model):
    penultimate = base_model.layers[-2].output
    feat_model  = keras.Model(base_model.input, penultimate, ...)
    feat_model.trainable = False
    return feat_model

def build_dapm_encoder(feature_dim, latent_dim=64, hidden_dim=256):
    ...
    z_mu    = layers.Dense(latent_dim, name='z_mu')(h)
    z_logvar = layers.Dense(latent_dim, name='z_logvar')(h)
    z       = Sampling(name='z_sample')([z_mu, z_logvar])
    return keras.Model(inp, [z_mu, z_logvar, z], ...)
```

**What this does:** `get_feature_extractor` wraps the second-to-last layer of the base model as a frozen sub-model. `build_dapm_encoder` creates the VAE encoder: two 256-unit hidden layers followed by two parallel dense heads (mean and log-variance), then the `Sampling` layer.

**Why:** The base model's penultimate layer is a rich, task-adapted representation. Freezing it prevents fine-tuning from destroying what the base model learned; the VAE encoder then learns to *compress* this representation into a probabilistic latent space on top of it.

```python
def build_dapm_diffusion(latent_dim, num_classes, T=100, ...):
    ...
    t_emb  = layers.Embedding(input_dim=T + 1, output_dim=t_embed_dim, ...)(t_in)
    x      = layers.Concatenate()([z_in, y_t_in, f_in, t_emb_flat])
    ...
    eps_pred = layers.Dense(num_classes, activation='linear', ...)(x)
```

**What this does:** Builds the denoiser network. It concatenates the latent code $z$, the current noisy label $y_t$, the guidance vector $g$, and a learnable time embedding into a single vector, then passes it through two 256-unit hidden layers to predict the added noise $\varepsilon$.

**Why:** All four inputs must be presented jointly so the network can condition its noise prediction on both the input identity ($z$), the current denoising state ($y_t$), the soft guidance ($g$), and the diffusion step ($t$). The time step gets an embedding rather than a scalar because discrete step indices have no natural metric meaning as raw numbers.

---

### 4.3 Reverse Diffusion Sampler (Section 7.1)

```python
def reverse_diffusion(bundle, z_np, guidance_np):
    y = tf.random.normal((n, nc), dtype=tf.float32)   # start from noise

    for step in range(T, 0, -1):
        t_arr    = tf.cast(tf.fill((n, 1), step), tf.int32)
        eps_pred = _diffusion_step_compiled(diffusion, z_tf, y, guidance_tf, t_arr)

        alpha     = float(alphas[step - 1])
        alpha_bar = float(alpha_bars[step - 1])
        beta      = float(betas[step - 1])
        coef      = (1.0 - alpha) / max(np.sqrt(1.0 - alpha_bar), 1e-8)
        y         = (y - coef * eps_pred) / max(np.sqrt(alpha), 1e-8)

        if step > 1:
            noise = tf.random.normal(tf.shape(y), dtype=tf.float32)
            y     = y + np.sqrt(max(beta, 1e-8)) * noise

    return softmax_np(y.numpy(), axis=-1)
```

**What this does:** Implements the DDPM reverse diffusion loop. Starting from pure Gaussian noise in label space, it applies $T$ denoising steps, each time predicting the noise $\varepsilon$ and subtracting a scaled version of it, then re-adding a smaller stochastic perturbation for all non-final steps.

**Why:** This is the standard DDPM reverse process. The `@tf.function` decorator on `_diffusion_step_compiled` triggers XLA tracing on the first call, dramatically accelerating the tight inner loop. The numerically stable softmax is applied at the end because the raw network output $y_0$ is an unconstrained real vector.

---

### 4.4 Chunked DAPM Sampler (Section 7.1)

```python
def sample_dapm_chunk(bundle, x_chunk, n_samples, batch_size):
    feat              = fe(x_tf, training=False)
    z_mu, z_logvar, _ = enc(feat, training=False)
    guidance_np       = clf(z_mu, training=False).numpy()

    z_mu_tiled     = np.tile(z_mu_np, (n_samples, 1))
    std_tiled      = np.tile(std_np, (n_samples, 1))
    eps            = np.random.normal(size=z_mu_tiled.shape)
    z_all          = z_mu_tiled + std_tiled * eps        # N independent samples

    probs_flat = reverse_diffusion(bundle, z_all, guidance_tiled)
    return probs_flat.reshape(n_samples, n_points, nc)
```

**What this does:** For each chunk of pixels, (1) extracts features via the frozen base model, (2) encodes to latent statistics, (3) tiles the statistics $N$ times and samples $N$ independent latent codes, (4) runs reverse diffusion on the whole tiled batch at once, (5) reshapes the flat output to $(N, \text{pixels}, K)$.

**Why:** Processing the scene in chunks of 1000 pixels at a time avoids out-of-memory errors. Tiling and running all $N$ samples simultaneously exploits GPU parallelism far more efficiently than calling the denoiser $N$ times in sequence.

---

### 4.5 Welch T-Test Uncertainty (Section 7.1)

```python
def compute_dapm_ttest_uncertainty_chunk(probs_samples, p_thresh=0.05):
    mean_prob = np.mean(probs_samples, axis=0)
    order     = np.argsort(-mean_prob, axis=1)   # descending rank

    for i in range(n_points):
        c1, c2 = int(order[i, 0]), int(order[i, 1])
        g1, g2 = probs_samples[:, i, c1], probs_samples[:, i, c2]
        _, pval = _safe_ttest_ind(g1, g2)
        uncertain_mask[i] = bool(pval > p_thresh)
```

**What this does:** Computes the mean class probability across $N$ samples, finds the two top-ranked classes, performs a Welch t-test between the distributions of those two classes' probabilities across $N$ samples, and flags the pixel as uncertain if $p > 0.05$.

**Why:** The Welch test is non-parametric with respect to variance assumptions and appropriate here because the top-1 and top-2 class probability distributions can have quite different spreads (e.g., top-1 may be concentrated near 0.8 while top-2 is spread between 0.1 and 0.4). The `_safe_ttest_ind` wrapper handles degenerate cases where one or both distributions are constant (e.g., all predictions agree on the same class), returning $p=1.0$ for exactly equal constants and $p=0.0$ for distinct constants.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

We illustrate DAPM on a simplified scene with **$K = 4$ classes** and $N = 5$ stochastic samples (rather than the full 30 used in the notebook) to keep the arithmetic tractable. We use $\alpha_{\mathrm{ttest}} = 0.05$.

**Test samples:**

| Sample | Description | True class |
|--------|-------------|-----------|
| T1 | **Easy** — patch clearly in class 0 | 0 |
| T2 | **Borderline** — patch on boundary between class 0 and class 1 | 0 |
| T3 | **Ambiguous** — patch has no dominant class | 2 |

---

### 5.1 Calibration (Latent Encoding)

For each test pixel, DAPM's VAE encoder produces $(\mu_z, \sigma_z)$, and $N = 5$ latent codes are sampled:

$$z^{(n)} = \mu_z + \sigma_z \odot \varepsilon^{(n)}$$

For the purpose of this example, we skip the VAE arithmetic and directly give the 5 probability vectors $\hat{y}^{(n)}$ produced by the full diffusion chain, as this is what the Welch test operates on.

---

### 5.2 Method: DAPM — Stochastic Probability Draws

#### Step A — 5 probability draws for each test pixel

**Test pixel T1 (Easy, true class = 0):**

| Draw $n$ | Class 0 | Class 1 | Class 2 | Class 3 |
|----------|---------|---------|---------|---------|
| 1 | **0.81** | 0.09 | 0.06 | 0.04 |
| 2 | **0.78** | 0.11 | 0.07 | 0.04 |
| 3 | **0.83** | 0.08 | 0.06 | 0.03 |
| 4 | **0.80** | 0.10 | 0.07 | 0.03 |
| 5 | **0.79** | 0.10 | 0.07 | 0.04 |

Mean: Class 0 = **0.802**, Class 1 = 0.096, Class 2 = 0.066, Class 3 = 0.036  
Top-1 = class 0, Top-2 = class 1

**Test pixel T2 (Borderline, true class = 0):**

| Draw $n$ | Class 0 | Class 1 | Class 2 | Class 3 |
|----------|---------|---------|---------|---------|
| 1 | **0.52** | 0.38 | 0.07 | 0.03 |
| 2 | 0.43 | **0.46** | 0.08 | 0.03 |
| 3 | **0.55** | 0.35 | 0.07 | 0.03 |
| 4 | 0.44 | **0.45** | 0.08 | 0.03 |
| 5 | **0.48** | 0.41 | 0.07 | 0.04 |

Mean: Class 0 = **0.484**, Class 1 = 0.410, Class 2 = 0.074, Class 3 = 0.032  
Top-1 = class 0, Top-2 = class 1

**Test pixel T3 (Ambiguous, true class = 2):**

| Draw $n$ | Class 0 | Class 1 | Class 2 | Class 3 |
|----------|---------|---------|---------|---------|
| 1 | 0.28 | 0.30 | **0.32** | 0.10 |
| 2 | **0.35** | 0.28 | 0.29 | 0.08 |
| 3 | 0.25 | 0.29 | **0.36** | 0.10 |
| 4 | 0.31 | **0.32** | 0.27 | 0.10 |
| 5 | 0.28 | 0.29 | **0.31** | 0.12 |

Mean: Class 0 = 0.294, Class 1 = **0.296**, Class 2 = 0.310, Class 3 = 0.100  
Top-1 = class 2, Top-2 = class 1

---

#### Step B — Welch t-test for each pixel

**T1 (Easy):**

Group 1 (class 0 probs): `[0.81, 0.78, 0.83, 0.80, 0.79]`  
Group 2 (class 1 probs): `[0.09, 0.11, 0.08, 0.10, 0.10]`

$\bar{g}_1 = 0.802, \quad \bar{g}_2 = 0.096$

$s_1^2 = \frac{(0.008)^2 + (0.022)^2 + (0.028)^2 + (0.002)^2 + (0.012)^2}{4} \approx 0.000315$

$s_2^2 \approx 0.000130$

$t = \frac{0.802 - 0.096}{\sqrt{0.000315/5 + 0.000130/5}} = \frac{0.706}{\sqrt{0.0000890}} = \frac{0.706}{0.00943} \approx 74.9$

$p \approx 4 \times 10^{-8} \ll 0.05 \Rightarrow$ **CERTAIN** ✓ (true class 0 correctly predicted)

---

**T2 (Borderline):**

Group 1 (class 0 probs): `[0.52, 0.43, 0.55, 0.44, 0.48]`  
Group 2 (class 1 probs): `[0.38, 0.46, 0.35, 0.45, 0.41]`

$\bar{g}_1 = 0.484, \quad \bar{g}_2 = 0.410$

$s_1^2 \approx \frac{(0.036)^2 + (0.054)^2 + (0.066)^2 + (0.044)^2 + (0.004)^2}{4} \approx 0.00255$

$s_2^2 \approx \frac{(0.030)^2 + (0.050)^2 + (0.060)^2 + (0.040)^2 + (0.000)^2}{4} \approx 0.00185$

$t = \frac{0.484 - 0.410}{\sqrt{0.00255/5 + 0.00185/5}} = \frac{0.074}{\sqrt{0.000880}} = \frac{0.074}{0.02966} \approx 2.49$

With $\nu \approx 7$ effective degrees of freedom, $p \approx 0.041 < 0.05 \Rightarrow$ **CERTAIN** ✓  
(Prediction = class 0, true class = 0; marginal certainty due to competing class 1)

*Note: if samples were even more balanced, $p$ could exceed 0.05 and the pixel would be flagged uncertain.*

---

**T3 (Ambiguous):**

Group 1 (class 2 probs): `[0.32, 0.29, 0.36, 0.27, 0.31]`  
Group 2 (class 1 probs): `[0.30, 0.28, 0.29, 0.32, 0.29]`

$\bar{g}_1 = 0.310, \quad \bar{g}_2 = 0.296$

$s_1^2 \approx 0.000945, \quad s_2^2 \approx 0.000230$

$t = \frac{0.310 - 0.296}{\sqrt{0.000945/5 + 0.000230/5}} = \frac{0.014}{\sqrt{0.000235}} = \frac{0.014}{0.01533} \approx 0.913$

With $\nu \approx 6$ degrees of freedom, $p \approx 0.40 > 0.05 \Rightarrow$ **UNCERTAIN** (flagged correctly — no class dominates across draws)

---

#### Step C — Summary table

| Test pixel | Prediction | True class | P-value | Certain? | Coverage |
|-----------|-----------|-----------|---------|---------|---------|
| T1 (Easy) | Class 0 | 0 | ~0.000 | ✓ Yes | ✓ |
| T2 (Borderline) | Class 0 | 0 | ~0.041 | ✓ Yes | ✓ |
| T3 (Ambiguous) | Class 2 | 2 | ~0.40 | ✗ No (uncertain) | ✓ |

**Interpretation:** DAPM correctly identifies T3 as uncertain (flagged for review), while T1 and T2 are retained with prediction class 0. The key insight is that T3's diffusion chain oscillates between classes 1 and 2 across different draws, reflecting genuine label ambiguity; the t-test detects this as a statistically insignificant difference between the top-two class distributions.

---

### 5.3 Observed Results from the Notebook

The notebook ran DAPM on a real 330×307 multispectral scene with 7 classes across three models:

| Model | Overall Accuracy | Uncertain Rate | Mean P-Value |
|-------|-----------------|---------------|-------------|
| AlexNet | 99.92% | 0.25% | 0.0011 |
| GFNet | 99.90% | 0.16% | 0.00065 |
| ViT | 99.79% | 0.09% | 0.00037 |

Key observations:
- **Very few pixels are uncertain** — less than 0.3% of the scene for any model. The t-test threshold of $p = 0.05$ is strict: only genuinely ambiguous pixels (where the distribution of top-1 and top-2 probabilities across 30 samples genuinely overlaps) are flagged.
- **ViT has the lowest uncertainty rate** — despite having slightly lower overall accuracy than AlexNet. This suggests ViT's predictions, when correct, are more consistently confident across different latent draws.
- **Overall accuracy on labelled pixels exceeds 99.7%** for all models, validating that the DAPM backbone achieves strong discriminative performance while adding calibrated uncertainty.

---

## 6. References

[1] Du, Zhekai and Li, Jingjing. "Diffusion-Based Probabilistic Uncertainty Estimation for Active Domain Adaptation." *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. [https://proceedings.neurips.cc/paper_files/paper/2023/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html)

[2] Kingma, Diederik P. and Welling, Max. "Auto-Encoding Variational Bayes." *arXiv preprint arXiv:1312.6114*, 2013. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

[3] Ho, Jonathan, Jain, Ajay, and Abbeel, Pieter. "Denoising Diffusion Probabilistic Models." *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

[4] Welch, Bernard Lewis. "The Generalization of Student's Problem when Several Different Population Variances are Involved." *Biometrika*, 34(1–2), 28–35, 1947.

[5] GitHub repository for DAPM: [https://github.com/TL-UESTC/DAPM](https://github.com/TL-UESTC/DAPM)
