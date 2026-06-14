# DAPM (Diffusion-Augmented Probabilistic Model): Theory & Implementation Summary

> **One-line description:** A two-stage framework that trains a VAE encoder with domain-adversarial alignment and a conditional diffusion model over class-label distributions, then exploits the resulting stochastic inference pipeline to produce calibrated per-pixel uncertainty estimates via Welch t-test on repeated diffusion draws.

---

## 1. Overview & Intuition

Standard deep classifiers produce a single point-estimate prediction per input. Under domain shift — where a model trained on labelled pixels (the *source* domain) is deployed on unlabelled pixels from the same scene with a different appearance distribution (the *target* domain) — softmax scores become miscalibrated: the model can be confidently wrong and there is no reliable signal to detect it.

DAPM addresses this by jointly modelling two kinds of uncertainty. **Data-level uncertainty** captures variability in *how* an input is encoded: a VAE encoder maps each input to a Gaussian distribution in latent space, so nearby-boundary samples produce spread-out latent distributions. **Prediction-level uncertainty** captures variability in the output: instead of a deterministic classifier head, DAPM trains a conditional Denoising Diffusion Probabilistic Model (DDPM) that operates in the *label simplex* — running a reverse diffusion chain from noise to a plausible class-probability vector, conditioned on the latent code and soft classifier guidance.

At inference time, drawing N independent samples from both sources of randomness (the latent posterior and the diffusion chain) yields an empirical distribution of prediction vectors for each pixel. A Welch two-sample t-test then compares the top-1 and top-2 class probability streams across those N draws: if they are statistically indistinguishable (large p-value), the pixel is *uncertain*; if clearly separated (small p-value), it is *certain*.

The system is trained in two sequential stages. Stage 1 jointly trains the VAE encoder, source and target decoders, a softmax classifier, and a domain discriminator whose gradients are reversed to push source and target latent distributions into alignment. Stage 2 freezes those components and trains the conditional diffusion model to predict the noise added to label vectors, using real labels for source pixels and the classifier's soft predictions as pseudo-labels for target pixels.

The two uploaded notebooks implement this end-to-end: `Model_training_dapm_full.ipynb` covers Stages 1 and 2 for three backbone architectures (AlexNet_CNN, GFNet, ViT_UNet) on a 6-band multispectral remote-sensing image; `Model_uncertainty_dapm_full.ipynb` loads the saved weights and runs the full-scene stochastic inference and uncertainty mapping pipeline.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let `D_s = {(x_i^s, y_i^s)}` be labelled source patches and `D_t = {x_j^t}` be unlabelled target patches. A frozen backbone `φ(·)` maps each 9×9×6 patch to a feature vector `f ∈ ℝ^d`. The goal is to learn a distribution over class labels `y ∈ {0, …, C−1}` for any input, sharing a stochastic latent space `z ∈ ℝ^{d_z}` across both domains (d_z = 64).

### 2.2 VAE Encoder and Reparameterization

The encoder maps a feature vector to a distribution in latent space:

```
z_μ, z_logσ² = Encoder(f)

z = z_μ + exp(½ · z_logσ²) · ε,    ε ~ N(0, I)
```

| Symbol | Meaning |
|--------|---------|
| `z_μ ∈ ℝ^{d_z}` | Posterior mean |
| `z_logσ² ∈ ℝ^{d_z}` | Log-variance of the posterior |
| `ε` | Standard Gaussian noise (reparameterization enables backpropagation through sampling) |

### 2.3 KL Divergence Loss

The posterior is regularised towards a standard Gaussian prior:

```
L_KL = −½ · E[ Σ_{j=1}^{d_z} ( 1 + z_logσ²_j − z_μ_j² − exp(z_logσ²_j) ) ]
```

### 2.4 Reconstruction Loss

Separate decoders `Dec_s` and `Dec_t` reconstruct the original backbone feature from the latent sample, allowing domain-specific reconstruction while the encoder learns a shared space:

```
L_recon(f, f̂) = E[ ‖f − f̂‖² ]
```

### 2.5 Gradient Reversal Layer and Domain Adversarial Loss

A domain discriminator `D` classifies whether `z` came from the source or target. During the backward pass through the encoder, gradients are reversed by a factor `−λ_GRL`:

```
∂L_domain/∂z |_encoder  =  −λ_GRL · ∂L_domain/∂z

L_domain = BCE(0, D(z_s))  +  BCE(1, D(z_t))
```

**What this means:** The GRL forces the encoder to fool the discriminator by making source and target latents indistinguishable — domain alignment via adversarial training, without alternating optimisation steps.

### 2.6 Stage 1 Total Loss

```
L_1 = λ_src · L_recon^s  +  λ_tgt · L_recon^t
    + λ_KL · ( L_KL^s + L_KL^t )
    + λ_CE · L_CE
    + λ_dom · L_domain
```

Default weights: `λ_src = λ_tgt = 1.0`, `λ_KL = 0.01`, `λ_CE = 1.0`, `λ_dom = 0.2`.

### 2.7 Forward Diffusion Process (Stage 2)

The diffusion model operates over label vectors `y_0 ∈ Δ^{C-1}` (one-hot encodings). A linear beta schedule defines the forward noising process:

```
β_t = β_start + (t−1)/(T−1) · (β_end − β_start)

ᾱ_t = Π_{s=1}^{t} (1 − β_s)
```

The closed-form single-step sample at any timestep t is:

```
q(y_t | y_0):   y_t = √ᾱ_t · y_0 + √(1−ᾱ_t) · ε,    ε ~ N(0, I)
```

| Symbol | Value / Meaning |
|--------|----------------|
| `T` | 100 diffusion steps |
| `β_start` | 1e-4 |
| `β_end` | 2e-2 |
| `t = 0` | Clean label |
| `t = T` | Approximately pure Gaussian noise |

### 2.8 Conditional Diffusion Denoiser

A neural network `ε_θ` is trained to predict the noise `ε` added at timestep t, conditioned on the latent code z, noisy label `y_t`, soft guidance g, and a learned timestep embedding:

```
L_diff = E_{y_0, t, ε} [ ‖ε − ε_θ(z, y_t, g, t)‖² ]
```

For unlabelled target pixels, the classifier's soft predictions serve as pseudo-labels:

```
L_2 = L_diff^s  +  λ_tgt_diff · L_diff^t,    λ_tgt_diff = 0.5
```

### 2.9 Stochastic Inference via Multiple Diffusion Draws

At inference, for each input pixel, N = 30 independent latent codes are sampled from the encoder posterior:

```
z^(k) = z_μ + exp(½ · z_logσ²) · ε^(k),    k = 1, …, N
```

For each `z^(k)`, the full reverse diffusion chain runs from `y_T ~ N(0, I)`:

```
y_{t-1} = (1/√α_t) · ( y_t − (1−α_t)/√(1−ᾱ_t) · ε_θ(z^(k), y_t, g, t) )
         + √β_t · η_t,    η_t ~ N(0, I)
```

The stochastic term is dropped at the final step (t = 1). Each completed chain yields a softmax-normalised probability vector `p̂^(k) = softmax(y_0^(k))`, giving an empirical distribution `{p̂^(k)}_{k=1}^N` per pixel.

### 2.10 Uncertainty Estimation via Welch t-Test

From the N probability vectors, identify the mean top-1 and top-2 class indices:

```
c_1 = argmax_c  p̄_c
c_2 = argmax_{c ≠ c_1}  p̄_c
p̄_c = (1/N) · Σ_k  p̂_c^(k)
```

Extract two groups of scalar samples:

```
G_1 = { p̂_{c_1}^(k) }_{k=1}^N        (N top-1 probabilities)
G_2 = { p̂_{c_2}^(k) }_{k=1}^N        (N top-2 probabilities)
```

Apply the Welch two-sample t-test (unequal variance):

```
t = ( Ḡ_1 − Ḡ_2 ) / √( s_1²/N + s_2²/N )
```

A pixel is marked **uncertain** when:

```
p-value(t) > τ,    τ = 0.05
```

**What this means:** A large p-value indicates the top-1 and top-2 class probability distributions are statistically indistinguishable across the N draws — the model is genuinely unsure which class is correct. This is more powerful than single-prediction entropy or margin because it reasons about the full distribution of N stochastic draws, capturing both latent and diffusion stochasticity.

---

## 3. Full Algorithm

### Stage 1 — VAE + Domain Adversarial Training (20 epochs)

**Input:** Source patches `(x_s, y_s)`, target patches `x_t`, frozen backbone `φ`

1. Compute backbone features: `f_s = φ(x_s)`, `f_t = φ(x_t)`
2. Encode to stochastic latents: `(z_μ^s, z_logσ²^s, z^s) = Encoder(f_s)` and similarly for target
3. Reconstruct features: `f̂_s = Dec_s(z^s)`, `f̂_t = Dec_t(z^t)`
4. Compute source class predictions: `p̂_s = Classifier(z^s)`
5. Compute domain predictions via discriminator (GRL reverses gradients to encoder): `D(z^s)`, `D(z^t)`
6. Compute `L_1` and update all Stage 1 weights via Adam

### Stage 2 — Conditional Diffusion Training (20 epochs)

7. Freeze encoder and classifier from Stage 1
8. Construct one-hot source labels `y_0^s`; use `stop_gradient(Classifier(z_μ^t))` as `y_0^t`
9. Sample random timestep `t ∈ {1, …, T}`; apply forward diffusion to get `y_t`, `ε`
10. Predict noise: `ε̂ = ε_θ(z, y_t, g, t)`
11. Compute `L_2` and update diffusion model via Adam

### Inference — Stochastic Uncertainty Mapping

12. Process all H×W pixels in chunks of 1000
13. For each chunk: encode once, tile N = 30 times, sample N latent codes
14. Run full reverse diffusion chain (T = 100 steps) for all N·n samples in one batched call
15. Softmax outputs → empirical distribution `{p̂^(k)}` per pixel
16. Welch t-test on top-1 vs top-2 groups → p-value and uncertainty flag per pixel
17. Reshape results to (H, W) maps and export

---

## 4. Implementation Walkthrough

> Based on: `Model_training_dapm_full.ipynb` (Sections 4.1–4.6) and `Model_uncertainty_dapm_full.ipynb` (Sections 4.7–4.9)

### 4.1 Sampling Layer and Gradient Reversal Layer

```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mu, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mu))
        return z_mu + tf.exp(0.5 * z_logvar) * eps

class GradientReversal(layers.Layer):
    def call(self, x):
        @tf.custom_gradient
        def _flip_gradients(v):
            def grad(dy): return -lambda_ * dy
            return v, grad
        return _flip_gradients(x)
```

**What this does:** `Sampling` implements the reparameterisation trick as a Keras layer, allowing gradients to flow through the sampling operation. `GradientReversal` uses `tf.custom_gradient` to negate gradients in-graph during backpropagation.

**Why:** Both must be custom Keras layers to serialise correctly when saving weights, and to integrate cleanly into a single `GradientTape` training step.

### 4.2 DAPM Encoder

```python
def build_dapm_encoder(feature_dim, latent_dim=64, hidden_dim=256):
    inp      = keras.Input(shape=(feature_dim,))
    h        = layers.Dense(hidden_dim, activation='relu')(inp)
    h        = layers.Dense(hidden_dim, activation='relu')(h)
    z_mu     = layers.Dense(latent_dim)(h)
    z_logvar = layers.Dense(latent_dim)(h)
    z        = Sampling()([z_mu, z_logvar])
    return keras.Model(inp, [z_mu, z_logvar, z])
```

**What this does:** Two-hidden-layer MLP outputting posterior mean, log-variance, and a reparameterised sample.

**Why:** All three outputs are returned so the training loop can use `z_μ` for classifier guidance (lower variance) while using the stochastic `z` for reconstruction and adversarial losses.

### 4.3 Diffusion Model Builder

```python
def build_dapm_diffusion(latent_dim, num_classes, T=100, t_embed_dim=32, hidden_dim=256):
    z_in   = keras.Input(shape=(latent_dim,))
    y_t_in = keras.Input(shape=(num_classes,))
    f_in   = keras.Input(shape=(num_classes,))          # classifier guidance
    t_in   = keras.Input(shape=(1,), dtype='int32')
    t_emb  = layers.Embedding(T+1, t_embed_dim)(t_in)
    t_emb  = layers.Flatten()(t_emb)
    x      = layers.Concatenate()([z_in, y_t_in, f_in, t_emb])
    x      = layers.Dense(hidden_dim, activation='relu')(x)
    x      = layers.Dense(hidden_dim, activation='relu')(x)
    return keras.Model([z_in, y_t_in, f_in, t_in],
                       layers.Dense(num_classes)(x))
```

**What this does:** Four-input network that concatenates the latent code, noisy label, classifier guidance, and a learned timestep embedding before two dense layers and a linear output.

**Why:** The timestep is embedded (not passed as a raw scalar) so the network can learn smooth, timestep-specific denoising behaviour. Classifier guidance injects the model's current best label estimate as a conditioning signal.

### 4.4 Forward Diffusion and Beta Schedule

```python
def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    betas      = np.linspace(beta_start, beta_end, T)
    alpha_bars = np.cumprod(1.0 - betas)
    return betas, 1.0 - betas, alpha_bars

def q_sample(y0, t_idx, alpha_bars):
    a_bar = tf.gather(alpha_bars, tf.squeeze(t_idx - 1, -1))
    a_bar = tf.reshape(a_bar, (-1, 1))
    eps   = tf.random.normal(tf.shape(y0))
    return tf.sqrt(a_bar) * y0 + tf.sqrt(1.0 - a_bar) * eps, eps
```

**What this does:** Pre-computes the full noise schedule, then applies the closed-form `q(y_t | y_0)` formula to corrupt any label vector at any timestep in a single operation.

**Why:** The closed form avoids iterating through t sequential steps during training, enabling efficient random timestep sampling across the full schedule every batch.

### 4.5 Stage 1 Training Step

```python
with tf.GradientTape() as tape:
    feat_src = feature_extractor(xb_src, training=False)   # backbone frozen
    z_mu_src, z_logvar_src, z_src = encoder(feat_src, training=True)
    feat_src_rec = src_decoder(z_src, training=True)
    y_src_prob   = classifier(z_src, training=True)
    dom_src_prob = discriminator(z_src, training=True)      # GRL inside
    ...
    loss = (LAMBDA_SRC_RECON * src_recon + LAMBDA_TGT_RECON * tgt_recon
          + LAMBDA_KL * (src_kl + tgt_kl)
          + LAMBDA_CE * src_ce + LAMBDA_DOMAIN * dom_loss)
grads = tape.gradient(loss, stage1_vars)
opt.apply_gradients(zip(grads, stage1_vars))
```

**What this does:** A single joint gradient update over all Stage 1 parameters within one `GradientTape` context.

**Why:** The GRL inside the discriminator means the domain loss, when backpropagated, pushes the encoder to confuse the discriminator (domain alignment), while the discriminator's own weights are updated normally.

### 4.6 Stage 2 Training Step

```python
y0_src = tf.one_hot(yb_src, depth=num_classes)
y0_tgt = tf.stop_gradient(y_guidance_tgt)            # pseudo-labels
t      = sample_timesteps(batch_size, DIFFUSION_T)
y_t, eps = q_sample(y0_src, t, alpha_bars)
eps_pred = diffusion([z_src, y_t, y_guidance_src, t])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(eps - eps_pred), axis=-1))
```

**What this does:** Samples a random timestep, corrupts the label vector, predicts the noise, and minimises MSE between true and predicted noise — the simplified DDPM objective.

**Why:** `tf.stop_gradient` on both guidance and pseudo-labels prevents gradients from flowing back into the frozen classifier during Stage 2.

### 4.7 Reverse Diffusion (Inference)

```python
@tf.function(reduce_retracing=True)
def _diffusion_step_compiled(diffusion_model, z_tf, y_tf, guidance_tf, t_tf):
    return diffusion_model([z_tf, y_tf, guidance_tf, t_tf], training=False)

def reverse_diffusion(bundle, z_np, guidance_np):
    y = tf.random.normal((n, nc))
    for step in range(T, 0, -1):
        eps_pred = _diffusion_step_compiled(diffusion, z_tf, y, guidance_tf,
                                            tf.fill((n,1), step))
        coef = (1.0 - alpha) / sqrt(1.0 - alpha_bar)
        y    = (y - coef * eps_pred) / sqrt(alpha)
        if step > 1:
            y = y + sqrt(beta) * tf.random.normal(tf.shape(y))
    return softmax_np(y.numpy())
```

**What this does:** Runs the full 100-step reverse DDPM chain from Gaussian noise to a label distribution, with the inner step compiled by `@tf.function` for speed.

**Why:** XLA compilation of the single step reduces Python overhead dramatically when looped 100 times. The final step is deterministic (no noise term) to produce a clean output.

### 4.8 Batched Tiling for N Samples

```python
z_mu_tiled = np.tile(z_mu_np, (n_samples, 1))   # shape: (N*n, d_z)
std_tiled  = np.tile(std_np,  (n_samples, 1))
z_all      = z_mu_tiled + std_tiled * np.random.normal(size=z_mu_tiled.shape)
probs_flat = reverse_diffusion(bundle, z_all, guidance_tiled)
probs      = probs_flat.reshape(n_samples, n_points, nc)
```

**What this does:** Encodes the chunk once, tiles the latent statistics N times, draws N different noise vectors, and runs one large reverse diffusion call to get all N×n probability vectors at once.

**Why:** GPU throughput is far better with one large batch than N sequential small batches.

### 4.9 Welch t-Test Uncertainty Estimator

```python
for i in range(n_points):
    c1, c2 = order[i, 0], order[i, 1]          # top-1 and top-2 classes
    g1 = probs_samples[:, i, c1]                # N top-1 probabilities
    g2 = probs_samples[:, i, c2]                # N top-2 probabilities
    _, pval = ttest_ind(g1, g2, equal_var=False)
    uncertain_mask[i] = bool(pval > p_thresh)
```

**What this does:** For each pixel, compares the N top-1 and top-2 probability streams via Welch's t-test and flags the pixel uncertain when the two distributions are not statistically separated.

**Why:** Welch's variant (unequal variance) is appropriate here because the top-1 distribution is typically tighter than the top-2 distribution. Using all N draws rather than a single prediction makes the test sensitive to genuine distributional ambiguity, not just softmax sharpness.

---

## 5. Worked Numerical Example

**Setup:** C = 3 classes, d_z = 4, T = 5 diffusion steps, N = 5 inference draws.

### Training: Forward Diffusion

Beta schedule:

```
β     = [0.10, 0.20, 0.30, 0.40, 0.50]
α     = [0.90, 0.80, 0.70, 0.60, 0.50]
ᾱ     = [0.90, 0.72, 0.50, 0.30, 0.15]
```

Source label for class 0: `y_0 = [1, 0, 0]`. At timestep t = 3, `ᾱ_3 = 0.50`:

```
ε     = [−0.50,  0.90, −0.40]

y_3   = √0.50 · [1, 0, 0]  +  √0.50 · [−0.50, 0.90, −0.40]
      = [0.71 − 0.35,  0 + 0.64,  0 − 0.28]
      = [0.36,  0.64,  −0.28]
```

The diffusion network predicts `ε̂ = [−0.48, 0.88, −0.39]`. The training loss is:

```
L = (−0.50 − (−0.48))²  +  (0.90 − 0.88)²  +  (−0.40 − (−0.39))²
  = 0.0004  +  0.0004  +  0.0001
  = 0.0009
```

### Inference: Uncertainty Test

After N = 5 complete reverse diffusion draws for two pixels:

**Certain pixel** (true label: class 0):

| Draw | p̂_0 | p̂_1 | p̂_2 |
|:----:|:----:|:----:|:----:|
| 1 | 0.72 | 0.20 | 0.08 |
| 2 | 0.68 | 0.25 | 0.07 |
| 3 | 0.74 | 0.18 | 0.08 |
| 4 | 0.70 | 0.22 | 0.08 |
| 5 | 0.66 | 0.27 | 0.07 |

```
G_1 = [0.72, 0.68, 0.74, 0.70, 0.66]    (top-1 class 0)
G_2 = [0.20, 0.25, 0.18, 0.22, 0.27]    (top-2 class 1)

t ≈ 23.8,   p-value ≪ 0.001  <  0.05   →   CERTAIN (Class 0)
```

**Uncertain pixel** (near class boundary):

| Draw | p̂_0 | p̂_1 | p̂_2 |
|:----:|:----:|:----:|:----:|
| 1 | 0.48 | 0.42 | 0.10 |
| 2 | 0.45 | 0.46 | 0.09 |
| 3 | 0.52 | 0.38 | 0.10 |
| 4 | 0.44 | 0.47 | 0.09 |
| 5 | 0.50 | 0.41 | 0.09 |

```
G_1 = [0.48, 0.45, 0.52, 0.44, 0.50]    (top-1 class 0)
G_2 = [0.42, 0.46, 0.38, 0.47, 0.41]    (top-2 class 1)

t ≈ 2.4,   p-value ≈ 0.06  >  0.05   →   UNCERTAIN
```

The t-test correctly identifies the second pixel as ambiguous even though its argmax is still class 0 — a margin of 0.05 between top-1 and top-2 is flagged, while a margin of 0.48 is not.

---

## 6. References

[1] Zhekai Du and Jingjing Li. "Diffusion-Based Probabilistic Uncertainty Estimation for Active Domain Adaptation." *Advances in Neural Information Processing Systems (NeurIPS) 36*, 2023. [Link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html)

[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion Probabilistic Models." *NeurIPS 33*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

[3] Yaroslav Ganin and Victor Lempitsky. "Unsupervised Domain Adaptation by Backpropagation." *ICML*, 2015. *(Foundational work on gradient reversal for domain-adversarial training.)*

[4] Diederik P. Kingma and Max Welling. "Auto-Encoding Variational Bayes." *ICLR*, 2014. *(The VAE reparameterisation trick used in the encoder.)*

[5] B. L. Welch. "The Generalisation of Student's Problem when Several Different Population Variances are Involved." *Biometrika*, 34(1–2):28–35, 1947. *(Statistical basis of the t-test used for uncertainty thresholding.)*
