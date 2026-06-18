# DAPM: Theory & Implementation Summary

> **One-line description:** DAPM (as implemented in this notebook) is a two-stage, domain-adaptive deep model that learns a shared latent representation of frozen-backbone image-patch features via a VAE, aligns that representation across labeled and unlabeled pixels with adversarial (gradient-reversal) training, and additionally trains a conditional denoising-diffusion network on the *label distribution* to model class uncertainty in that latent space.

---

## 1. Overview & Intuition

The notebook tackles pixel-wise classification of a multispectral scene (six spectral bands) using small spatial patches (9×9 pixels) cut out around each pixel. Some pixels in the scene have ground-truth land-cover labels ("source domain"); the vast majority do not ("target domain"). Both domains come from the *same* image, but they are treated as two distributions that a model trained only on the labeled pixels might not generalize to equally well — the labeled pixels tend to be easier, more homogeneous regions, while the broader unlabeled area can contain spectral variability the labeled set never showed the model.

DAPM addresses this with three ideas working together. First, a pretrained, frozen backbone (one of three architectures: an AlexNet-style CNN, a GFNet "global filter" transformer-style network, or a ViT/U-Net hybrid) turns each raw patch into a feature vector — this part is fixed and not retrained. Second, a shared Variational Autoencoder (VAE) compresses that feature vector into a small latent code, and a domain discriminator equipped with a Gradient Reversal Layer (GRL) is trained adversarially against the encoder so that the latent code becomes hard to tell apart by domain — i.e., the encoder is pushed to produce *domain-invariant* representations, in the spirit of Domain-Adversarial Neural Networks. A classifier head sits on top of this shared latent space and is trained only with the labeled (source) examples. Third, once this shared representation exists, a conditional diffusion model is trained to model the distribution of class labels given the latent code, using the standard denoising-diffusion training recipe but applied to label vectors (one-hot for labeled pixels, the classifier's own soft prediction as a self-training pseudo-label for unlabeled pixels) instead of images.

The core insight is to combine three well-known but separately motivated mechanisms — variational encoding for compact informative features, adversarial domain alignment for invariance to where the patch came from, and diffusion-based generative modeling of the label distribution for a probabilistic view of class uncertainty — into one shared latent space, so that the encoder used for ordinary classification is also the conditioning signal for an explicit generative model of class probabilities. What makes this notebook's instantiation distinctive is that it does this on top of three interchangeable, completely frozen feature-extraction backbones, letting the team compare how well the same DAPM head adapts the same underlying features depending on which backbone produced them.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let the multispectral image be $X \in \mathbb{R}^{H \times W \times B}$ (here $H=330$, $W=307$, $B=6$ bands), with a per-pixel reference label map $Y$. For every pixel $(r,c)$ a patch of size $9 \times 9 \times B$ centered at $(r,c)$ is extracted (with edge padding). Pixels with $Y(r,c) > 0$ form the **source domain**

$$\mathcal{D}_s = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}, \qquad y_i^s \in \{0, \dots, K-1\}$$

and pixels with $Y(r,c) = 0$ form the **target domain** (subsampled for tractability)

$$\mathcal{D}_t = \{x_j^t\}_{j=1}^{n_t}$$

A frozen, pretrained backbone $f_\phi$ (one of AlexNet-style CNN, GFNet, or ViT/U-Net) maps a patch to a feature vector $h = f_\phi(x) \in \mathbb{R}^{d}$, where $d$ depends on the chosen backbone's penultimate layer width. DAPM then learns everything downstream of $h$: an encoder, two decoders, a classifier, a discriminator, and a diffusion model — never touching $f_\phi$'s weights (`FREEZE_BACKBONE = True`).

### 2.2 Shared VAE Encoder and Reparameterization

A single encoder $E_\theta$ is shared by both domains and maps a feature vector to the parameters of a latent Gaussian:

$$E_\theta(h) = (\mu, \log\sigma^2)$$

A sample is drawn with the reparameterization trick so gradients can flow through the sampling step:

$$z = \mu + \exp(0.5 \cdot \log\sigma^2) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

**Where:**
- $h$ — backbone feature vector for one patch
- $\mu, \log\sigma^2 \in \mathbb{R}^{L}$ — predicted mean and log-variance of the latent code ($L$ = latent dimension)
- $\epsilon$ — standard Gaussian noise sampled fresh every forward pass
- $z$ — the stochastic latent code, used as input to every downstream head

**What this means:** the encoder doesn't output a single point in latent space; it outputs a small Gaussian "cloud" for each input, and a different random draw from that cloud is used every time. This regularizes the latent space and is what makes the KL term below meaningful.

The encoder's regularization term pulls the latent distribution toward a standard normal prior:

$$\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2) \,\|\, \mathcal{N}(0, I)\big) = -\tfrac{1}{2}\sum_{k=1}^{L}\Big(1 + \log\sigma_k^2 - \mu_k^2 - \sigma_k^2\Big)$$

This is computed identically for source and target features (since the encoder is shared), giving $\mathrm{KL}_{src}$ and $\mathrm{KL}_{tgt}$.

### 2.3 Domain-Specific Decoders and Reconstruction

Although the encoder is shared, **two separate decoders** $D_s$ and $D_t$ reconstruct the feature vector from $z$, one used only for source-domain codes and one only for target-domain codes:

$$\hat{h}^s = D_s(z^s), \qquad \hat{h}^t = D_t(z^t)$$

$$\mathcal{L}_{recon}^{(\cdot)} = \frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{d} \big(h_{i,k} - \hat{h}_{i,k}\big)^2$$

**What this means:** the shared encoder is forced to keep enough information in $z$ to reconstruct the original feature vector, but the decoder is allowed to be domain-specific — this keeps the reconstruction task achievable even while the encoder is simultaneously being pushed toward domain-invariance (Section 2.4), since the asymmetry is absorbed by the decoders rather than by $z$ itself.

### 2.4 Domain-Adversarial Alignment via the Gradient Reversal Layer

A domain discriminator $G_\psi$ tries to predict, from the latent code alone, whether $z$ came from the source or target domain:

$$p_{dom} = G_\psi(z) \in (0,1) \quad (\text{sigmoid output})$$

Between $z$ and $G_\psi$ sits a **Gradient Reversal Layer (GRL)**. In the forward pass it is the identity function; in the backward pass it negates and scales the incoming gradient:

$$\text{forward: } \mathrm{GRL}(z) = z \qquad\qquad \text{backward: } \frac{\partial \mathcal{L}}{\partial z} \leftarrow -\lambda \cdot \frac{\partial \mathcal{L}_{dom}}{\partial \mathrm{GRL}(z)}$$

**Where:**
- $\lambda$ — the GRL strength (set to $1.0$ in this notebook)
- $\mathcal{L}_{dom}$ — the domain binary cross-entropy below

The domain loss itself is ordinary BCE, with source patches labeled $0$ and target patches labeled $1$:

$$\mathcal{L}_{dom} = \mathrm{BCE}\big(0,\, G_\psi(z^s)\big) + \mathrm{BCE}\big(1,\, G_\psi(z^t)\big)$$

**What this means:** the discriminator is trained normally to get *better* at telling domains apart, but because its gradient is flipped before it reaches the encoder, the encoder is pushed in the *opposite* direction — to make $z^s$ and $z^t$ look as similar as possible to the discriminator. This single mechanism is what turns ordinary feature learning into *domain-adversarial* feature learning (Ganin & Lempitsky, 2015).

### 2.5 Classifier and the Stage-1 Objective

A classifier $C_\omega$ predicts class probabilities from the latent code, trained only on source (labeled) samples:

$$\hat{y}^s = C_\omega(z^s), \qquad \mathcal{L}_{CE} = -\frac{1}{n}\sum_{i=1}^n \log \hat{y}^s_{i, y_i}$$

All five pieces are combined into the **Stage-1 training objective**:

$$\mathcal{L}_1 = \lambda_{src}\,\mathcal{L}_{recon}^{s} + \lambda_{tgt}\,\mathcal{L}_{recon}^{t} + \lambda_{KL}\big(\mathrm{KL}_{src} + \mathrm{KL}_{tgt}\big) + \lambda_{CE}\,\mathcal{L}_{CE} + \lambda_{dom}\,\mathcal{L}_{dom}$$

with weights set in the notebook to $\lambda_{src}=\lambda_{tgt}=\lambda_{CE}=1.0$, $\lambda_{KL}=0.01$, $\lambda_{dom}=0.2$.

### 2.6 Conditional Diffusion over the Label Distribution

Stage 2 freezes the encoder and classifier from Stage 1 and trains a separate conditional diffusion model $\epsilon_\psi$ whose job is to predict the Gaussian noise added to a **label vector** $y_0 \in \mathbb{R}^K$ at a random timestep, conditioned on the latent code and the classifier's own soft prediction.

A linear noise schedule is precomputed once:

$$\beta_t = \text{linspace}(\beta_{start}, \beta_{end}, T), \qquad \alpha_t = 1-\beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s$$

**Forward diffusion** (closed form, no iteration needed at training time):

$$q(y_t \mid y_0) = \mathcal{N}\big(y_t;\ \sqrt{\bar\alpha_t}\, y_0,\ (1-\bar\alpha_t) I\big) \quad\Longrightarrow\quad y_t = \sqrt{\bar\alpha_t}\,y_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ \ \epsilon\sim\mathcal{N}(0,I)$$

**Where:**
- $y_0$ — the *clean* target: a one-hot class vector for source samples, but for target samples the (stop-gradiented) classifier's own softmax output — a pseudo-label, since no ground truth exists
- $t$ — a timestep sampled uniformly from $\{1,\dots,T\}$ ($T=100$ in the notebook)
- $\bar\alpha_t$ — cumulative signal-retention factor; small $\bar\alpha_t$ (later $t$) means $y_t$ is mostly noise
- $\epsilon$ — the noise the network must learn to recover

The noise predictor itself is conditioned on four things:

$$\hat\epsilon = \epsilon_\psi\big(z,\, y_t,\, f_{guid},\, t\big), \qquad f_{guid} = \mathrm{stopgrad}\big(C_\omega(\mu)\big)$$

**Where:**
- $z$ — the (frozen-encoder) latent code for this patch
- $y_t$ — the noised label vector at timestep $t$
- $f_{guid}$ — the classifier's softmax prediction at the latent *mean* $\mu$, used purely as an extra conditioning signal (akin to classifier guidance), with gradients blocked so it doesn't get updated through the diffusion loss
- $t$ — the timestep, passed through a learned embedding

**What this means:** rather than denoising images, this diffusion model denoises *label distributions*, learning how class probability vectors are statistically structured around a given latent code and classifier guess. This is a generative model of "what label vector looks plausible here," trained with the same simplified noise-prediction loss as image diffusion models:

$$\mathcal{L}_{diff} = \mathbb{E}\big[\|\epsilon - \hat\epsilon\|_2^2\big]$$

The full **Stage-2 objective** combines source and target diffusion losses, with the target term down-weighted because its pseudo-labels are noisier:

$$\mathcal{L}_2 = \mathcal{L}_{diff}^{s} + \lambda_{tgt\_diff}\cdot \mathcal{L}_{diff}^{t}, \qquad \lambda_{tgt\_diff} = 0.5$$

---

## 3. Algorithm

**Input:** multispectral image $X$, reference labels $Y$, three frozen backbones $f_\phi$, hyperparameters ($L$, $T$, $\beta_{start}$, $\beta_{end}$, loss weights, epoch counts)
**Output:** per-backbone DAPM artifacts (encoder, source/target decoders, classifier, discriminator, diffusion network weights + config) and test-set classification metrics

For each backbone in `{AlexNet_CNN, GFNet, ViT_UNet}`:

1. Extract source patches (labeled pixels) and stratified train/val/test splits; extract a random subsample of target patches (unlabeled pixels) with a train/val split.
2. Load the frozen backbone and slice off its penultimate layer to obtain feature vectors for every patch.
3. **Stage 1 (per epoch, per mini-batch):**
   a. Run a source batch and a target batch through the shared encoder to obtain $(\mu,\log\sigma^2,z)$ for both.
   b. Reconstruct features with the matching domain-specific decoder; compute reconstruction loss for each domain.
   c. Compute KL loss for each domain.
   d. Classify the source batch's $z$ and compute cross-entropy against true labels.
   e. Pass both domains' $z$ through the GRL → discriminator; compute domain BCE loss.
   f. Combine into $\mathcal{L}_1$ (Section 2.5) and backpropagate through encoder, both decoders, classifier, and discriminator jointly.
   g. At epoch end, evaluate validation accuracy, validation CE, and validation reconstruction error on both domains.
4. **Stage 2 (per epoch, per mini-batch), with encoder and classifier now frozen:**
   a. For a source batch: get $z$ and the one-hot true label as $y_0$; sample a timestep $t$ and noise $\epsilon$; form $y_t$ via the closed-form forward process; predict $\hat\epsilon$ with the diffusion network conditioned on $(z, y_t, f_{guid}, t)$; compute the MSE noise loss.
   b. For a target batch: identical, but $y_0$ is the classifier's own (stop-gradiented) soft prediction rather than a true label.
   c. Combine into $\mathcal{L}_2$ (Section 2.6) and backpropagate through the diffusion network only.
5. Save all five trained sub-network weights and a JSON config recording paths, dimensions, and hyperparameters; append a summary row (final losses/accuracy) for this backbone.
6. After all three backbones are trained: validate that every expected artifact file exists, then reload each backbone's encoder + classifier (diffusion network is *not* used at evaluation time in this notebook), run inference on the held-out source test set, and compute overall accuracy (OA), average accuracy (AA), Cohen's kappa, and weighted F1.
7. Optionally, run the encoder+classifier over every pixel of the full scene (labeled and unlabeled) to produce a dense classification map for visual comparison against the ground truth.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training_dapm_full.ipynb`

### 4.1 Defining Source and Target Domains
```python
coords = np.argwhere(y_img > 0)          # labeled pixels  -> source domain
...
coords = np.argwhere(y_img == 0)         # unlabeled pixels -> target domain
```
**What this does:** the notebook's notion of "domain" is not two different sensors or scenes — it is simply *labeled vs. unlabeled pixels within the same image*. The target set is randomly subsampled (`MAX_TARGET_UNLABELED = 20000`) since the unlabeled region is far larger than the labeled one.
**Why:** this framing lets domain-adversarial training double as a form of semi-supervised learning: the model is pushed to represent labeled and unlabeled pixels the same way in latent space, which should help the classifier generalize beyond the (potentially narrower) labeled region.

### 4.2 The Reparameterization and Gradient-Reversal Layers
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
            def grad(dy):
                return -lambda_ * dy
            return v, grad
        return _flip_gradients(x)
```
**What this does:** `Sampling` implements the VAE reparameterization trick exactly as in Section 2.2. `GradientReversal` uses TensorFlow's `tf.custom_gradient` to make a layer that is invisible in the forward pass but multiplies its incoming gradient by $-\lambda$ on the way back.
**Why:** this is the entire mechanical trick behind domain-adversarial training (Section 2.4) — no separate adversarial optimization loop or min-max alternation is needed; a single backward pass through the GRL achieves the adversarial update for both the discriminator (normal gradient) and the encoder (reversed gradient) simultaneously.

### 4.3 Network Builders
```python
def build_dapm_encoder(feature_dim, latent_dim=64, hidden_dim=256):
    ...
    z = Sampling(name='z_sample')([z_mu, z_logvar])
    return keras.Model(inp, [z_mu, z_logvar, z], name='dapm_full_encoder')

def build_dapm_discriminator(latent_dim, hidden_dim=128, grl_lambda=1.0):
    inp = keras.Input(shape=(latent_dim,), name='disc_z_in')
    x   = GradientReversal(lambda_=grl_lambda, name='disc_grl')(inp)
    ...

def build_dapm_diffusion(latent_dim, num_classes, T=100, t_embed_dim=32, hidden_dim=256):
    ...
    x = layers.Concatenate(...)([z_in, y_t_in, f_in, t_emb])
    ...
```
**What this does:** factory functions assemble each sub-network as a small Keras MLP. The discriminator inserts the GRL right after its input. The diffusion network concatenates the latent code, the noised label, the guidance vector, and a learned timestep embedding before two hidden layers and a linear output (the predicted noise, one value per class).
**Why:** keeping every sub-network as an independent, separately-saved Keras model lets Stage 1 and Stage 2 be trained, checkpointed, and reloaded independently — important because Stage 2 explicitly freezes the Stage-1 modules.

### 4.4 Loss Functions
```python
def kl_loss_from_stats(z_mu, z_logvar):
    return -0.5 * tf.reduce_mean(
        tf.reduce_sum(1.0 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), axis=-1)
    )

def recon_loss(x_true, x_rec):
    return tf.reduce_mean(tf.reduce_sum(tf.square(x_true - x_rec), axis=-1))

def domain_bce(y_true, y_prob):
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_prob))
```
**What this does:** direct, literal implementations of the KL term (Section 2.2), the sum-of-squared-errors reconstruction loss (Section 2.3), and domain BCE (Section 2.4).
**Why:** these are the three loss primitives combined into $\mathcal{L}_1$.

### 4.5 Diffusion Schedule and Forward Process
```python
def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    betas      = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas     = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars

def q_sample(y0, t_idx, alpha_bars):
    a_bar = tf.gather(alpha_bars, tf.cast(tf.squeeze(t_idx, axis=-1) - 1, tf.int32))
    a_bar = tf.cast(tf.reshape(a_bar, (-1, 1)), tf.float32)
    eps   = tf.random.normal(tf.shape(y0), dtype=tf.float32)
    y_t   = tf.sqrt(a_bar) * y0 + tf.sqrt(1.0 - a_bar) * eps
    return y_t, eps
```
**What this does:** precomputes the entire $\bar\alpha_t$ schedule once (Section 2.6), then `q_sample` implements the closed-form forward-noising formula directly — no need to simulate $T$ sequential noising steps, since the Gaussian forward process has an exact one-shot form.
**Why:** this efficiency trick (standard in DDPM-style training) is what makes training tractable: a random timestep is drawn per example per batch, and a single computation produces the noised label and the noise target for the loss.

### 4.6 Stage-1 Training Step
```python
feat_src = feature_extractor(xb_src, training=not FREEZE_BACKBONE)
z_mu_src, z_logvar_src, z_src = encoder(feat_src, training=True)
feat_src_rec = src_decoder(z_src, training=True)
y_src_prob   = classifier(z_src, training=True)
dom_src_prob = discriminator(z_src, training=True)
...
dom_loss = (domain_bce(tf.zeros_like(dom_src_prob), dom_src_prob) +
            domain_bce(tf.ones_like(dom_tgt_prob),  dom_tgt_prob))
loss = (LAMBDA_SRC_RECON*src_recon + LAMBDA_TGT_RECON*tgt_recon +
        LAMBDA_KL*(src_kl+tgt_kl) + LAMBDA_CE*src_ce + LAMBDA_DOMAIN*dom_loss)
grads = tape.gradient(loss, stage1_vars)
opt.apply_gradients(zip(grads, stage1_vars))
```
**What this does:** runs source and target patches through the shared encoder, reconstructs through the matching decoder, classifies the source batch, and discriminates domain on both; sums the weighted losses and updates all five Stage-1 sub-networks in one gradient step.
**Why:** because all five networks share the single computational graph through `z`, one `GradientTape` and one `apply_gradients` call is enough — the GRL inside the discriminator path automatically supplies the adversarial (sign-flipped) component of the encoder's gradient.

### 4.7 Stage-2 Training Step
```python
y_guidance_src = tf.stop_gradient(classifier(z_mu_src, training=False))
y0_src         = tf.one_hot(yb_src, depth=num_classes, dtype=tf.float32)
y_t_src, eps_src = q_sample(y0_src, t_src, alpha_bars)
eps_src_pred   = diffusion([z_src, y_t_src, y_guidance_src, t_src], training=True)
src_loss       = tf.reduce_mean(tf.reduce_sum(tf.square(eps_src - eps_src_pred), axis=-1))
...
y0_tgt = tf.stop_gradient(y_guidance_tgt)  # classifier soft labels as proxy
...
loss = src_loss + LAMBDA_TGT_DIFF * tgt_loss
grads = tape.gradient(loss, diffusion.trainable_variables)
```
**What this does:** builds the diffusion training pair $(y_t, \epsilon)$ for both source (true one-hot labels) and target (classifier's own soft prediction as a pseudo-label) batches, and trains only the diffusion network's weights — the encoder and classifier are called with `training=False` and never appear in `tape.gradient`'s variable list.
**Why:** this two-stage design avoids the diffusion objective from disturbing the already-trained classification representation; the diffusion model is purely a downstream consumer of the frozen latent space.

### 4.8 Evaluation: Diffusion is Trained but Not Used at Test Time
```python
def predict_with_dapm_classifier(feature_extractor, encoder, classifier, x_data, batch_size=256):
    ...
    z_mu, _, _ = encoder(feat, training=False)
    probs      = classifier(z_mu, training=False)
    ...
```
**What this does:** the function used for both held-out test metrics and the full-scene classification map relies only on the backbone → encoder (using the deterministic mean $\mu$, not a stochastic sample) → classifier path. The diffusion network's weights are saved to disk but are never loaded or called anywhere in the evaluation cells.
**Why this matters for interpreting the notebook:** Stage 2 functions purely as an auxiliary training signal / artifact in this implementation — it is trained and checkpointed but plays no role in producing the reported OA/AA/Kappa/F1 numbers or the classification maps. Any benefit it provides to the final results would have to come indirectly, if at all, since the classifier head it conditions on is frozen before Stage 2 even begins.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

To make every number traceable by hand, this toy example uses smaller dimensions than the real notebook configuration, but the exact same formulas:

- $K = 4$ classes (C0–C3), instead of the scene's real `num_classes`.
- Toy feature dimension $d=3$ and latent dimension $L=2$ (the real run uses backbone-dependent $d$ and `LATENT_DIM = 64`).
- Two **source** (labeled) patches $s_1, s_2$ and two **target** (unlabeled) patches $t_1, t_2$, mirroring one mini-batch from each domain.
  - $s_1$ is an **easy** case: a clean reconstruction and confident, correct classification.
  - $s_2$ is a normal, moderately-confident correct classification.
  - $t_1$ is a **well-aligned** target patch: its latent code lands close to the source cluster (small KL, good reconstruction, discriminator unsure of its domain).
  - $t_2$ is a **domain-shifted** target patch: it reconstructs poorly and the discriminator easily identifies it as target — the case domain-adversarial training is meant to fix.
- Loss weights are taken directly from the notebook's configuration: $\lambda_{src}=\lambda_{tgt}=\lambda_{CE}=1.0$, $\lambda_{KL}=0.01$, $\lambda_{dom}=0.2$, $\lambda_{tgt\_diff}=0.5$.
- For the diffusion schedule, this example uses $T=4$ toy steps with $\beta_{start}=0.10,\ \beta_{end}=0.40$ (the real notebook uses $T=100,\ \beta_{start}=10^{-4},\ \beta_{end}=2\times10^{-2}$ — too fine-grained to show by hand) — the *formulas* are identical.

**Backbone feature vectors (given, as if produced by a frozen backbone):**

| Patch | Domain | $h$ (feature vector) |
|---|---|---|
| $s_1$ | source, true class C0 | $[1.0,\ 0.5,\ -1.0]$ |
| $s_2$ | source, true class C2 | $[0.2,\ 1.2,\ 0.3]$ |
| $t_1$ | target | $[0.9,\ 0.4,\ -0.8]$ |
| $t_2$ | target | $[-0.5,\ 0.1,\ 1.5]$ |

Because the encoder, decoders, classifier, and discriminator are themselves multi-layer MLPs whose weights are learned (not part of the method's mathematical contribution), this example posits concrete, *plausible* outputs for each network call — exactly as if a trained network had produced them — and then carries every subsequent formula through in full.

### 5.1 Encoder: Reparameterization

**Given encoder outputs** $(\mu, \log\sigma^2)$ and **given noise** $\epsilon$:

| Patch | $\mu$ | $\log\sigma^2$ | $\epsilon$ |
|---|---|---|---|
| $s_1$ | $[0.50,-0.20]$ | $[-0.40,0.10]$ | $[0.30,-0.10]$ |
| $s_2$ | $[0.10,0.30]$ | $[-0.20,-0.30]$ | $[-0.20,0.40]$ |
| $t_1$ | $[0.45,-0.15]$ | $[-0.30,0.00]$ | $[0.10,0.20]$ |
| $t_2$ | $[-0.60,0.50]$ | $[0.20,0.40]$ | $[-0.30,-0.10]$ |

Computing $\sigma = \exp(0.5\log\sigma^2)$ and then $z=\mu+\sigma\odot\epsilon$:

- $s_1$: $\sigma=[e^{-0.20},e^{0.05}]=[0.8187,1.0513]$ → $z_{s_1}=[0.50+0.8187(0.30),\,-0.20+1.0513(-0.10)]=[0.7456,\,-0.3051]$
- $s_2$: $\sigma=[e^{-0.10},e^{-0.15}]=[0.9048,0.8607]$ → $z_{s_2}=[0.10-0.1810,\,0.30+0.3443]=[-0.0810,\,0.6443]$
- $t_1$: $\sigma=[e^{-0.15},e^{0}]=[0.8607,1.0000]$ → $z_{t_1}=[0.45+0.0861,\,-0.15+0.20]=[0.5361,\,0.0500]$
- $t_2$: $\sigma=[e^{0.10},e^{0.20}]=[1.1052,1.2214]$ → $z_{t_2}=[-0.60-0.3316,\,0.50-0.1221]=[-0.9316,\,0.3779]$

### 5.2 KL Divergence Per Sample

Using $\mathrm{KL}=-\tfrac12\sum(1+\log\sigma^2-\mu^2-\sigma^2)$:

- $s_1$: $1+\log\sigma^2=[0.60,1.10]$, $\mu^2=[0.25,0.04]$, $\sigma^2=e^{\log\sigma^2}=[0.6703,1.1052]$ → terms $=[0.60-0.25-0.6703,\ 1.10-0.04-1.1052]=[-0.3203,-0.0452]$, sum $=-0.3655$ → $\mathrm{KL}_{s_1}=0.1828$
- $s_2$: terms $=[0.80-0.01-0.8187,\ 0.70-0.09-0.7408]=[-0.0287,-0.1308]$, sum$=-0.1595$ → $\mathrm{KL}_{s_2}=0.0797$
- $t_1$: terms $=[0.70-0.2025-0.7408,\ 1.00-0.0225-1.0000]=[-0.2433,-0.0225]$, sum$=-0.2658$ → $\mathrm{KL}_{t_1}=0.1329$
- $t_2$: terms $=[1.20-0.36-1.2214,\ 1.40-0.25-1.4918]=[-0.3814,-0.3418]$, sum$=-0.7232$ → $\mathrm{KL}_{t_2}=0.3616$

Batch means: $\mathrm{KL}_{src}=\frac{0.1828+0.0797}{2}=0.1313$, $\ \mathrm{KL}_{tgt}=\frac{0.1329+0.3616}{2}=0.2473$.

Note how $t_2$'s latent distribution is much further from the standard-normal prior than any other sample — an early signal that this patch's representation is unusual relative to what the encoder normally produces.

### 5.3 Reconstruction Loss

**Given decoder outputs** $\hat h$ and the true $h$:

| Patch | Decoder used | $\hat h$ | $h$ | $\sum(h-\hat h)^2$ |
|---|---|---|---|---|
| $s_1$ | source decoder | $[0.95,0.55,-0.90]$ | $[1.0,0.5,-1.0]$ | $0.0025+0.0025+0.01=0.0150$ |
| $s_2$ | source decoder | $[0.25,1.10,0.25]$ | $[0.2,1.2,0.3]$ | $0.0025+0.01+0.0025=0.0150$ |
| $t_1$ | target decoder | $[0.85,0.35,-0.75]$ | $[0.9,0.4,-0.8]$ | $0.0025+0.0025+0.0025=0.0075$ |
| $t_2$ | target decoder | $[-0.30,0.20,1.30]$ | $[-0.5,0.1,1.5]$ | $0.04+0.01+0.04=0.0900$ |

Batch means: $\mathcal{L}_{recon}^{s}=\frac{0.0150+0.0150}{2}=0.0150$, $\ \mathcal{L}_{recon}^{t}=\frac{0.0075+0.0900}{2}=0.04875$.

$t_2$ reconstructs far worse than $t_1$ — exactly the behavior expected of a patch whose features the decoder rarely sees.

### 5.4 Classification Cross-Entropy (Source Only)

**Given classifier softmax outputs** at the sampled $z$:

| Patch | True class | Softmax $\hat y$ | $-\log \hat y_{y_i}$ |
|---|---|---|---|
| $s_1$ | C0 | $[0.70,0.10,0.15,0.05]$ | $-\ln(0.70)=0.3567$ |
| $s_2$ | C2 | $[0.05,0.15,0.65,0.15]$ | $-\ln(0.65)=0.4308$ |

$\mathcal{L}_{CE}=\frac{0.3567+0.4308}{2}=0.3937$

### 5.5 Domain Discriminator and the GRL

**Given discriminator outputs** $p_{dom}=G_\psi(z)$:

| Patch | $p_{dom}$ (P[target]) | Target label | Per-sample BCE |
|---|---|---|---|
| $s_1$ | 0.35 | 0 | $-\ln(1-0.35)=0.4308$ |
| $s_2$ | 0.40 | 0 | $-\ln(1-0.40)=0.5108$ |
| $t_1$ | 0.55 | 1 | $-\ln(0.55)=0.5978$ |
| $t_2$ | 0.85 | 1 | $-\ln(0.85)=0.1625$ |

Per-domain means: source $=\frac{0.4308+0.5108}{2}=0.4708$, target $=\frac{0.5978+0.1625}{2}=0.3802$.

$$\mathcal{L}_{dom} = 0.4708 + 0.3802 = 0.8510$$

This is consistent with the design of the toy example: the discriminator is *uncertain* about $t_1$ ($p_{dom}=0.55$, near the 0.5 boundary — the encoder is successfully confusing it about $t_1$'s domain), but very confident about $t_2$ ($p_{dom}=0.85$ — the encoder has failed to disguise $t_2$, which is exactly the signal that pushes future encoder updates, via the GRL, to fix this for samples like $t_2$).

**GRL backward illustration:** suppose the gradient of $\mathcal{L}_{dom}$ with respect to $z_{t_2}$, as computed by ordinary backpropagation through the discriminator, is $g = [0.12, -0.08]$ (toy values). Without a GRL, the encoder would receive exactly this gradient and update $z_{t_2}$ in the direction that helps the discriminator. With the GRL ($\lambda=1.0$) in between, the encoder instead receives

$$-\lambda \cdot g = -1.0 \times [0.12,-0.08] = [-0.12,\ 0.08]$$

— the opposite direction, which moves the encoder toward making $t_2$ harder to distinguish from source samples next time, even though the discriminator's own weights are still updated normally with $g$.

### 5.6 Stage-1 Total Loss

$$\mathcal{L}_1 = 1.0(0.0150) + 1.0(0.04875) + 0.01(0.1313+0.2473) + 1.0(0.3937) + 0.2(0.8510)$$

$$= 0.0150 + 0.04875 + 0.003785 + 0.3937 + 0.1702 = 0.6314$$

### 5.7 Diffusion Schedule

With toy values $T=4,\ \beta_{start}=0.10,\ \beta_{end}=0.40$:

$$\beta = \mathrm{linspace}(0.10,0.40,4) = [0.10,\ 0.20,\ 0.30,\ 0.40]$$
$$\alpha = 1-\beta = [0.90,\ 0.80,\ 0.70,\ 0.60]$$
$$\bar\alpha_1=0.90,\quad \bar\alpha_2=0.90\times0.80=0.72,\quad \bar\alpha_3=0.72\times0.70=0.504,\quad \bar\alpha_4=0.504\times0.60=0.3024$$

### 5.8 Forward Diffusion and Diffusion Loss — Source Sample $s_1$

Sampled timestep $t=2$ → $\bar\alpha_2=0.72$, so $\sqrt{\bar\alpha_2}=0.8485$, $\sqrt{1-\bar\alpha_2}=\sqrt{0.28}=0.5292$.

$$y_0 = [1,0,0,0] \ \text{(one-hot for C0)}, \qquad \epsilon=[0.20,-0.10,0.05,-0.15] \ \text{(given noise)}$$

$$y_t = 0.8485\,[1,0,0,0] + 0.5292\,[0.20,-0.10,0.05,-0.15] = [0.9543,\ -0.0529,\ 0.0265,\ -0.0794]$$

Conditioning the diffusion network on $\big(z_{s_1}=[0.7456,-0.3051],\ y_t,\ f_{guid}=[0.70,0.10,0.15,0.05],\ t=2\big)$, suppose the network predicts

$$\hat\epsilon = [0.18,-0.08,0.07,-0.12]$$

$$\mathcal{L}_{diff}^{s_1} = (0.20-0.18)^2+(-0.10+0.08)^2+(0.05-0.07)^2+(-0.15+0.12)^2 = 0.0004+0.0004+0.0004+0.0009 = 0.0021$$

### 5.9 Forward Diffusion and Diffusion Loss — Target Sample $t_1$

Sampled timestep $t=3$ → $\bar\alpha_3=0.504$, so $\sqrt{\bar\alpha_3}=0.7099$, $\sqrt{1-\bar\alpha_3}=\sqrt{0.496}=0.7043$.

Because $t_1$ is unlabeled, $y_0$ is the classifier's own **soft** pseudo-label (the guidance vector itself), not a one-hot vector:

$$y_0 = f_{guid} = [0.55,0.20,0.15,0.10], \qquad \epsilon = [-0.10,0.25,-0.05,0.10] \ \text{(given noise)}$$

$$y_t = 0.7099\,[0.55,0.20,0.15,0.10] + 0.7043\,[-0.10,0.25,-0.05,0.10]$$
$$= [0.3905,0.1420,0.1065,0.0710] + [-0.0704,0.1761,-0.0352,0.0704] = [0.3201,\ 0.3181,\ 0.0713,\ 0.1414]$$

Conditioning on $\big(z_{t_1}=[0.5361,0.0500],\ y_t,\ f_{guid}=[0.55,0.20,0.15,0.10],\ t=3\big)$, suppose the network predicts

$$\hat\epsilon = [-0.05,0.18,-0.08,0.06]$$

$$\mathcal{L}_{diff}^{t_1} = (-0.10+0.05)^2+(0.25-0.18)^2+(-0.05+0.08)^2+(0.10-0.06)^2 = 0.0025+0.0049+0.0009+0.0016 = 0.0099$$

### 5.10 Stage-2 Total Loss

$$\mathcal{L}_2 = \mathcal{L}_{diff}^{s_1} + \lambda_{tgt\_diff}\cdot\mathcal{L}_{diff}^{t_1} = 0.0021 + 0.5(0.0099) = 0.0021+0.00495 = 0.00705$$

### 5.11 Summary Table

| Quantity | $s_1$ | $s_2$ | $t_1$ | $t_2$ | Batch value used in loss |
|---|---|---|---|---|---|
| KL | 0.1828 | 0.0797 | 0.1329 | 0.3616 | $\mathrm{KL}_{src}=0.1313$, $\mathrm{KL}_{tgt}=0.2473$ |
| Reconstruction SSE | 0.0150 | 0.0150 | 0.0075 | 0.0900 | $\mathcal{L}_{recon}^s=0.0150$, $\mathcal{L}_{recon}^t=0.04875$ |
| Classification $-\log p$ | 0.3567 | 0.4308 | — | — | $\mathcal{L}_{CE}=0.3937$ |
| Domain BCE | 0.4308 | 0.5108 | 0.5978 | 0.1625 | $\mathcal{L}_{dom}=0.8510$ |
| Diffusion noise-MSE | 0.0021 (t=2) | — | 0.0099 (t=3) | — | $\mathcal{L}_2=0.00705$ |
| **Stage-1 total** $\mathcal{L}_1$ | | | | | **0.6314** |

This end-to-end trace shows precisely how a single domain-shifted patch like $t_2$ propagates through every term of the objective — high KL, poor reconstruction, and a discriminator that easily flags it as off-domain — which is the exact signal the gradient-reversed encoder gradient uses to gradually pull such patches' representations back toward the shared, domain-invariant latent region.

---

## 6. References

[1] Ganin, Y. & Lempitsky, V. "Unsupervised Domain Adaptation by Backpropagation." *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 2015. [arXiv:1409.7495](https://arxiv.org/abs/1409.7495)

[2] Kingma, D. P. & Welling, M. "Auto-Encoding Variational Bayes." *International Conference on Learning Representations (ICLR)*, 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

[3] Ho, J., Jain, A. & Abbeel, P. "Denoising Diffusion Probabilistic Models." *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

[4] Dhariwal, P. & Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233) — introduces the classifier-guidance idea that the notebook's `f_guidance` conditioning signal echoes.

[5] Rao, Y., Zhao, W., Zhu, Z., Lu, J. & Zhou, J. "Global Filter Networks for Image Classification." *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [arXiv:2107.00645](https://arxiv.org/abs/2107.00645) — the GFNet backbone used as one of the three feature extractors.

[6] Akkari, F. et al. "DiffusionAAE: Enhancing hyperspectral image classification with conditional diffusion model and Adversarial Autoencoder." *Computers and Electronics in Agriculture / ScienceDirect*, 2025. [https://www.sciencedirect.com/science/article/pii/S157495412500127X](https://www.sciencedirect.com/science/article/pii/S157495412500127X) — closely related work combining an adversarial autoencoder with a conditional diffusion model for the same application domain (multispectral/hyperspectral classification).
