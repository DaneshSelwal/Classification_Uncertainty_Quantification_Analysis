# DAPM Uncertainty Estimation for Multispectral Scene Classification — Notebook Explainer

> **Who this document is for:** Anyone who used AI to help write this notebook and now wants to
> deeply understand *how* each method works — both for personal learning and for writing an
> academic paper. Every method is covered end-to-end: plain-English explanation, step-by-step
> logic, worked numerical example, annotated code, and limitations.

---

## Table of Contents

1. [Overview](#overview)
2. [Environment & Dependencies](#environment--dependencies)
3. [Data & Problem Setup](#data--problem-setup)
4. [Method 1 — Per-Band Min-Max Normalisation](#method-1--per-band-min-max-normalisation)
5. [Method 2 — Patch Extraction](#method-2--patch-extraction)
6. [Method 3 — Custom Keras Layers (PatchExtractor, PatchPositionEncoder, GlobalFilterLayer, PatchEncoderWithCLS, Sampling)](#method-3--custom-keras-layers)
7. [Method 4 — Frozen Feature Extractor](#method-4--frozen-feature-extractor)
8. [Method 5 — VAE Encoder with Reparameterisation Trick](#method-5--vae-encoder-with-reparameterisation-trick)
9. [Method 6 — Latent-Space Softmax Classifier](#method-6--latent-space-softmax-classifier)
10. [Method 7 — Conditional Diffusion Denoiser](#method-7--conditional-diffusion-denoiser)
11. [Method 8 — Cosine / Linear Beta Schedule](#method-8--cosine--linear-beta-schedule)
12. [Method 9 — Reverse Diffusion Sampling](#method-9--reverse-diffusion-sampling)
13. [Method 10 — Stochastic Latent Tiling (sample_dapm_chunk)](#method-10--stochastic-latent-tiling-sample_dapm_chunk)
14. [Method 11 — Welch t-Test Uncertainty Estimation](#method-11--welch-t-test-uncertainty-estimation)
15. [Method 12 — Chunked Scene Inference with NPZ Caching](#method-12--chunked-scene-inference-with-npz-caching)
16. [Results & Comparisons](#results--comparisons)
17. [Academic Paper Summary](#academic-paper-summary)
18. [References](#references)

---

## Overview

This notebook implements a complete uncertainty-aware inference pipeline for multispectral land-cover classification. Three pre-trained deep learning models — **AlexNet**, **GFNet**, and **ViT-UNet** — are each paired with a **Diffusion-Augmented Probabilistic Model (DAPM)**: a generative architecture that combines a Variational Autoencoder (VAE) with a conditional diffusion denoiser to produce multiple stochastic class-probability samples for every pixel in a 330×307 six-band image.

Uncertainty is then quantified by applying a **Welch t-test** to compare the top-1 versus top-2 class probability distributions across those samples. A pixel is flagged as *uncertain* when the two distributions are statistically indistinguishable (p-value above a configurable threshold). The result is a set of spatial maps, per-class metrics, comparison plots, and a multi-sheet Excel workbook — all saved back to Google Drive.

---

## Environment & Dependencies

| Library | Purpose |
|---|---|
| `os`, `sys`, `io`, `gc`, `json`, `random`, `shutil`, `time`, `warnings` | Standard library utilities: file I/O, garbage collection, timing, random seeds |
| `subprocess`, `pathlib.Path` | Run shell commands (pip installs); object-oriented file paths |
| `numpy` | Numerical array operations, sampling, statistics |
| `pandas` | CSV loading, DataFrame manipulation, Excel writing via ExcelWriter |
| `scipy.stats.ttest_ind` | Welch independent-samples t-test for uncertainty estimation |
| `seaborn` | Statistical bar charts and histogram plots |
| `matplotlib` | Core plotting engine; colormaps, spatial maps, histograms |
| `tensorflow` / `keras` | Deep learning framework for loading and running all neural network models |
| `xlsxwriter` | Backend for writing richly formatted Excel workbooks with embedded images |
| `openpyxl` | Used in validation step to open and inspect the saved Excel file |
| `google.colab` | Optional: mounts Google Drive when running in Colab |

**Global seeds** are set to 42 across `random`, `numpy`, and `tensorflow` for full reproducibility. Warnings are suppressed and Seaborn's `whitegrid` style is applied globally.

---

## Data & Problem Setup

**Dataset:** A single multispectral scene stored as two CSV files:
- `data.csv` — flattened pixel values reshaped to `(330, 307, 6)`: 330 rows × 307 columns × 6 spectral bands.
- `ref.csv` — integer class labels reshaped to `(330, 307)`. Label value `0` means unlabelled; values `1` through `N` represent land-cover classes (0-indexed after loading: class 0 = label 1).

**Problem type:** Multi-class per-pixel classification (semantic segmentation of a remote sensing scene).

**Scale:** 330 × 307 = 101,310 pixels total. Only pixels with label > 0 are used for accuracy evaluation; all 101,310 pixels receive inference.

**Preprocessing:** Each of the 6 spectral bands is independently normalised to [0, 1] via min-max scaling (detailed in Method 1). Pixels are not processed individually — each is represented by a 9×9 spatial patch centred on it (Method 2).

---

## Method 1 — Per-Band Min-Max Normalisation

### a) What it is

> Think of each spectral band as a separate grayscale photograph of the scene taken through a different coloured lens. Min-max normalisation simply rescales each photo so its darkest pixel becomes 0 and its brightest becomes 1, making all bands comparable in scale regardless of the sensor's original units.

Min-max normalisation maps each pixel value in band `b` to the range [0, 1] using the band's global minimum and maximum.

### b) Why it's used here

Deep learning models are sensitive to the scale of their inputs. The six multispectral bands may have very different raw value ranges (e.g., near-infrared vs visible blue). Normalising each band independently ensures no single band dominates simply due to larger numeric values.

### c) How it works — Step by step

1. Load the CSV into a NumPy array and reshape to `(H, W, B)` — here `(330, 307, 6)`.
2. For each band `bi` in `0..5`:
   - Extract the 2D slice: `band = x[:, :, bi]`
   - Compute `mn = min(band)` and `mx = max(band)`
   - Compute `denom = max(mx - mn, 1e-8)` — the small epsilon prevents division by zero for constant bands
   - Apply: `x_norm[:, :, bi] = (band - mn) / denom`
3. Return the normalised array with the same shape.

```
normalised_value = (raw_value - band_minimum) / max(band_maximum - band_minimum, 1e-8)
```

### d) ASCII Flow Diagram

```
data.csv (flat)
      |
      v
 reshape → (330, 307, 6)
      |
      v
 for each band b in [0..5]:
      |
      ├─ find min, max
      ├─ denom = max - min (or 1e-8)
      └─ normalised = (band - min) / denom
      |
      v
 x_norm: (330, 307, 6)  values in [0, 1]
```

### e) Worked Numerical Example

Suppose band 0 has raw values: `[100, 200, 150, 300, 50]`

- `mn = 50`, `mx = 300`, `denom = 250`
- Normalised: `[(100-50)/250, (200-50)/250, (150-50)/250, (300-50)/250, (50-50)/250]`
- Result: `[0.2, 0.6, 0.4, 1.0, 0.0]`

The band's value range is now perfectly mapped to [0, 1].

### f) Code Walkthrough

```python
def load_multispectral_6band(data_path, label_path, h, w, b):
    # Load CSV and reshape to (H, W, B) — each row in CSV is one pixel's bands
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(h, w, b)
    # Load integer labels and reshape to (H, W)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(h, w)

    x_norm = np.empty_like(x, dtype=np.float32)  # pre-allocate output
    for bi in range(b):                           # process each band independently
        band  = x[:, :, bi]
        mn, mx = float(np.min(band)), float(np.max(band))
        denom  = max(mx - mn, 1e-8)              # protect against constant bands
        x_norm[:, :, bi] = (band - mn) / denom   # scale to [0, 1]
    return x_norm, y
```

### g) Output & Interpretation

Returns `x_norm` of shape `(330, 307, 6)` with all values in `[0.0, 1.0]`, and `y` of shape `(330, 307)` with integer class IDs. All subsequent operations use `x_norm`.

### h) Limitations

- Min-max scaling is sensitive to outliers: a single very bright pixel sets the maximum, compressing all other values.
- Normalisation is applied to the full scene independently per band — no reference to a train set distribution, which can cause issues if inference scenes differ from training.
- The epsilon `1e-8` prevents crashes on constant bands but the resulting band carries no information.
- Per-band normalisation ignores correlations between bands.

---

## Method 2 — Patch Extraction

### a) What it is

> Instead of feeding a single pixel's 6 values into the model, we cut out a small square "window" of 9×9 pixels centred on the target pixel — like taking a zoomed postage stamp of the scene — and use all 9×9×6 = 486 values as the model's input. This gives the model spatial context (texture, edges, neighbourhood patterns) rather than just a single spectral point.

Each pixel of interest is represented by a fixed-size spatial patch extracted from the padded image.

### b) Why it's used here

The pre-trained models (AlexNet, GFNet, ViT-UNet) expect 2D image patches as input, not single-pixel vectors. Spatial patches encode local texture and neighbourhood structure, which are important features for land-cover classification.

### c) How it works — Step by step

1. Compute `pad = patch_size // 2` — for a 9×9 patch, `pad = 4`.
2. Pad the image on all spatial sides by `pad` pixels using **edge replication** (`mode='edge'`): border pixels are repeated, preventing artificial discontinuities.
3. For each coordinate `(r, c)` in the input list:
   - In the padded image, the pixel at `(r, c)` maps to `(r + pad, c + pad)`.
   - Extract the slice `[r : r + patch_size, c : c + patch_size, :]` — this is the 9×9×6 patch.
4. Stack all patches into output array of shape `(N, 9, 9, 6)`.

```
patch_shape = (patch_size, patch_size, n_bands)
output_shape = (n_coords, 9, 9, 6)
```

### d) ASCII Flow Diagram

```
x_img: (330, 307, 6)
      |
      v
  edge-pad by 4 → (338, 315, 6)
      |
      v
  for each (r, c) in coords:
      |
      └─ slice padded[r : r+9, c : c+9, :] → patch (9, 9, 6)
      |
      v
  stack → (N_coords, 9, 9, 6)
```

### e) Worked Numerical Example

Image is `(5, 5, 1)` (tiny single-band example). `patch_size = 3`, `pad = 1`.
After edge-padding, image is `(7, 7, 1)`.

For coordinate `(0, 0)` (top-left corner):
- In padded image: slice `[0:3, 0:3, :]`
- This returns a 3×3 patch whose top-left corner is the replicated border pixel — no "black padding" is introduced.

### f) Code Walkthrough

```python
def extract_patches_from_coords(x_img, coords, patch_size=9):
    pad   = patch_size // 2                           # 4 for a 9x9 patch
    # Pad spatial dims only; leave band dim untouched
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    out   = np.empty(
        (coords.shape[0], patch_size, patch_size, x_img.shape[-1]), dtype=np.float32
    )
    for i, (r, c) in enumerate(coords):
        # Row r in original maps to row r in padded (offset handled by pad)
        out[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
    return out
```

### g) Output & Interpretation

Returns an array of shape `(N, 9, 9, 6)` containing one patch per requested coordinate. These patches are the direct input to the feature extractor network.

### h) Limitations

- A fixed patch size (9×9) may be too small to capture large-scale spatial context and too large to be precise at small feature boundaries.
- Edge replication can introduce subtle artefacts at scene borders since border patches are partly filled with replicated values.
- Extracting patches in a Python loop is slower than a fully vectorised approach; the notebook mitigates this through chunked processing.
- Patches overlap significantly (stride 1), so nearby pixels share most of their input data.

---

## Method 3 — Custom Keras Layers

### a) What it is

> These five custom layers are like LEGO bricks specially designed for the three model architectures. Without registering them, Keras would not know how to reconstruct the pre-trained models from their `.keras` files — like trying to assemble furniture with missing instructions. Each layer encodes one specific transformation needed by AlexNet, GFNet, or ViT-UNet.

Five `tf.keras.layers.Layer` subclasses are defined and registered for model deserialisation.

### b) Why it's used here

The pre-trained models use non-standard Keras layers. When `keras.models.load_model` reads a `.keras` file, it needs to reconstruct every layer by name. Without registering these custom classes, loading would fail with an "Unknown layer" error.

### c) How it works — Step by step (one per layer)

**PatchExtractor:**
1. Uses `tf.image.extract_patches` with stride = patch size to tile the input image into non-overlapping patches.
2. Reshapes output to `(batch, num_patches, patch_dim)`.

```
output_shape = (batch, (H/P) * (W/P), P*P*C)
where P = patch_size, C = channels
```

**PatchPositionEncoder:**
1. Projects each patch vector to `projection_dim` via a Dense layer.
2. Adds a learned positional embedding (one embedding per patch position index).

```
output = Dense(projection_dim)(patches) + Embedding(position_index)
```

**GlobalFilterLayer (GFNet):**
1. Reshapes token sequence to a 2D spatial grid.
2. Applies 2D FFT to move to frequency domain.
3. Multiplies element-wise by a learned complex filter `w_real + i*w_imag`.
4. Applies inverse 2D FFT to return to spatial domain.
5. Reshapes back to token sequence.

```
output = IFFT2D( FFT2D(input) * (w_real + i * w_imag) )
```

**PatchEncoderWithCLS (ViT):**
1. Projects patches to `projection_dim`.
2. Prepends a learnable CLS (classification) token — a single trainable vector.
3. Adds positional embeddings for all `num_patches + 1` positions (patches + CLS).

```
output = concat([CLS_token, Dense(patches)]) + Embedding(positions)
```

**Sampling (VAE reparameterisation trick):**
1. Takes `z_mu` (mean) and `z_logvar` (log-variance) as inputs.
2. Samples `eps ~ N(0, I)`.
3. Returns `z = z_mu + exp(0.5 * z_logvar) * eps`.

```
z = z_mu + std * eps
where std = exp(0.5 * z_logvar), eps ~ Normal(0, 1)
```

### d) ASCII Flow Diagram

```
PatchExtractor:       Image (B, H, W, C) → Patches (B, N_patches, dim)
PatchPositionEncoder: Patches            → Embedded patches + positional encoding
GlobalFilterLayer:    Token grid         → FFT → filter multiply → IFFT → Token grid
PatchEncoderWithCLS:  Patches            → [CLS | Embedded patches] + positional encoding
Sampling:             (z_mu, z_logvar)   → z ~ N(z_mu, exp(0.5*z_logvar))
```

### e) Worked Numerical Example (Sampling layer)

Suppose `z_mu = [0.5, -0.3]` and `z_logvar = [0.0, 0.0]` (zero log-variance → std = 1.0).

- `std = exp(0.5 * [0.0, 0.0]) = [1.0, 1.0]`
- Sample `eps = [0.2, -0.7]`
- `z = [0.5, -0.3] + [1.0, 1.0] * [0.2, -0.7] = [0.7, -1.0]`

Each call produces a different sample; the layer is stochastic at inference time.

### f) Code Walkthrough

```python
@tf.keras.utils.register_keras_serializable()  # tells Keras to remember this class by name
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mu, z_logvar = inputs          # unpack the two encoder outputs
        eps = tf.random.normal(shape=tf.shape(z_mu))  # standard normal noise
        return z_mu + tf.exp(0.5 * z_logvar) * eps    # reparameterised sample

# Registry dict passed to keras.models.load_model
CUSTOM_OBJECTS = {
    'PatchExtractor':       PatchExtractor,
    'PatchPositionEncoder': PatchPositionEncoder,
    'GlobalFilterLayer':    GlobalFilterLayer,
    'PatchEncoderWithCLS':  PatchEncoderWithCLS,
}
```

### g) Output & Interpretation

These layers have no stand-alone output at inference time — they are internal building blocks of the loaded models. The `Sampling` layer is the stochastic heart of the VAE: it introduces randomness that is later exploited by the diffusion sampler.

### h) Limitations

- `GlobalFilterLayer` assumes a square token grid (`token_side × token_side`); non-square inputs require architectural changes.
- The reparameterisation trick requires the encoder to output both mean and log-variance — models without this cannot be adapted without retraining.
- Custom layers increase complexity of model serialisation and version compatibility.
- The FFT filter introduces a fixed spatial frequency resolution tied to `token_side`.

---

## Method 4 — Frozen Feature Extractor

### a) What it is

> Imagine the pre-trained classification model as a factory with two departments: the first department looks at the image and extracts rich, abstract descriptions of what it sees (the feature extractor); the second department reads those descriptions and makes a class decision (the classifier head). The DAPM discards the second department and uses only the first as a fixed perceptual lens.

The penultimate layer output of the pre-trained Keras model is used as a fixed feature vector for each input patch. The base model's weights are frozen.

### b) Why it's used here

Pre-trained models (trained on the full dataset) have already learned to produce discriminative feature representations. Re-using these representations as inputs to the DAPM avoids retraining from scratch and leverages existing domain knowledge.

### c) How it works — Step by step

1. Load the full pre-trained model from its `.keras` file.
2. Access `base_model.layers[-2].output` — the output of the second-to-last layer (just before the final softmax classification layer).
3. Build a new `keras.Model` with the same input but with the penultimate layer output as the output.
4. Set `feat_model.trainable = False` — no gradients are computed through this model.

```
base_model input → ... → penultimate layer output (feature_dim,) → [DAPM]
                                                  ^
                                         feature extractor stops here
```

### d) ASCII Flow Diagram

```
Input Patch (9, 9, 6)
        |
        v
[Base Model: AlexNet / GFNet / ViT-UNet]
        |
        v
Penultimate Layer Output  →  feature vector (feature_dim,)   [frozen]
        |
        v
        DAPM Encoder
```

### e) Worked Numerical Example

Suppose the base model has layers: `[Input, Conv1, Conv2, Dense, Softmax]`.

- `layers[-2]` = `Dense` (the layer before the Softmax).
- The feature extractor outputs the Dense layer's activations — a vector of, say, 512 values.
- The final Softmax layer is never executed.

### f) Code Walkthrough

```python
def get_feature_extractor(base_model):
    penultimate = base_model.layers[-2].output  # grab second-to-last layer's output tensor
    feat_model  = keras.Model(
        base_model.input, penultimate,           # same input, new output
        name=f'{base_model.name}_feature_extractor'
    )
    feat_model.trainable = False                 # freeze all weights
    return feat_model
```

### g) Output & Interpretation

Returns a Keras Model that accepts patches of shape `(batch, 9, 9, 6)` and outputs a feature vector of shape `(batch, feature_dim)`. The feature dimension varies by base model and is read from the saved DAPM config JSON.

### h) Limitations

- The quality of DAPM predictions is bounded by the quality of the frozen feature representations.
- If the scene distribution shifts significantly from training, frozen features may be poor inputs to the DAPM.
- Using `layers[-2]` assumes a specific model topology — models with branching architectures (residual connections, multi-head outputs) may require a different layer index.
- The feature extractor is frozen and cannot be fine-tuned on the target scene.

---

## Method 5 — VAE Encoder with Reparameterisation Trick

### a) What it is

> A regular encoder maps an input to a single fixed point in "feature space." The VAE encoder maps it to a *cloud* — a Gaussian distribution centred at `z_mu` with spread `exp(0.5 * z_logvar)`. By sampling from this cloud rather than using the fixed centre, the model can explore multiple plausible interpretations of the same input — the source of the uncertainty estimates.

The VAE encoder takes the frozen feature vector and outputs a probabilistic latent code parameterised by its mean (`z_mu`) and log-variance (`z_logvar`), plus a single stochastic sample `z`.

### b) Why it's used here

A deterministic encoder produces a single latent code per input; running it many times gives identical outputs. The VAE encoder's probabilistic nature means each of the `N_SAMPLES = 30` samples draws a different `z`, producing a distribution of class-probability predictions whose spread quantifies uncertainty.

### c) How it works — Step by step

1. Input: feature vector of shape `(feature_dim,)`.
2. Two hidden Dense layers with ReLU activation produce a shared representation.
3. Two parallel output Dense layers produce `z_mu` (mean) and `z_logvar` (log-variance) — both of shape `(latent_dim,)`.
4. The `Sampling` layer draws `eps ~ N(0, I)` and returns `z = z_mu + exp(0.5 * z_logvar) * eps`.

```
z_std = exp(0.5 * z_logvar)
z     = z_mu + z_std * eps,   eps ~ Normal(0, 1)
```

### d) ASCII Flow Diagram

```
feature_vector (feature_dim,)
        |
        v
   Dense(256, relu)  ← enc_h1
        |
        v
   Dense(256, relu)  ← enc_h2
        |
       / \
      /   \
z_mu       z_logvar   [shape: (latent_dim,) each]
      \   /
       \ /
    Sampling layer
        |
        v
    z_sample  (latent_dim,)
```

### e) Worked Numerical Example

Suppose `latent_dim = 2`. After the hidden layers, suppose:
- `z_mu = [1.0, -0.5]`
- `z_logvar = [0.0, 0.693]`   (log(2) ≈ 0.693, so std for dim 1 = sqrt(2) ≈ 1.41)

Sample `eps = [0.3, -1.0]`.

- `std = exp(0.5 * [0.0, 0.693]) = [1.0, 1.41]`
- `z = [1.0 + 1.0*0.3,  -0.5 + 1.41*(-1.0)] = [1.3, -1.91]`

A different `eps` on the next call yields a different `z`.

### f) Code Walkthrough

```python
def build_dapm_encoder(feature_dim, latent_dim=64, hidden_dim=256):
    inp     = keras.Input(shape=(feature_dim,), name='enc_feature_in')
    h       = layers.Dense(hidden_dim, activation='relu', name='enc_h1')(inp)  # first hidden
    h       = layers.Dense(hidden_dim, activation='relu', name='enc_h2')(h)    # second hidden
    z_mu    = layers.Dense(latent_dim, name='z_mu')(h)       # mean of latent distribution
    z_logvar = layers.Dense(latent_dim, name='z_logvar')(h)  # log-variance of latent dist.
    z       = Sampling(name='z_sample')([z_mu, z_logvar])    # stochastic sample via reparam
    return keras.Model(inp, [z_mu, z_logvar, z], name='dapm_full_encoder')
```

### g) Output & Interpretation

Returns three tensors: `z_mu`, `z_logvar`, and `z_sample`. In the inference pipeline, `z_mu` is used to compute the soft guidance signal (through the classifier), while the stochastic samples are generated externally by tiling `z_mu` and adding random perturbations (see Method 10).

### h) Limitations

- The Gaussian assumption for the latent space may not capture multi-modal distributions (e.g., spectrally ambiguous land-cover types).
- `latent_dim` (default 64) controls the information bottleneck; too small loses discriminative information, too large approaches a deterministic encoder.
- VAE training requires a KL-divergence regularisation term; if the training was poorly configured, `z_logvar` may not accurately reflect true uncertainty.
- The reparameterisation trick only works for continuous, differentiable distributions.

---

## Method 6 — Latent-Space Softmax Classifier

### a) What it is

> Once the VAE encoder has compressed the feature vector into a small latent code `z`, this mini neural network acts as the "instant opinion" module — it takes the latent code (using just its mean, not a sample) and produces a class probability vector. This probability vector is used as a *guidance signal* to steer the diffusion denoiser toward realistic class predictions.

A two-layer MLP with a softmax output head maps the latent mean `z_mu` to class probabilities.

### b) Why it's used here

The classifier provides a deterministic baseline prediction that is used as the "conditioning" or guidance signal for the conditional diffusion denoiser. It helps the diffusion model know which class region of the output space to denoise toward.

### c) How it works — Step by step

1. Input: `z_mu` of shape `(latent_dim,)`.
2. One hidden Dense layer with ReLU — shape `(hidden_dim,)`.
3. Output Dense layer with Softmax — shape `(num_classes,)`.
4. Output values sum to 1.0 and are interpreted as class probabilities.

```
class_probs = Softmax( Dense(num_classes)( ReLU( Dense(128)(z_mu) ) ) )
```

### d) ASCII Flow Diagram

```
z_mu (latent_dim,)
        |
        v
   Dense(128, relu)
        |
        v
   Dense(num_classes, softmax)
        |
        v
   class_probabilities (num_classes,)  — sums to 1.0
```

### e) Worked Numerical Example

`num_classes = 3`, `z_mu = [0.5, -1.2]` (latent_dim=2 for brevity).

After hidden layer: `h = [0.8, 0.0, 1.1]` (relu applied).

Raw logits after output layer: `[2.1, -0.3, 0.7]`.

Softmax: `exp([2.1, -0.3, 0.7]) = [8.17, 0.74, 2.01]`, sum = 10.92.

Class probs: `[0.748, 0.068, 0.184]` — model is most confident in class 0.

### f) Code Walkthrough

```python
def build_dapm_classifier(latent_dim, num_classes, hidden_dim=128):
    inp = keras.Input(shape=(latent_dim,), name='clf_z_in')      # accepts the latent mean
    h   = layers.Dense(hidden_dim, activation='relu', name='clf_h1')(inp)  # hidden layer
    out = layers.Dense(num_classes, activation='softmax', name='clf_out')(h) # prob output
    return keras.Model(inp, out, name='dapm_full_classifier')
```

### g) Output & Interpretation

Returns a probability vector of shape `(num_classes,)`. At inference, this vector (from `z_mu`, not a stochastic sample) is replicated `N_SAMPLES` times and provided as the conditioning signal to the diffusion denoiser. High-confidence guidance helps the diffusion process converge to a consistent class.

### h) Limitations

- The classifier acts on `z_mu` only; it doesn't capture uncertainty inherent in the VAE's latent distribution.
- If the classifier guidance is wrong (e.g., for a spectrally ambiguous pixel), it may bias the diffusion denoiser toward an incorrect class, reducing calibration.
- A single hidden layer may be insufficient for complex multi-class boundaries.
- The softmax outputs are not calibrated probabilities without temperature scaling or isotonic regression.

---

## Method 7 — Conditional Diffusion Denoiser

### a) What it is

> Imagine starting with a jar of noise (pure random values) and having an AI progressively remove the noise, step by step, guided by what the pixel's latent code says and what the classifier already thinks. At the end of T steps, you're left with a clean class-probability vector — a single "opinion" about what land-cover class this pixel belongs to. Run this process 30 times and you get 30 different opinions; their agreement or disagreement tells you how certain the model is.

The conditional diffusion denoiser is a neural network that predicts the noise component `eps` at each step of the reverse diffusion chain, conditioned on the latent code `z`, the current noisy state `y_t`, and the classifier guidance.

### b) Why it's used here

The diffusion denoiser generates diverse, high-quality samples of class-probability vectors. Unlike the deterministic classifier, it can produce multi-modal outputs for ambiguous pixels, giving the uncertainty estimator richer signal to work with.

### c) How it works — Step by step

1. **Inputs:**
   - `z`: latent code from the VAE encoder, shape `(latent_dim,)`.
   - `y_t`: current noisy class-probability estimate at diffusion step `t`, shape `(num_classes,)`.
   - `guidance`: soft classifier probabilities from the latent classifier, shape `(num_classes,)`.
   - `t`: integer time step, shape `(1,)`.
2. The timestep `t` is embedded via a learned Embedding layer and flattened to `(t_embed_dim,)`.
3. All four inputs are concatenated: `[z, y_t, guidance, t_embed]`.
4. Two hidden Dense layers with ReLU process the concatenated vector.
5. A linear output Dense layer predicts the noise `eps_pred` of shape `(num_classes,)`.

```
eps_pred = Linear( ReLU( Dense(hidden_dim)( concat([z, y_t, guidance, t_emb]) ) ) )
```

### d) ASCII Flow Diagram

```
z (latent_dim)     y_t (num_classes)    guidance (num_classes)   t (int)
      |                   |                      |                    |
      |                   |                      |           Embedding(T+1, t_embed_dim)
      |                   |                      |                    |
      └───────────────────┴──────────────────────┴────────────────────┘
                                        |
                                   Concatenate
                                        |
                                        v
                              Dense(hidden_dim, relu)
                                        |
                                        v
                              Dense(hidden_dim, relu)
                                        |
                                        v
                              Dense(num_classes, linear)
                                        |
                                        v
                                   eps_pred  (num_classes,)
```

### e) Worked Numerical Example

`num_classes = 3`, `latent_dim = 4`, `t_embed_dim = 4`.

Suppose:
- `z = [0.2, -0.5, 0.8, 0.1]`
- `y_t = [0.4, 0.3, 0.3]` (noisy, near-uniform)
- `guidance = [0.7, 0.2, 0.1]`
- `t = 50`, embedded as `[0.1, -0.3, 0.4, 0.2]`

Concatenated input: `[0.2, -0.5, 0.8, 0.1, 0.4, 0.3, 0.3, 0.7, 0.2, 0.1, 0.1, -0.3, 0.4, 0.2]` (14 values).

This feeds through two 256-unit hidden layers and produces `eps_pred = [0.05, -0.02, -0.03]` — the predicted noise to subtract.

### f) Code Walkthrough

```python
def build_dapm_diffusion(latent_dim, num_classes, T=100, t_embed_dim=32, hidden_dim=256):
    z_in   = keras.Input(shape=(latent_dim,),   name='diff_z_in')      # latent code
    y_t_in = keras.Input(shape=(num_classes,),  name='diff_y_t')       # current noisy state
    f_in   = keras.Input(shape=(num_classes,),  name='diff_guidance')  # classifier guidance
    t_in   = keras.Input(shape=(1,), dtype='int32', name='diff_t')     # timestep index

    t_emb  = layers.Embedding(input_dim=T + 1, output_dim=t_embed_dim,
                               name='diff_t_embed')(t_in)  # embed timestep as dense vector
    t_emb  = layers.Flatten(name='diff_t_flat')(t_emb)    # flatten to (t_embed_dim,)

    x      = layers.Concatenate(name='diff_concat')([z_in, y_t_in, f_in, t_emb])  # combine all
    x      = layers.Dense(hidden_dim, activation='relu', name='diff_h1')(x)
    x      = layers.Dense(hidden_dim, activation='relu', name='diff_h2')(x)
    eps_pred = layers.Dense(num_classes, activation='linear',
                             name='diff_eps_pred')(x)  # noise prediction (no activation)

    return keras.Model([z_in, y_t_in, f_in, t_in], eps_pred, name='dapm_full_diffusion')
```

### g) Output & Interpretation

At each step, `eps_pred` estimates how much noise to remove from `y_t`. Across T steps, the denoiser progressively refines a pure-noise vector into a meaningful class-probability distribution. The conditioning on `z` and `guidance` ensures the output is consistent with the input patch's semantics.

### h) Limitations

- The denoiser's architecture (MLP) is simpler than state-of-the-art U-Net denoisers used for images; it may underfit complex distributions.
- The model is conditioned on `z_mu`, not a stochastic latent sample — the stochasticity comes from the starting noise `y_0 ~ N(0, I)` and intermediate step noise.
- Longer diffusion chains (larger T) give smoother samples but increase inference time linearly.
- The denoiser was trained with a specific noise schedule; mismatches during inference would degrade quality.

---

## Method 8 — Cosine / Linear Beta Schedule

### a) What it is

> The beta schedule controls how much noise is added at each step of the forward diffusion process. Think of it as a recipe for gradually filling a glass of clear water (the clean signal) with ink (noise) — a linear schedule adds ink at a constant rate, while a cosine schedule adds it slowly at first, then faster. Here, a linear beta schedule is used, going from nearly zero noise at step 1 to significant noise at step T.

The beta schedule defines the noise variance at each of the T diffusion timesteps.

### b) Why it's used here

The reverse diffusion sampler needs to know exactly how much noise was added at each forward step in order to subtract the right amount. The betas, alphas, and cumulative alpha bars encode this information.

### c) How it works — Step by step

1. Create T evenly-spaced beta values between `beta_start` (e.g., 1e-4) and `beta_end` (e.g., 0.02).
2. Compute `alpha[t] = 1 - beta[t]` — the signal retention fraction at each step.
3. Compute `alpha_bar[t] = product(alpha[0], ..., alpha[t])` — the cumulative signal retention from step 0 to t.

```
betas[t]      = linspace(beta_start, beta_end, T)[t]
alphas[t]     = 1 - betas[t]
alpha_bars[t] = alpha_bars[t-1] * alphas[t]    (cumulative product)
```

### d) ASCII Flow Diagram

```
T steps, beta_start, beta_end
        |
        v
betas = linspace(beta_start, beta_end, T)
        |
        v
alphas = 1 - betas
        |
        v
alpha_bars = cumprod(alphas)    (running product: alpha_bars[t] = prod of alphas[0..t])
```

### e) Worked Numerical Example

`T = 5`, `beta_start = 0.1`, `beta_end = 0.3`.

- `betas = [0.10, 0.15, 0.20, 0.25, 0.30]`
- `alphas = [0.90, 0.85, 0.80, 0.75, 0.70]`
- `alpha_bars = [0.90, 0.765, 0.612, 0.459, 0.321]`

By step 5, only 32.1% of the original signal remains — the rest is noise.

### f) Code Walkthrough

```python
def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    betas      = np.linspace(beta_start, beta_end, T, dtype=np.float32)  # T evenly spaced values
    alphas     = 1.0 - betas                                              # signal retention per step
    alpha_bars = np.cumprod(alphas)                                       # cumulative product
    return betas, alphas, alpha_bars
```

### g) Output & Interpretation

Returns three arrays of length T. The reverse diffusion sampler indexes into these at each step to correctly scale the denoiser's output. `alpha_bars[t]` approaching 0 as `t → T` means the forward process has added so much noise that the signal is nearly destroyed — which is why reverse diffusion starts from pure noise.

### h) Limitations

- The linear schedule is simple but less efficient than cosine schedules in practice; cosine schedules spend more time in intermediate noise levels where the denoiser does most of its useful work.
- The schedule must match the one used during training exactly — even small differences cause inference quality to degrade.
- For class-probability outputs (bounded [0, 1]), standard Gaussian diffusion may not be the most appropriate choice; clipping/thresholding could introduce artefacts.

---

## Method 9 — Reverse Diffusion Sampling

### a) What it is

> Picture a sculptor starting with a featureless block of marble (pure noise) and removing material step by step until a figure emerges. Each step, the denoiser looks at the current rough shape, the latent description of the pixel, and the classifier's suggestion, and chisels away a small amount of noise. After T such steps, the marble becomes a class-probability vector. The element of randomness in each chisel stroke (the added noise in intermediate steps) is what makes each run produce a slightly different result.

The reverse diffusion process iteratively denoises a random initial vector `y_T ~ N(0, I)` over T steps to produce a clean class-probability vector, conditioned on `z` and guidance.

### b) Why it's used here

This is the core sampling mechanism of the DAPM. By running it 30 times, the pipeline generates a *distribution* of class-probability vectors for each pixel, enabling uncertainty quantification.

### c) How it works — Step by step

1. Initialise `y ~ N(0, I)` of shape `(n, num_classes)` — pure noise for `n` pixels simultaneously.
2. For `step` from T down to 1:
   - Run the compiled denoiser: `eps_pred = denoiser(z, y, guidance, step)`.
   - Look up the schedule values `alpha`, `alpha_bar`, `beta` for this step.
   - Compute the de-noising coefficient: `coef = (1 - alpha) / sqrt(1 - alpha_bar)`.
   - Update: `y = (y - coef * eps_pred) / sqrt(alpha)`.
   - If `step > 1`, add scheduled noise: `y = y + sqrt(beta) * eps`, where `eps ~ N(0, I)`.
3. After step 1 (no noise added), apply softmax to produce class probabilities.

```
y_update = (y - coef * eps_pred) / sqrt(alpha)
coef     = (1 - alpha) / sqrt(1 - alpha_bar + epsilon)
```

### d) ASCII Flow Diagram

```
y_T ~ Normal(0, I)    z (latent)    guidance (classifier probs)
        |
        v
  ┌── for step T → 1 ──────────────────────────────────────────┐
  │                                                             │
  │   eps_pred = Denoiser(z, y, guidance, step)                │
  │   y = (y - coef * eps_pred) / sqrt(alpha)                  │
  │   if step > 1: y += sqrt(beta) * Normal(0, I)              │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
        |
        v
  softmax(y) → class probabilities (num_classes,)
```

### e) Worked Numerical Example

`num_classes = 3`, `T = 3`, processing 1 pixel.

Initial: `y = [0.5, -1.2, 0.8]`

**Step 3:** `alpha=0.7`, `alpha_bar=0.3`, `beta=0.3`, `coef = (1-0.7)/sqrt(1-0.3) = 0.3/0.837 = 0.358`.
`eps_pred = [0.1, -0.5, 0.4]` (from denoiser).
`y = ([0.5, -1.2, 0.8] - 0.358*[0.1, -0.5, 0.4]) / sqrt(0.7)`
`  = ([0.464, -1.021, 0.657]) / 0.837 = [0.554, -1.220, 0.785]`
Add noise (step > 1): `y = [0.554, -1.220, 0.785] + sqrt(0.3)*[0.3, 0.1, -0.2] = [0.718, -1.165, 0.676]`

**Steps 2, 1:** Repeated similarly; at step 1, no noise is added.

Final softmax: `softmax([...])` → class probs.

### f) Code Walkthrough

```python
def reverse_diffusion(bundle, z_np, guidance_np):
    T          = int(bundle['T'])
    alphas     = bundle['alphas']      # per-step alpha values
    alpha_bars = bundle['alpha_bars']  # cumulative products
    betas      = bundle['betas']       # per-step beta values
    n          = z_np.shape[0]         # number of pixels in batch

    z_tf        = tf.constant(z_np, dtype=tf.float32)
    guidance_tf = tf.constant(guidance_np, dtype=tf.float32)
    y           = tf.random.normal((n, nc), dtype=tf.float32)  # start from pure noise

    for step in range(T, 0, -1):                               # T steps counting DOWN
        t_arr    = tf.cast(tf.fill((n, 1), step), tf.int32)
        eps_pred = _diffusion_step_compiled(diffusion, z_tf, y, guidance_tf, t_arr)

        alpha     = float(alphas[step - 1])
        alpha_bar = float(alpha_bars[step - 1])
        beta      = float(betas[step - 1])
        coef      = (1.0 - alpha) / max(np.sqrt(1.0 - alpha_bar), 1e-8)  # de-noise scale
        y         = (y - coef * eps_pred) / max(np.sqrt(alpha), 1e-8)    # DDPM update

        if step > 1:
            noise = tf.random.normal(tf.shape(y), dtype=tf.float32)
            y     = y + np.sqrt(max(beta, 1e-8)) * noise  # add stochastic noise

    return softmax_np(y.numpy(), axis=-1)  # convert final state to probabilities
```

### g) Output & Interpretation

Returns an array of shape `(n, num_classes)` with class probabilities for each of the `n` pixels in the current batch. Each call to this function with the same inputs but different random seeds gives a different output — this variability across 30 calls is what the t-test measures.

### h) Limitations

- Inference runs T full denoising steps per sample; with T=100 and N_SAMPLES=30, this is 3000 denoiser forward passes per chunk — computationally expensive.
- The `@tf.function` compilation helps speed but requires careful management of dynamic shapes.
- Standard DDPM sampling is slower than accelerated samplers (DDIM, DPM-Solver); these were not used here.
- The softmax applied at the end is a heuristic to convert an unbounded final state to probabilities; it assumes the model was trained to produce logit-like outputs at step 0.

---

## Method 10 — Stochastic Latent Tiling (sample_dapm_chunk)

### a) What it is

> For each patch, we ask the VAE encoder "where is this patch in latent space?" — it gives us a mean position `z_mu` and a spread `std`. Then, for each of the 30 samples, we throw a dart randomly around that position (adding `std * random_noise` to `z_mu`), producing 30 slightly different latent codes. Each code then runs through the full reverse diffusion process to get one class-probability estimate. The 30 estimates form a mini-distribution.

`sample_dapm_chunk` produces `n_samples` stochastic class-probability matrices for a chunk of input patches by tiling the latent distribution and running reverse diffusion on all samples in a single batched call.

### b) Why it's used here

This is the most computationally efficient way to generate multiple stochastic samples. Instead of running the encoder 30 times (which would give the same `z_mu` each time since the encoder is deterministic given fixed weights), the tiling approach samples from the encoder's predicted distribution by drawing 30 different `eps` vectors and adding them to `z_mu`, scaled by `std`.

### c) How it works — Step by step

1. Run the feature extractor on the chunk: `feat = feature_extractor(x_chunk)`.
2. Run the encoder: get `z_mu`, `z_logvar`, and compute `std = exp(0.5 * z_logvar)`.
3. Run the classifier on `z_mu` to get `guidance_np` (deterministic guidance signals).
4. **Tile:** Repeat `z_mu`, `std`, and `guidance_np` each `n_samples` times along axis 0 → shape `(n_samples * n_points, latent_dim)`.
5. **Sample:** Draw `eps ~ N(0, I)` of the same tiled shape; compute `z_all = z_mu_tiled + std_tiled * eps`.
6. Run `reverse_diffusion` on the entire tiled batch `z_all` in one call → shape `(n_samples * n_points, num_classes)`.
7. Reshape output to `(n_samples, n_points, num_classes)`.

```
z_all = z_mu_tiled + std_tiled * eps     (vectorised across all samples and points)
probs = reverse_diffusion(z_all, guidance_tiled)
probs.reshape(n_samples, n_points, num_classes)
```

### d) ASCII Flow Diagram

```
x_chunk (chunk_size, 9, 9, 6)
        |
        v
   Feature Extractor → feat (chunk_size, feature_dim)
        |
        v
   VAE Encoder → z_mu (chunk, latent_dim), std (chunk, latent_dim)
        |
        |── Classifier → guidance (chunk, num_classes)
        v
   Tile x N_SAMPLES → z_mu_tiled (N*chunk, latent_dim)
        |
        v
   eps ~ N(0,I)  →  z_all = z_mu_tiled + std_tiled * eps
        |
        v
   Reverse Diffusion (single batched call)
        |
        v
   probs_flat (N_SAMPLES * chunk, num_classes)
        |
        v
   reshape → (N_SAMPLES, chunk_size, num_classes)
```

### e) Worked Numerical Example

`n_points = 2`, `n_samples = 3`, `latent_dim = 2`, `num_classes = 3`.

Suppose: `z_mu = [[1.0, 0.5], [−0.5, 2.0]]`, `std = [[0.1, 0.2], [0.3, 0.1]]`.

Tiled (3×2=6 rows):
```
z_mu_tiled = [[1.0, 0.5], [−0.5, 2.0],   # sample 1
               [1.0, 0.5], [−0.5, 2.0],   # sample 2
               [1.0, 0.5], [−0.5, 2.0]]   # sample 3
```

Draw eps (6×2): each row is different.

`z_all = z_mu_tiled + std_tiled * eps` → 6 unique latent codes.

Reverse diffusion → 6 probability vectors → reshape → `(3, 2, 3)`.

### f) Code Walkthrough

```python
def sample_dapm_chunk(bundle, x_chunk, n_samples, batch_size):
    n_points = x_chunk.shape[0]
    fe, enc, clf = bundle['feature_extractor'], bundle['encoder'], bundle['classifier']

    x_tf              = tf.constant(x_chunk, dtype=tf.float32)
    feat              = fe(x_tf, training=False)             # frozen feature vectors
    z_mu, z_logvar, _ = enc(feat, training=False)            # encode to distribution params
    z_mu_np           = z_mu.numpy()
    std_np            = np.exp(0.5 * z_logvar.numpy())       # std = exp(0.5 * logvar)
    guidance_np       = clf(z_mu, training=False).numpy()    # deterministic guidance

    # Tile all arrays n_samples times along the batch dimension
    z_mu_tiled     = np.tile(z_mu_np, (n_samples, 1))       # (n_samples*n_points, latent_dim)
    std_tiled      = np.tile(std_np, (n_samples, 1))
    guidance_tiled = np.tile(guidance_np, (n_samples, 1))

    # Sample n_samples different noise vectors → n_samples different latent codes
    eps   = np.random.normal(size=z_mu_tiled.shape).astype(np.float32)
    z_all = (z_mu_tiled + std_tiled * eps).astype(np.float32)

    # Run full reverse diffusion on the mega-batch; reshape to (n_samples, n_points, nc)
    probs_flat = reverse_diffusion(bundle, z_all, guidance_tiled)
    return probs_flat.reshape(n_samples, n_points, nc)
```

### g) Output & Interpretation

Returns shape `(30, chunk_size, num_classes)` — 30 independent class-probability estimates for each pixel in the chunk. The spread of these 30 estimates is the raw material for the uncertainty estimator.

### h) Limitations

- The tiling approach multiplies the batch size by `n_samples` — memory usage scales as `O(n_samples * chunk_size)`.
- Guidance is computed once from `z_mu` and tiled; it does not vary across samples, which may underestimate uncertainty.
- The eps are sampled independently across the `latent_dim` dimensions; correlations in the encoder's posterior are not modelled.
- Very large `n_samples` reduces variance in the uncertainty estimate but increases compute time linearly.

---

## Method 11 — Welch t-Test Uncertainty Estimation

### a) What it is

> After generating 30 class-probability samples per pixel, we need to decide: is the model sure about its top prediction? We do this by treating the 30 top-1 probability scores as one group and the 30 top-2 scores as another, then asking: "Are these two groups statistically different?" If yes (low p-value), the model reliably prefers class 1 over class 2 — it's certain. If no (high p-value), the two classes are statistically indistinguishable — the pixel is *uncertain*.

A Welch two-sample t-test is applied per pixel to the top-1 versus top-2 class probability distributions across the N samples. Pixels where the top-two distributions are not significantly different are flagged as uncertain.

### b) Why it's used here

The Welch t-test is a well-calibrated, assumption-light statistical test for comparing two groups with potentially different variances. It avoids the equal-variance assumption of Student's t-test, which is appropriate here since different classes may have different variability in their probability estimates.

### c) How it works — Step by step

1. Compute `mean_prob[i, c]` — the average probability assigned to class `c` across 30 samples for pixel `i`.
2. Sort classes by `mean_prob` descending → find top-1 class `c1` and top-2 class `c2`.
3. Extract two groups: `g1 = probs_samples[:, i, c1]` and `g2 = probs_samples[:, i, c2]` (each length 30).
4. Run Welch t-test on `g1` vs `g2` via `scipy.stats.ttest_ind(g1, g2, equal_var=False)`.
5. If `p_value > P_THRESH` (default 0.05): pixel is **uncertain** (the null hypothesis — equal means — cannot be rejected).
6. Record `pred_class = c1`, `p_value`, and `gap = mean(g1) - mean(g2)`.

```
uncertain[i] = True  if  p_value(g1, g2) > P_THRESH
uncertain[i] = False if  p_value(g1, g2) <= P_THRESH
```

### d) ASCII Flow Diagram

```
probs_samples (30, n_points, num_classes)
        |
        v
  mean_prob = mean over 30 samples   → (n_points, num_classes)
        |
        v
  argsort descending → top-1 class c1, top-2 class c2   per pixel
        |
        v
  g1 = probs_samples[:, i, c1]  (length 30)
  g2 = probs_samples[:, i, c2]  (length 30)
        |
        v
  Welch t-test(g1, g2) → t-statistic, p-value
        |
        v
  p_value > P_THRESH?
   YES → uncertain = True
   NO  → uncertain = False
```

### e) Worked Numerical Example

`N_SAMPLES = 5` (simplified), `num_classes = 3`, for pixel 0.

```
probs (5 samples × 3 classes):
  [[0.70, 0.20, 0.10],
   [0.65, 0.25, 0.10],
   [0.68, 0.22, 0.10],
   [0.72, 0.18, 0.10],
   [0.69, 0.21, 0.10]]
```

`mean_prob = [0.688, 0.212, 0.100]` → `c1 = 0`, `c2 = 1`.

`g1 = [0.70, 0.65, 0.68, 0.72, 0.69]`, mean = 0.688, std ≈ 0.025.
`g2 = [0.20, 0.25, 0.22, 0.18, 0.21]`, mean = 0.212, std ≈ 0.025.

The two groups are clearly separated → Welch t-test gives a very small p-value (e.g., p < 0.001).
Since `p < 0.05`, `uncertain = False`.

**Uncertain case:** If both groups had means around 0.40 and similar spreads, the t-test might give `p = 0.3 > 0.05` → `uncertain = True`.

### f) Code Walkthrough

```python
def compute_dapm_ttest_uncertainty_chunk(probs_samples, p_thresh=0.05):
    n_samples, n_points, n_classes = probs_samples.shape

    mean_prob = np.mean(probs_samples, axis=0)           # average over 30 samples
    order     = np.argsort(-mean_prob, axis=1)           # descending rank per pixel

    for i in range(n_points):
        c1, c2 = int(order[i, 0]), int(order[i, 1])     # top-1 and top-2 class indices
        g1, g2 = probs_samples[:, i, c1], probs_samples[:, i, c2]  # 30 samples each

        _, pval = _safe_ttest_ind(g1, g2)                # Welch t-test
        pred_class[i]     = c1                           # predicted as top-1 class
        p_values[i]       = pval                         # save p-value for mapping
        mean_gap[i]       = float(np.mean(g1) - np.mean(g2))  # probability gap
        uncertain_mask[i] = bool(pval > p_thresh)        # flag uncertain pixels
```

```python
def _safe_ttest_ind(g1, g2):
    # Handle edge cases: constant inputs (e.g., all zeros or identical distributions)
    if np.allclose(g1, g1[0]) and np.allclose(g2, g2[0]):
        if abs(float(np.mean(g1)) - float(np.mean(g2))) < 1e-8:
            return 0.0, 1.0    # identical constant groups → p=1 (maximally uncertain)
        return np.inf, 0.0     # different constants → completely separable → p=0

    out  = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')  # Welch t-test
    tval = float(np.nan_to_num(out.statistic, nan=0.0, posinf=np.inf, neginf=-np.inf))
    pval = float(np.nan_to_num(out.pvalue,    nan=1.0, posinf=1.0,   neginf=0.0))
    return tval, float(np.clip(pval, 0.0, 1.0))
```

### g) Output & Interpretation

Per pixel:
- `pred_class`: the index of the top-1 class by mean probability.
- `uncertain_mask`: `True` if the top-two classes are statistically indistinguishable.
- `p_values`: raw p-value from the Welch test — lower means more confident.
- `top1_top2_mean_gap`: how much higher the top-1 mean probability is compared to top-2 — larger gaps indicate higher confidence.

A **low uncertainty rate** across the scene suggests the model is globally confident. High per-class uncertainty rates highlight spectrally ambiguous classes.

### h) Limitations

- The t-test requires N_SAMPLES to be large enough for valid inference; with only 30 samples and non-normal distributions, the test may not be well-calibrated.
- The t-test only compares top-1 vs top-2; it ignores uncertainty arising from ambiguity among three or more classes.
- The significance threshold `P_THRESH = 0.05` is a conventional but arbitrary choice — different thresholds produce different uncertainty maps.
- The t-test assumes independence between samples, but samples from the same VAE distribution may be correlated.

---

## Method 12 — Chunked Scene Inference with NPZ Caching

### a) What it is

> Running the full DAPM pipeline on all 101,310 pixels at once would require enormous GPU memory. Instead, the scene is processed in "chunks" of 1,000 pixels at a time. Each chunk's results are saved to a compressed `.npz` file on disk immediately after processing. If the notebook is interrupted and restarted, completed chunks are loaded from disk rather than recomputed — like a save-game system for expensive computations.

The main inference loop processes the scene pixel-by-pixel in fixed-size chunks, with chunk-level disk caching for fault tolerance and memory efficiency.

### b) Why it's used here

A 330×307 scene has 101,310 pixels. With `N_SAMPLES=30`, each chunk of 1,000 pixels requires running the DAPM 30,000 times (1,000 × 30 diffusion chains of T steps each). This is memory-intensive and takes substantial compute time. Chunking keeps GPU memory bounded; caching prevents costly recomputation if the Colab runtime is interrupted.

### c) How it works — Step by step

1. Flatten the scene into `coords_scene` of shape `(101310, 2)` — all `(row, col)` pairs.
2. For each chunk `[st:ed]` of size `SCENE_CHUNK_SIZE = 1000`:
   a. Check if `CHUNK_DIR / '{model_key}_chunk_{st}_{ed}.npz'` exists.
   b. If yes: load arrays from cache and fill the output arrays.
   c. If no: extract patches, run `sample_dapm_chunk`, run `compute_dapm_ttest_uncertainty_chunk`, fill output arrays, and save a compressed `.npz` to disk.
3. After all chunks, call `build_dapm_outputs` to assemble maps, metrics, and plots.

```
total_pixels = 101,310
chunk_size   = 1,000
total_chunks = ceil(101,310 / 1,000) = 102
```

### d) ASCII Flow Diagram

```
coords_scene (101310, 2)
        |
        v
  split into 102 chunks of 1000 pixels each
        |
  ┌─────┴────────────────────────────────────────────────┐
  │  for each chunk [st:ed]:                             │
  │                                                      │
  │   .npz exists? ──YES──> load from cache              │
  │        |                                             │
  │        NO                                            │
  │        |                                             │
  │        v                                             │
  │   extract patches (1000, 9, 9, 6)                   │
  │        |                                             │
  │        v                                             │
  │   sample_dapm_chunk → probs (30, 1000, num_classes)  │
  │        |                                             │
  │        v                                             │
  │   compute_ttest_uncertainty → pred, mask, p, gap     │
  │        |                                             │
  │        v                                             │
  │   save chunk .npz                                    │
  └──────────────────────────────────────────────────────┘
        |
        v
  build_dapm_outputs (maps, metrics, plots)
```

### e) Worked Numerical Example

Scene: 5 pixels, chunk_size=2. Chunks: [0:2], [2:4], [4:5].

- Chunk `chunk_0_2.npz` doesn't exist → compute → save.
- Chunk `chunk_2_4.npz` exists → load instantly.
- Chunk `chunk_4_5.npz` doesn't exist → compute → save.

Resuming after interruption (say, after chunk [0:2]): only [0:2] is cached, so only [2:4] and [4:5] need computing.

### f) Code Walkthrough

```python
for chunk_idx, st in enumerate(range(0, n_points, SCENE_CHUNK_SIZE)):
    ed         = min(st + SCENE_CHUNK_SIZE, n_points)       # handle last partial chunk
    chunk_file = CHUNK_DIR / f'{model_key}_chunk_{st}_{ed}.npz'

    if chunk_file.exists():                                  # cache hit: load and skip
        chunk = np.load(chunk_file)
        pred_class_all[st:ed]     = chunk['pred_class']
        uncertain_mask_all[st:ed] = chunk['uncertain_mask']
        p_values_all[st:ed]       = chunk['p_values']
        gaps_all[st:ed]           = chunk['gaps']
        continue

    # Cache miss: full compute pipeline
    coords_chunk  = coords_scene[st:ed]
    x_chunk       = extract_patches_from_coords(x_img, coords_chunk, patch_size=PATCH_SIZE)
    probs_samples = sample_dapm_chunk(bundle, x_chunk, n_samples=N_SAMPLES, batch_size=BATCH_SIZE)
    out_chunk     = compute_dapm_ttest_uncertainty_chunk(probs_samples, p_thresh=P_THRESH)

    # Fill global output arrays
    pred_class_all[st:ed]     = out_chunk['pred_class']
    uncertain_mask_all[st:ed] = out_chunk['uncertain_mask']
    p_values_all[st:ed]       = out_chunk['p_values']
    gaps_all[st:ed]           = out_chunk['top1_top2_mean_gap']

    # Save to disk with compression
    np.savez_compressed(chunk_file,
        pred_class    = out_chunk['pred_class'],
        uncertain_mask= out_chunk['uncertain_mask'],
        p_values      = out_chunk['p_values'],
        gaps          = out_chunk['top1_top2_mean_gap']
    )
```

### g) Output & Interpretation

After all chunks, four flat arrays of length 101,310 are assembled:
- `pred_class_all`: predicted class index per pixel.
- `uncertain_mask_all`: boolean uncertain flag per pixel.
- `p_values_all`: Welch t-test p-value per pixel.
- `gaps_all`: top-1 minus top-2 mean probability gap per pixel.

These are reshaped to `(330, 307)` maps in `build_dapm_outputs` for visualisation.

### h) Limitations

- The NPZ cache is keyed by exact `(st, ed)` range and `model_key` only — if `N_SAMPLES`, `P_THRESH`, or the model weights change, stale caches must be manually deleted.
- A very small `SCENE_CHUNK_SIZE` increases cache overhead; a very large size may exceed GPU memory.
- No parallel chunk processing — chunks are processed strictly sequentially.
- The cache directory grows indefinitely; cleanup is not automated.

---

## Results & Comparisons

The notebook produces the following outputs for each of the three models (AlexNet, GFNet, ViT):

**Per-model summary metrics (saved to `dapm_full_summary_metrics.csv`):**

| Metric | Description |
|---|---|
| `overall_accuracy_labeled` | Fraction of labelled pixels correctly classified by top-1 prediction |
| `uncertain_rate` | Fraction of all 101,310 pixels flagged as uncertain (p > 0.05) |
| `certainty_rate` | 1 - uncertain_rate |
| `mean_p_value` | Average Welch t-test p-value across all pixels |
| `median_p_value` | Median Welch t-test p-value |
| `mean_top1_top2_gap` | Average probability gap between top-1 and top-2 class |
| `mean_per_class_accuracy` | Macro-averaged per-class accuracy on labelled pixels |
| `mean_per_class_uncertainty` | Macro-averaged per-class uncertainty rate |

**Per-class metrics (saved to `dapm_full_per_class_metrics.csv`):**

| Metric | Description |
|---|---|
| `class_accuracy` | Fraction of labelled pixels of this class correctly predicted |
| `uncertainty_rate` | Fraction of this class's pixels flagged uncertain |
| `mean_p_value` | Mean p-value for this class's pixels |
| `mean_gap` | Mean top-1 vs top-2 gap for this class's pixels |

**Summary comparison table structure:**

| Method | OA (labelled) | Uncertain Rate | Mean P-Value | Mean Gap | Notes |
|---|---|---|---|---|---|
| AlexNet + DAPM | (from run) | (from run) | (from run) | (from run) | CNN baseline |
| GFNet + DAPM | (from run) | (from run) | (from run) | (from run) | Freq-domain transformer |
| ViT + DAPM | (from run) | (from run) | (from run) | (from run) | Vision transformer |

> **Note:** Specific numeric results are not shown in the provided notebook (no cell outputs were included). The above table structure mirrors what the notebook produces. Fill in from the saved `dapm_full_summary_metrics.csv`.

**Spatial outputs per model:**
- Certain vs Uncertain binary map (yellow/dark-blue)
- Class + Uncertainty mask map (colour-coded with grey for uncertain)
- Pixel count bar chart
- P-value histogram with threshold line
- Top-1 vs Top-2 probability gap histogram
- Per-class accuracy bar chart
- Per-class uncertainty rate bar chart
- Per-class mean p-value bar chart

All outputs are assembled into a multi-sheet Excel workbook (`dapm_full_reports_all_models.xlsx`) and copied to Google Drive.

---

## Academic Paper Summary

### Problem Statement

Quantifying predictive uncertainty in deep learning-based multispectral land-cover classification is critical for applications where erroneous predictions carry high costs, such as environmental monitoring and precision agriculture. Standard softmax classifiers produce overconfident point estimates that do not reflect the model's epistemic or aleatoric uncertainty. This work addresses the challenge of producing spatially explicit, statistically grounded uncertainty maps for pixel-wise remote sensing classification using a novel Diffusion-Augmented Probabilistic Model (DAPM) framework.

### Methodology

**Feature Extraction.** Three pre-trained convolutional and transformer-based models — an AlexNet-style CNN, a Global Filter Network (GFNet), and a Vision Transformer U-Net (ViT-UNet) — were employed as frozen feature extractors. For each input pixel, a 9×9 spatial patch of six spectral bands was extracted and passed through the penultimate layer of each base model, yielding a compact, discriminative feature representation.

**Variational Autoencoder Encoder.** A VAE encoder comprising two fully-connected hidden layers maps the frozen feature vector to a Gaussian latent distribution parameterised by mean `z_mu` and log-variance `z_logvar`, via the reparameterisation trick `z = z_mu + exp(0.5 * z_logvar) * eps`, where `eps ~ N(0, I)`. This probabilistic encoding enables stochastic sampling from the learned latent manifold.

**Conditional Diffusion Denoiser.** A conditional DDPM denoiser, structured as a four-input MLP accepting the latent code, the current noisy class-probability state, soft classifier guidance, and an embedded timestep, iteratively refines random noise into a class-probability distribution over T reverse diffusion steps. A linear beta noise schedule parameterises the forward diffusion process. Classifier guidance is provided by a separate latent-space MLP that maps `z_mu` to class probabilities via a softmax head.

**Uncertainty Quantification via Welch t-Test.** For each pixel, N=30 stochastic latent codes are sampled by tiling `z_mu` and perturbing with random noise scaled by the encoder's predicted standard deviation. Each code is processed by the full reverse diffusion chain, yielding 30 class-probability vectors. Uncertainty is then estimated by a Welch two-sample independent t-test comparing the top-1 versus top-2 class probability distributions across the 30 samples. Pixels for which this comparison yields a p-value exceeding 0.05 are classified as uncertain.

### Experimental Setup

**Dataset:** A single 330×307 six-band multispectral image with a partial ground-truth label map specifying N land-cover classes (exact class count and dataset identity not specified in the notebook; determined at runtime from the label file).

**Patch size:** 9×9 pixels, extracted with edge-replication padding.

**Diffusion parameters:** T steps (model-specific, loaded from per-model JSON config), linear schedule from `beta_start` to `beta_end`.

**Sampling:** N_SAMPLES=30 stochastic draws per pixel.

**Uncertainty threshold:** P_THRESH=0.05 (Welch t-test significance level).

**Evaluation:** Overall accuracy and per-class accuracy on labelled pixels; uncertainty rate and mean p-value across the full scene; Top-1 vs Top-2 mean probability gap.

**Baselines:** The three DAPM configurations (AlexNet+DAPM, GFNet+DAPM, ViT+DAPM) are compared against each other. No non-probabilistic baseline (e.g., deterministic softmax) is explicitly included.

### Results Summary

The DAPM framework was applied to all three backbone architectures. Outputs including spatial uncertainty maps, per-class accuracy metrics, and uncertainty rates were generated for each model and aggregated in a cross-model comparison workbook. Overall accuracy on labelled pixels, uncertainty rates, and mean p-values vary across the three DAPM configurations, with differences attributable to the differing quality of the feature representations produced by AlexNet, GFNet, and ViT-UNet respectively. Specific numerical results are dependent on the trained model weights and label data and are available in the saved `dapm_full_summary_metrics.csv`.

### Conclusion

This work demonstrates a practical pipeline for uncertainty-aware remote sensing classification by coupling frozen deep learning feature extractors with a VAE-diffusion generative model and a non-parametric statistical uncertainty criterion. The modular DAPM framework is backbone-agnostic and can be applied to any pre-trained model with a penultimate-layer feature representation. Limitations include the computational cost of T-step reverse diffusion for each of 30 samples across 101,310 pixels, the reliance on a 0.05 significance threshold, and the absence of a calibration study validating that flagged uncertain pixels correspond to genuinely ambiguous land-cover boundaries. Future work should explore faster diffusion samplers (DDIM, DPM-Solver), calibration of the uncertainty threshold to ground-truth ambiguity annotations, and extension to multi-temporal or hyperspectral imagery.

---

## References

[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 6840–6851.

[2] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *International Conference on Learning Representations (ICLR)*.

[3] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global Filter Networks for Image Classification. *Advances in Neural Information Processing Systems (NeurIPS)*, 34.

[4] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*.

[5] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 25.

[6] Welch, B. L. (1947). The Generalization of Student's Problem When Several Different Population Variances are Involved. *Biometrika*, 34(1–2), 28–35.

[7] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *International Conference on Machine Learning (ICML)*, 1050–1059.

[8] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

[9] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.

[10] Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning. *USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 12, 265–283.
