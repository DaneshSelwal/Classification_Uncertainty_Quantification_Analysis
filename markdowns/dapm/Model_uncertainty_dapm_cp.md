# 1.0 — Setup & Imports

Mount Google Drive (when running in Colab), install required packages, and import all libraries
used throughout the notebook. Identical to the baseline uncertainty-comparison notebook.

```python
import os
import sys
import subprocess

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xlsxwriter', 'openpyxl'], check=True)
```

```python
import io
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from openpyxl import load_workbook

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print('Python:', sys.version.split()[0])
print('TensorFlow:', tf.__version__)
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
```

# 2.0 — Configuration

All project-wide constants, file paths, and hyperparameters. `DAPM_DIR` points to the saved
DAPM artifacts (encoder / classifier / diffusion weights + JSON configs) produced by the DAPM
training notebook; `MODEL_DIR` points to the underlying baseline `.keras` backbones that the
DAPM feature extractors are built from. All output filenames are prefixed `dapm_conformal_`
so they never collide with the plain-model conformal run or the DAPM t-test run.

```python
# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ── Directory layout ─────────────────────────────────────────
PROJECT_ROOT = Path('/content/drive/My Drive/Classification')

# Baseline trained .keras backbones (used as DAPM feature extractors)
MODEL_DIR = PROJECT_ROOT / 'baseline' / 'models'

# Saved DAPM artifacts (encoder / classifier / diffusion weights + configs)
DAPM_DIR = PROJECT_ROOT / 'dapm' / 'models' / 'dapm_full_artifacts'

RESULTS_DIR = PROJECT_ROOT / 'dapm' / 'results'
OUTPUT_DIR  = RESULTS_DIR / 'conformal_uncertainty_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / 'data'

# ── Input files ──────────────────────────────────────────────
DATA_FILE  = DATA_DIR / 'data.csv'
LABEL_FILE = DATA_DIR / 'ref.csv'

MODEL_KEYS = ['AlexNet_CNN', 'GFNet', 'ViT_UNet']
MODEL_FILES = {
    'AlexNet_CNN': MODEL_DIR / 'AlexNet_CNN_best.keras',
    'GFNet':       MODEL_DIR / 'GFNet_best.keras',
    'ViT_UNet':    MODEL_DIR / 'ViT_UNet_best.keras',
}
MODEL_NAME_MAP = {
    'AlexNet_CNN': 'AlexNet',
    'GFNet':       'GFNet',
    'ViT_UNet':    'ViT',
}

# ── Image / patch geometry ───────────────────────────────────
H, W, B    = 330, 307, 6
PATCH_SIZE = 9

# ── Train / calibration split ────────────────────────────────
TRAIN_PERCENT          = 0.75
CALIB_FRACTION_OF_TEST = 0.50

# ── Method hyperparameters ───────────────────────────────────
ALPHA      = 0.05
RAPS_LAM   = 0.01
RAPS_K_REG = 1
N_CLUSTERS = 4
BATCH_SIZE = 128
EPS        = 1e-12

# ── Output paths (distinct from both the baseline-model conformal run and the
#    DAPM t-test run, so all three result sets remain easy to tell apart) ────
EXCEL_PATH           = OUTPUT_DIR / 'dapm_conformal_reports_all_models.xlsx'
SUMMARY_CSV_PATH     = OUTPUT_DIR / 'dapm_conformal_summary_metrics.csv'
PER_CLASS_CSV_PATH   = OUTPUT_DIR / 'dapm_conformal_per_class_coverage.csv'
RUN_CONFIG_JSON_PATH = OUTPUT_DIR / 'dapm_conformal_run_config.json'

for k, v in MODEL_FILES.items():
    print(f'{k}: {v}')
print('DAPM artifacts dir:', DAPM_DIR)
print('Workbook:', EXCEL_PATH)
```

# 3.0 — Environment Guards

Validates that all required data files, baseline backbone files, and DAPM artifact configs
exist before the pipeline runs. Raises clear errors if anything is missing.

```python
def assert_environment_and_files():
    """Raise informative errors if required data, backbone, or DAPM artifact files are absent."""
    if not Path('/content/drive').exists() and not Path('/Users').exists():
        raise RuntimeError('Drive/local filesystem unavailable.')

    if not DATA_FILE.exists() or not LABEL_FILE.exists():
        raise FileNotFoundError(
            f'Data files missing:\n- {DATA_FILE}\n- {LABEL_FILE}'
        )

    missing_models = [str(p) for p in MODEL_FILES.values() if not p.exists()]
    if missing_models:
        msg = 'Missing baseline model files:\n' + '\n'.join(missing_models)
        raise FileNotFoundError(msg)

    missing_dapm = [
        str(DAPM_DIR / f'{mk}_dapm_full_config.json')
        for mk in MODEL_KEYS
        if not (DAPM_DIR / f'{mk}_dapm_full_config.json').exists()
    ]
    if missing_dapm:
        msg = 'Missing DAPM artifact configs:\n' + '\n'.join(missing_dapm)
        msg += '\n\nRun the DAPM training notebook first to produce these artifacts.'
        raise FileNotFoundError(msg)

assert_environment_and_files()
print('All required data / backbone / DAPM artifact files are present.')
```

# 4.0 — Data Pipeline

Loads the multispectral image and reference labels from CSV, extracts labeled patches, and
performs train / calibration / evaluation splits. Identical to the baseline uncertainty-comparison
notebook — the same patches feed both the baseline models and the DAPM bundles.

## 4.1 — Data Loading & Patch Extraction

```python
def load_multispectral_6band(data_path, label_path, h, w, b):
    """Load and per-band normalise a multispectral image from two CSV files."""
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(h, w, b)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(h, w)

    x_norm = np.empty_like(x, dtype=np.float32)
    for bi in range(b):
        band  = x[:, :, bi]
        mn, mx = float(np.min(band)), float(np.max(band))
        denom  = max(mx - mn, 1e-8)
        x_norm[:, :, bi] = (band - mn) / denom

    return x_norm, y


def extract_labeled_patches(x_img, y_img, patch_size=9):
    """Extract square patches centred on every labeled (non-zero) pixel."""
    pad   = patch_size // 2
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    coords    = np.argwhere(y_img > 0)
    x_patches = np.empty((coords.shape[0], patch_size, patch_size, x_img.shape[-1]), dtype=np.float32)
    y_labels  = np.empty((coords.shape[0],), dtype=np.int32)

    for i, (r, c) in enumerate(coords):
        x_patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        y_labels[i]  = int(y_img[r, c]) - 1   # convert 1-based labels to 0-based

    return x_patches, y_labels


def split_calib_eval(x_test, y_test, seed=42, calib_fraction=0.5):
    """Split a test pool into calibration and evaluation sets (stratified if possible)."""
    test_size = 1.0 - calib_fraction
    try:
        x_cal, x_eval, y_cal, y_eval = train_test_split(
            x_test, y_test, test_size=test_size, random_state=seed, stratify=y_test,
        )
    except ValueError:
        x_cal, x_eval, y_cal, y_eval = train_test_split(
            x_test, y_test, test_size=test_size, random_state=seed, stratify=None,
        )
    return x_cal, x_eval, y_cal, y_eval
```

## 4.2 — Build Splits

```python
x_img, y_img = load_multispectral_6band(DATA_FILE, LABEL_FILE, H, W, B)
X_all, y_all = extract_labeled_patches(x_img, y_img, PATCH_SIZE)
num_classes  = int(np.unique(y_all).size)

_, x_test_pool, _, y_test_pool = train_test_split(
    X_all, y_all,
    train_size=TRAIN_PERCENT,
    random_state=SEED,
    stratify=y_all,
)

x_cal, x_eval, y_cal, y_eval = split_calib_eval(
    x_test_pool, y_test_pool,
    seed=SEED,
    calib_fraction=CALIB_FRACTION_OF_TEST,
)

print('x_img:', x_img.shape, ' y_img:', y_img.shape)
print('X_all:', X_all.shape, ' y_all:', y_all.shape, ' num_classes:', num_classes)
print('x_cal:', x_cal.shape, ' x_eval:', x_eval.shape)
```

# 5.0 — Custom Keras Layers

Registers all project-specific Keras layers required to deserialise the saved baseline models
(`PatchExtractor`, `PatchPositionEncoder`, `GlobalFilterLayer`, `PatchEncoderWithCLS`) plus the
`Sampling` layer used inside the DAPM VAE encoder. These must be defined before any
`keras.models.load_model` call or DAPM sub-network construction.

## 5.1 — Backbone Deserialisation Layers

```python
@tf.keras.utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    """Extract non-overlapping image patches using tf.image.extract_patches."""

    def __init__(self, patch_size=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches  = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        batch    = tf.shape(images)[0]
        n_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        patch_dim = tf.shape(patches)[-1]
        return tf.reshape(patches, [batch, n_patches, patch_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'patch_size': self.patch_size})
        return cfg


@tf.keras.utils.register_keras_serializable()
class PatchPositionEncoder(layers.Layer):
    """Project patches to an embedding dimension and add positional encodings."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection         = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        pos = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(pos)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return cfg


@tf.keras.utils.register_keras_serializable()
class GlobalFilterLayer(layers.Layer):
    """GFNet global filter layer: learnable frequency-domain filtering via FFT."""

    def __init__(self, token_side, **kwargs):
        super().__init__(**kwargs)
        self.token_side = token_side

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.w_real = self.add_weight(
            name='w_real', shape=(self.token_side, self.token_side, channels),
            initializer='glorot_uniform', trainable=True,
        )
        self.w_imag = self.add_weight(
            name='w_imag', shape=(self.token_side, self.token_side, channels),
            initializer='zeros', trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        b  = tf.shape(x)[0]
        c  = tf.shape(x)[-1]
        x2 = tf.reshape(x, [b, self.token_side, self.token_side, c])
        x_fft = tf.signal.fft2d(tf.cast(x2, tf.complex64))
        w  = tf.complex(self.w_real, self.w_imag)
        x_f = x_fft * w
        x_i = tf.math.real(tf.signal.ifft2d(x_f))
        return tf.reshape(x_i, [b, self.token_side * self.token_side, c])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'token_side': self.token_side})
        return cfg


@tf.keras.utils.register_keras_serializable()
class PatchEncoderWithCLS(layers.Layer):
    """Patch encoder that prepends a learnable [CLS] token and adds positional embeddings."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection         = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name='cls_token', shape=(1, 1, self.projection_dim),
            initializer='zeros', trainable=True,
        )
        super().build(input_shape)

    def call(self, patches):
        batch      = tf.shape(patches)[0]
        patch_proj = self.projection(patches)
        cls_tokens = tf.repeat(self.cls_token, repeats=batch, axis=0)
        x   = tf.concat([cls_tokens, patch_proj], axis=1)
        pos = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(pos)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return cfg


CUSTOM_OBJECTS = {
    'PatchExtractor':       PatchExtractor,
    'PatchPositionEncoder': PatchPositionEncoder,
    'GlobalFilterLayer':    GlobalFilterLayer,
    'PatchEncoderWithCLS':  PatchEncoderWithCLS,
}
```

## 5.2 — DAPM VAE Sampling Layer

```python
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """VAE reparameterisation trick: sample z from N(z_mu, exp(0.5 * z_logvar))."""

    def call(self, inputs):
        z_mu, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mu))
        return z_mu + tf.exp(0.5 * z_logvar) * eps


# Extend the registry so it also covers the DAPM sampling layer
CUSTOM_OBJECTS['Sampling'] = Sampling
```

# 6.0 — DAPM Model Builders & Bundle Loader

Constructs the DAPM sub-networks (feature extractor, VAE encoder, latent classifier, and
conditional diffusion denoiser) and defines `load_dapm_bundle`, which reads a saved model's
JSON config, rebuilds every sub-network, and restores its trained weights. Reused unchanged
from the DAPM full-scene uncertainty notebook.

## 6.1 — Sub-network Builders

```python
def get_feature_extractor(base_model):
    """Wrap the penultimate layer of a Keras model as a frozen feature extractor."""
    penultimate = base_model.layers[-2].output
    feat_model  = keras.Model(
        base_model.input, penultimate,
        name=f'{base_model.name}_feature_extractor'
    )
    feat_model.trainable = False
    return feat_model


def build_dapm_encoder(feature_dim, latent_dim=64, hidden_dim=256):
    """Build the VAE encoder: features → (z_mu, z_logvar, z_sample)."""
    inp     = keras.Input(shape=(feature_dim,), name='enc_feature_in')
    h       = layers.Dense(hidden_dim, activation='relu', name='enc_h1')(inp)
    h       = layers.Dense(hidden_dim, activation='relu', name='enc_h2')(h)
    z_mu    = layers.Dense(latent_dim, name='z_mu')(h)
    z_logvar = layers.Dense(latent_dim, name='z_logvar')(h)
    z       = Sampling(name='z_sample')([z_mu, z_logvar])
    return keras.Model(inp, [z_mu, z_logvar, z], name='dapm_full_encoder')


def build_dapm_classifier(latent_dim, num_classes, hidden_dim=128):
    """Build the latent-space softmax classifier: z → class probabilities."""
    inp = keras.Input(shape=(latent_dim,), name='clf_z_in')
    h   = layers.Dense(hidden_dim, activation='relu', name='clf_h1')(inp)
    out = layers.Dense(num_classes, activation='softmax', name='clf_out')(h)
    return keras.Model(inp, out, name='dapm_full_classifier')


def build_dapm_diffusion(latent_dim, num_classes, T=100, t_embed_dim=32, hidden_dim=256):
    """Build the conditional diffusion denoiser: (z, y_t, guidance, t) → eps_pred."""
    z_in   = keras.Input(shape=(latent_dim,),   name='diff_z_in')
    y_t_in = keras.Input(shape=(num_classes,),  name='diff_y_t')
    f_in   = keras.Input(shape=(num_classes,),  name='diff_guidance')
    t_in   = keras.Input(shape=(1,), dtype='int32', name='diff_t')
    t_emb  = layers.Embedding(input_dim=T + 1, output_dim=t_embed_dim, name='diff_t_embed')(t_in)
    t_emb  = layers.Flatten(name='diff_t_flat')(t_emb)
    x      = layers.Concatenate(name='diff_concat')([z_in, y_t_in, f_in, t_emb])
    x      = layers.Dense(hidden_dim, activation='relu', name='diff_h1')(x)
    x      = layers.Dense(hidden_dim, activation='relu', name='diff_h2')(x)
    eps_pred = layers.Dense(num_classes, activation='linear', name='diff_eps_pred')(x)
    return keras.Model([z_in, y_t_in, f_in, t_in], eps_pred, name='dapm_full_diffusion')


def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    """Linear beta schedule → (betas, alphas, cumulative alpha_bars)."""
    betas      = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas     = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars
```

## 6.2 — Bundle Loader

```python
def load_dapm_bundle(model_key):
    """
    Load a complete DAPM bundle for `model_key`.

    Reads the JSON config, loads the base feature extractor, builds all sub-networks,
    performs a warm-up forward pass to materialise shapes, loads saved weights,
    and returns a dict with all components plus the diffusion schedule.
    """
    cfg_path = DAPM_DIR / f'{model_key}_dapm_full_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing config: {cfg_path}')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))

    base_model_path = MODEL_FILES[model_key]
    if not base_model_path.exists():
        raise FileNotFoundError(f'Missing local base model: {base_model_path}')

    print(f'  Loading base model from: {base_model_path}')
    base_model = keras.models.load_model(
        str(base_model_path), custom_objects=CUSTOM_OBJECTS,
        compile=False, safe_mode=False
    )
    feature_extractor = get_feature_extractor(base_model)

    feature_dim     = int(cfg['feature_dim'])
    latent_dim      = int(cfg['latent_dim'])
    num_classes_cfg = int(cfg['num_classes'])

    encoder    = build_dapm_encoder(
        feature_dim, latent_dim=latent_dim,
        hidden_dim=int(cfg['decoder_hidden_dim'])
    )
    classifier = build_dapm_classifier(
        latent_dim, num_classes_cfg,
        hidden_dim=int(cfg['classifier_hidden_dim'])
    )
    diffusion  = build_dapm_diffusion(
        latent_dim, num_classes_cfg,
        T=int(cfg['diffusion_T']),
        t_embed_dim=int(cfg.get('t_embed_dim', 32)),
        hidden_dim=int(cfg['diffusion_hidden_dim'])
    )

    # Warm-up forward pass to build all layer shapes before loading weights
    feat_dummy            = tf.zeros((1, feature_dim), dtype=tf.float32)
    z_mu_d, z_logvar_d, z_d = encoder(feat_dummy, training=False)
    y_guidance_d          = classifier(z_d, training=False)
    _                     = diffusion(
        [z_d, y_guidance_d, y_guidance_d, tf.ones((1, 1), dtype=tf.int32)],
        training=False
    )

    # Load saved weights
    encoder.load_weights(str(DAPM_DIR / f'{model_key}_dapm_full_encoder.weights.h5'))
    classifier.load_weights(str(DAPM_DIR / f'{model_key}_dapm_full_classifier.weights.h5'))
    diffusion.load_weights(str(DAPM_DIR / f'{model_key}_dapm_full_diffusion.weights.h5'))
    print(f'  Weights loaded for {model_key}')

    betas, alphas, alpha_bars = make_beta_schedule(
        int(cfg['diffusion_T']),
        beta_start=float(cfg['beta_start']),
        beta_end=float(cfg['beta_end'])
    )
    return {
        'cfg':               cfg,
        'feature_extractor': feature_extractor,
        'encoder':           encoder,
        'classifier':        classifier,
        'diffusion':         diffusion,
        'betas':             betas,
        'alphas':            alphas,
        'alpha_bars':        alpha_bars,
        'T':                 int(cfg['diffusion_T']),
        'num_classes':       num_classes_cfg,
    }
```

# 7.0 — Shared Utilities

Reusable helpers for probability normalisation, conformal metrics, plotting, and inference.
All conformal method builders in Section 8 depend on these.

## 7.1 — Probability & Conformal Metrics Helpers

```python
# Colour palette for class maps
CLASS_COLOR_BASE = [
    '#0000FF', '#00FF00', '#FF0000', '#00FFFF', '#FF00FF',
    '#FFFF00', '#A52A2A', '#FFA500', '#7FFF00', '#8A2BE2',
]
UNCERTAIN_COLOR = '#808080'


def get_class_colors(n):
    """Return n distinct hex colours, extending the base palette via tab20 if needed."""
    if n <= len(CLASS_COLOR_BASE):
        return CLASS_COLOR_BASE[:n]
    colors = CLASS_COLOR_BASE.copy()
    cmap   = plt.cm.get_cmap('tab20', n)
    for i in range(len(colors), n):
        c = cmap(i)
        colors.append('#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255)))
    return colors[:n]


def normalize_probs(prob, eps=1e-12):
    """Clip, sanitise, and row-normalise a probability matrix."""
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    rs   = prob.sum(axis=-1, keepdims=True)
    rs   = np.where(rs <= eps, 1.0, rs)
    return prob / rs


def predict_probs(model, x, batch_size=128):
    """Run model inference and return sanitised probability array."""
    prob = model.predict(x, batch_size=batch_size, verbose=0)
    return normalize_probs(prob, eps=EPS)


def safe_quantile(scores, q):
    """Compute a quantile with the 'higher' interpolation method, compatible across NumPy versions."""
    if len(scores) == 0:
        return 1.0
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(scores, q, method='higher'))
    except TypeError:
        return float(np.quantile(scores, q, interpolation='higher'))


def conformal_qhat(scores, alpha):
    """Compute the conformal quantile threshold q_hat for a given alpha."""
    n       = len(scores)
    if n == 0:
        return 1.0
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return safe_quantile(scores, q_level)


def compute_set_metrics(pred_sets, y_true):
    """Compute coverage, set-size statistics, singleton/empty rates from boolean prediction sets."""
    pred_sets = pred_sets.astype(bool)
    set_sizes = pred_sets.sum(axis=1)
    covered   = pred_sets[np.arange(len(y_true)), y_true].astype(int)
    return {
        'empirical_coverage': float(np.mean(covered)),
        'avg_set_size':       float(np.mean(set_sizes)),
        'median_set_size':    float(np.median(set_sizes)),
        'singleton_rate':     float(np.mean(set_sizes == 1)),
        'empty_set_rate':     float(np.mean(set_sizes == 0)),
        'set_sizes':          set_sizes,
        'covered':            covered,
    }


def per_class_coverage_df(pred_sets, y_true, n_classes):
    """Build a DataFrame of per-class empirical coverage and support counts."""
    rows = []
    for c in range(n_classes):
        mask    = (y_true == c)
        support = int(mask.sum())
        cov     = float(np.mean(pred_sets[mask, c])) if support > 0 else np.nan
        rows.append({'class_id': c, 'class_coverage': cov, 'support_count': support})
    return pd.DataFrame(rows)


def normalize_vector(x):
    """Min-max normalise a 1-D array to [0, 1]."""
    x     = np.asarray(x, dtype=np.float64)
    mn, mx = float(np.min(x)), float(np.max(x))
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)
```

## 7.2 — Plotting Helpers

```python
def fig_to_buffer(fig):
    """Save a matplotlib figure to an in-memory PNG buffer and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def add_bar_labels(ax, fmt='{:.2f}', y_pad=0.01):
    """Annotate each bar in a bar chart with its numeric value."""
    ymax = ax.get_ylim()[1]
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2, h + y_pad * ymax,
            fmt.format(h), ha='center', va='bottom', fontsize=10,
        )


def make_per_class_coverage_plot(per_cls_df, alpha, title):
    """Bar chart of per-class empirical coverage with a target coverage reference line."""
    fig, ax = plt.subplots(figsize=(15, 7))
    labels  = [f'Class {int(c)}' for c in per_cls_df['class_id'].tolist()]
    vals    = per_cls_df['class_coverage'].to_numpy(dtype=float)

    ax.bar(labels, vals, edgecolor='black', color='#4C72B0')
    ax.axhline(1 - alpha, color='red', linestyle='--', linewidth=2,
               label=f'Desired Coverage ({1-alpha:.2f})')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Class')
    ax.set_ylabel('Achieved Coverage')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    add_bar_labels(ax, fmt='{:.2f}', y_pad=0.01)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig_to_buffer(fig)


def make_certain_uncertain_map_plot(set_sizes_map, title):
    """Render a spatial map distinguishing certain (singleton set) from uncertain pixels."""
    # 0 = certain (yellow), 1 = uncertain (dark navy)
    disp = np.where(set_sizes_map == 1, 0, 1)
    cmap = ListedColormap(['#FFFF00', '#001F3F'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(disp, cmap=cmap)
    ax.set_title(title, fontsize=16)
    ax.axis('off')

    legend_handles = [
        Patch(facecolor='#FFFF00', edgecolor='black', label='Certain'),
        Patch(facecolor='#001F3F', edgecolor='black', label='Uncertain'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0.0, frameon=True)
    fig.tight_layout()
    return fig_to_buffer(fig)


def make_class_uncertain_mask_plot(combined_map, n_classes, title):
    """Colour-coded spatial map where uncertain pixels (multi-element prediction sets) are grey."""
    class_colors = get_class_colors(n_classes)
    cmap = ListedColormap(class_colors + [UNCERTAIN_COLOR])

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(combined_map, cmap=cmap, vmin=0, vmax=n_classes)
    ax.set_title(title, fontsize=16)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_classes + 1), fraction=0.046, pad=0.04)
    cbar.set_ticklabels([f'Class {i}' for i in range(n_classes)] + ['Uncertain'])
    fig.tight_layout()
    return fig_to_buffer(fig)


def make_pixel_counts_plot(pixel_counts_df, title, n_classes):
    """Bar chart of pixel counts per class plus an 'Uncertain' category."""
    class_colors = get_class_colors(n_classes)
    colors = class_colors + [UNCERTAIN_COLOR]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels  = pixel_counts_df['label'].tolist()
    counts  = pixel_counts_df['pixel_count'].tolist()
    ax.bar(labels, counts, color=colors[:len(labels)], edgecolor='black')
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Number of Pixels')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=45)

    ymax = max(counts) if len(counts) else 1
    for i, v in enumerate(counts):
        ax.text(i, v + 0.01 * ymax, f'{int(v):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig_to_buffer(fig)


def build_pixel_counts_df(combined_map, n_classes):
    """Tally per-class pixel counts from a combined class/uncertain map."""
    uniq, cnt = np.unique(combined_map, return_counts=True)
    counts    = {int(k): int(v) for k, v in zip(uniq, cnt)}

    rows = []
    for c in range(n_classes):
        rows.append({'class_id': c, 'label': f'Class {c}', 'pixel_count': counts.get(c, 0)})
    rows.append({'class_id': n_classes, 'label': 'Uncertain', 'pixel_count': counts.get(n_classes, 0)})
    return pd.DataFrame(rows)
```

## 7.3 — DAPM Inference Helpers

`predict_dapm_probs` mirrors `predict_with_dapm_classifier` from the DAPM training notebook
(encoder mean `z_mu` → latent classifier, no diffusion sampling) and layers on the same
`normalize_probs` sanitisation used for the baseline models, so its output is a drop-in
replacement for `predict_probs` everywhere a conformal method expects a probability matrix.
`predict_full_scene_dapm_probs` is the DAPM analogue of `predict_full_scene_probs`: same
column-by-column sliding-window loop, with the DAPM bundle used for inference instead of a
plain `.keras` model.
# DRY: consolidated repeated full-scene inference pattern, parameterised on the probability function

```python
def predict_dapm_probs(bundle, x_data, batch_size=128):
    """Run encoder(z_mu) -> classifier inference through a DAPM bundle; returns sanitised probabilities."""
    feature_extractor = bundle['feature_extractor']
    encoder            = bundle['encoder']
    classifier         = bundle['classifier']

    ds = tf.data.Dataset.from_tensor_slices(x_data.astype(np.float32)).batch(batch_size)
    all_probs = []
    for xb in ds:
        feat       = feature_extractor(xb, training=False)
        z_mu, _, _ = encoder(feat, training=False)
        probs      = classifier(z_mu, training=False)
        all_probs.append(probs.numpy())

    prob = np.concatenate(all_probs, axis=0)
    return normalize_probs(prob, eps=EPS)


def predict_full_scene_dapm_probs(bundle, x_img, H, W, B, patch_size, batch_size=128):
    """Run DAPM patch-based inference over every pixel and return an (H, W, C) probability cube."""
    pad   = patch_size // 2
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    # Infer number of classes from a single test patch
    test_patch = x_pad[0:patch_size, 0:patch_size, :][None, ...]
    test_prob  = predict_dapm_probs(bundle, test_patch, batch_size=1)
    n_classes  = test_prob.shape[1]

    full_prob = np.zeros((H, W, n_classes), dtype=np.float32)
    for col in range(W):
        patches = np.zeros((H, patch_size, patch_size, B), dtype=np.float32)
        for row in range(H):
            patches[row] = x_pad[row:row + patch_size, col:col + patch_size, :]

        prob_col              = predict_dapm_probs(bundle, patches, batch_size=batch_size)
        full_prob[:, col, :]  = prob_col

        if (col + 1) % 50 == 0 or (col + 1) == W:
            print(f'  full-scene progress: col {col + 1}/{W}')

    assert full_prob.shape == (H, W, n_classes), f'Unexpected full-scene shape: {full_prob.shape}'
    return full_prob
```

# 8.0 — Conformal Prediction Methods (applied to DAPM probabilities)

Each sub-section implements one conformal method: calibration, evaluation-set scoring,
full-scene map generation, and output packaging (summary dict, per-class DataFrame, plot
buffers, Excel tables). Split CP, Class-Conditional CP, RC3P, and RAPS operate purely on
probability matrices, so they are reused unchanged from the baseline conformal-comparison
notebook — only the *source* of `prob_cal` / `prob_eval` / `prob_full` changes (DAPM
encoder→classifier instead of a plain `.keras` model). Clustered CP is adapted because its
embeddings come from the DAPM latent space rather than a base-model penultimate layer.

## 8.1 — Split Conformal Prediction (SplitCP)

```python
def build_split_outputs_for_model(model_name, y_cal, prob_cal, y_eval, prob_eval, prob_full, alpha=0.05):
    """Run standard Split Conformal Prediction and return a standardised output dict."""
    t0 = time.perf_counter()

    # Calibration
    calib_scores = 1.0 - prob_cal[np.arange(len(y_cal)), y_cal]
    q_hat        = conformal_qhat(calib_scores, alpha)

    # Evaluation
    pred_sets_eval = prob_eval >= (1.0 - q_hat)
    metrics  = compute_set_metrics(pred_sets_eval, y_eval)
    per_cls  = per_class_coverage_df(pred_sets_eval, y_eval, prob_eval.shape[1])

    # Full-scene maps
    pred_sets_full  = prob_full >= (1.0 - q_hat)
    set_sizes_map   = np.sum(pred_sets_full, axis=2)
    pred_class_map  = np.argmax(prob_full, axis=2)
    combined_map    = np.where(set_sizes_map == 1, pred_class_map, prob_full.shape[2])
    pixel_counts_df = build_pixel_counts_df(combined_map, prob_full.shape[2])

    plot_buffers = {
        'Per-Class Coverage': make_per_class_coverage_plot(
            per_cls, alpha=alpha,
            title='Standard Split Conformal Prediction: Per-Class Coverage',
        ),
        'Certain vs Uncertain Map': make_certain_uncertain_map_plot(
            set_sizes_map,
            title='Predictions with 95% Uncertainty Map\n(Split Conformal Prediction — SCP)',
        ),
        'Class Map with Uncertain Mask': make_class_uncertain_mask_plot(
            combined_map, n_classes=prob_full.shape[2],
            title='Predictions with 95% Uncertainty Mask\n(Split Conformal Prediction — SCP)',
        ),
        'Pixel Counts': make_pixel_counts_plot(
            pixel_counts_df,
            title='Pixel Count per Class (Including Uncertain Regions)',
            n_classes=prob_full.shape[2],
        ),
    }

    runtime = time.perf_counter() - t0
    summary = {
        'model_name': model_name, 'method': 'SplitConformal',
        'target_coverage': float(1.0 - alpha),
        'empirical_coverage': metrics['empirical_coverage'],
        'avg_set_size': metrics['avg_set_size'],
        'median_set_size': metrics['median_set_size'],
        'singleton_rate': metrics['singleton_rate'],
        'empty_set_rate': metrics['empty_set_rate'],
        'runtime_sec': float(runtime),
        'alpha': float(alpha), 'lam': np.nan, 'n_clusters': np.nan,
        'mean_per_class_coverage': float(per_cls['class_coverage'].mean(skipna=True)),
    }

    tables = {
        'Summary':                 pd.DataFrame([summary]),
        'Per-Class Coverage Values': per_cls,
        'Pixel Counts':            pixel_counts_df,
        'Threshold':               pd.DataFrame([{'q_hat_split': float(q_hat)}]),
    }

    return {
        'model_name': model_name, 'method': 'SplitConformal',
        'summary': summary, 'per_class_df': per_cls,
        'plot_buffers': plot_buffers, 'tables': tables,
    }
```

## 8.2 — Class-Conditional Conformal Prediction (CcCP)

```python
def build_classconditional_outputs_for_model(model_name, y_cal, prob_cal, y_eval, prob_eval, prob_full, alpha=0.05):
    """Run Class-Conditional Conformal Prediction with per-class q_hat thresholds."""
    t0 = time.perf_counter()
    n_classes = prob_cal.shape[1]

    # Per-class calibration thresholds
    q_hats = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = (y_cal == c)
        if mask.sum() == 0:
            q_hats[c] = 1.0
            continue
        scores_c  = 1.0 - prob_cal[mask, c]
        q_hats[c] = conformal_qhat(scores_c, alpha)

    # Evaluation
    thresholds     = 1.0 - q_hats.reshape(1, -1)
    pred_sets_eval = prob_eval >= thresholds
    metrics  = compute_set_metrics(pred_sets_eval, y_eval)
    per_cls  = per_class_coverage_df(pred_sets_eval, y_eval, n_classes)

    # Full-scene maps
    pred_sets_full  = prob_full >= (1.0 - q_hats.reshape(1, 1, -1))
    set_sizes_map   = np.sum(pred_sets_full, axis=2)
    pred_class_map  = np.argmax(prob_full, axis=2)
    combined_map    = np.where(set_sizes_map == 1, pred_class_map, n_classes)
    pixel_counts_df = build_pixel_counts_df(combined_map, n_classes)

    plot_buffers = {
        'Per-Class Coverage': make_per_class_coverage_plot(
            per_cls, alpha=alpha,
            title='Class-Conditional Conformal Prediction: Per-Class Coverage',
        ),
        'Certain vs Uncertain Map': make_certain_uncertain_map_plot(
            set_sizes_map,
            title='Predictions with 95% Uncertainty Map\n(Class-Conditional Conformal Prediction — CcCP)',
        ),
        'Class Map with Uncertain Mask': make_class_uncertain_mask_plot(
            combined_map, n_classes=n_classes,
            title='Predictions with 95% Uncertainty Mask\n(Class-Conditional Conformal Prediction — CcCP)',
        ),
        'Pixel Counts': make_pixel_counts_plot(
            pixel_counts_df,
            title='Pixel Count per Class (Including Uncertain Regions)',
            n_classes=n_classes,
        ),
    }

    runtime = time.perf_counter() - t0
    summary = {
        'model_name': model_name, 'method': 'ClassConditionalConformal',
        'target_coverage': float(1.0 - alpha),
        'empirical_coverage': metrics['empirical_coverage'],
        'avg_set_size': metrics['avg_set_size'],
        'median_set_size': metrics['median_set_size'],
        'singleton_rate': metrics['singleton_rate'],
        'empty_set_rate': metrics['empty_set_rate'],
        'runtime_sec': float(runtime),
        'alpha': float(alpha), 'lam': np.nan, 'n_clusters': np.nan,
        'mean_per_class_coverage': float(per_cls['class_coverage'].mean(skipna=True)),
    }
    qhat_df = pd.DataFrame({'class_id': np.arange(n_classes), 'q_hat_classconditional': q_hats})

    tables = {
        'Summary':                 pd.DataFrame([summary]),
        'Per-Class Coverage Values': per_cls,
        'Pixel Counts':            pixel_counts_df,
        'Classwise q_hat':         qhat_df,
    }

    return {
        'model_name': model_name, 'method': 'ClassConditionalConformal',
        'summary': summary, 'per_class_df': per_cls,
        'plot_buffers': plot_buffers, 'tables': tables,
    }
```

## 8.3 — Rank Calibrated Class-Conditional CP (RC3P)

```python
def compute_topk_accuracy_matrix(prob, y, n_classes):
    """Compute a (K x C) matrix where entry [k, c] is top-k accuracy for class c."""
    acc_matrix = np.zeros((n_classes, n_classes))
    # Double argsort of negative probs gives 1-based rank (1 = highest prob)
    ranks = np.argsort(np.argsort(-prob, axis=1), axis=1) + 1
    for c in range(n_classes):
        mask        = (y == c)
        if mask.sum() == 0:
            continue
        class_ranks = ranks[mask, c]
        for k in range(1, n_classes + 1):
            acc_matrix[k-1, c] = np.mean(class_ranks <= k)
    return acc_matrix


def compute_rc3p_qhats_and_sets(prob_cal, y_cal, prob_eval, alpha, truncated_gap=0.1):
    """RC3P search: find optimal truncated rank limits and per-class q_hats minimising average set size."""
    n_classes = prob_cal.shape[1]
    num_cal   = len(y_cal)

    cal_ranks  = np.argsort(np.argsort(-prob_cal,  axis=1), axis=1) + 1
    eval_ranks = np.argsort(np.argsort(-prob_eval, axis=1), axis=1) + 1

    acc_matrix = compute_topk_accuracy_matrix(prob_cal, y_cal, n_classes)
    err_matrix = 1.0 - acc_matrix

    num_samples_per_class = num_cal / n_classes
    tc_alpha = alpha - (truncated_gap / np.sqrt(num_samples_per_class))

    # Minimum rank k such that top-k error falls below tc_alpha
    suit_k = []
    for c in range(n_classes):
        valid_k = np.where(err_matrix[:, c] < tc_alpha)[0]
        suit_k.append(valid_k[0] + 1 if len(valid_k) > 0 else n_classes)

    k_max    = max(suit_k)
    rank_all = n_classes
    mix_paras = np.linspace(0, 1, int(rank_all - k_max) + 1) if rank_all >= k_max else [1.0]

    smallest_ps          = float('inf')
    best_classwise_qhats = np.ones(n_classes)
    best_suit_indices    = suit_k

    for mix_para in mix_paras:
        test_indices = [int(np.ceil((1 - mix_para) * suit_k[i] + n_classes * mix_para))
                        for i in range(n_classes)]
        test_err     = [err_matrix[test_indices[i]-1, i] for i in range(n_classes)]
        test_alphas  = [tc_alpha - err for err in test_err]

        q_hats = np.zeros(n_classes)
        for c in range(n_classes):
            idx    = (y_cal == c) & (cal_ranks[:, c] <= test_indices[c])
            scores = 1.0 - prob_cal[idx, c]
            q_hats[c] = 1.0 if len(scores) == 0 else conformal_qhat(scores, test_alphas[c])

        thresholds   = 1.0 - q_hats
        meets_thresh = prob_eval >= thresholds
        meets_rank   = eval_ranks <= np.array(test_indices)
        pred_sets    = meets_thresh & meets_rank

        avg_size = np.mean(pred_sets.sum(axis=1))
        if avg_size < smallest_ps:
            smallest_ps          = avg_size
            best_classwise_qhats = q_hats
            best_suit_indices    = test_indices

    return best_classwise_qhats, best_suit_indices


def build_rc3p_outputs_for_model(model_name, y_cal, prob_cal, y_eval, prob_eval, prob_full, alpha=0.05, truncated_gap=0.1):
    """Run RC3P and return a standardised output dict including full-scene maps."""
    t0        = time.perf_counter()
    n_classes = prob_cal.shape[1]

    q_hats, suit_indices = compute_rc3p_qhats_and_sets(
        prob_cal, y_cal, prob_eval, alpha, truncated_gap,
    )

    # Evaluation
    eval_ranks       = np.argsort(np.argsort(-prob_eval, axis=1), axis=1) + 1
    meets_thresh_eval = prob_eval >= (1.0 - q_hats.reshape(1, -1))
    meets_rank_eval   = eval_ranks <= np.array(suit_indices).reshape(1, -1)
    pred_sets_eval    = meets_thresh_eval & meets_rank_eval

    metrics = compute_set_metrics(pred_sets_eval, y_eval)
    per_cls = per_class_coverage_df(pred_sets_eval, y_eval, n_classes)

    # Full-scene maps
    full_ranks        = np.argsort(np.argsort(-prob_full, axis=2), axis=2) + 1
    meets_thresh_full = prob_full >= (1.0 - q_hats.reshape(1, 1, -1))
    meets_rank_full   = full_ranks <= np.array(suit_indices).reshape(1, 1, -1)
    pred_sets_full    = meets_thresh_full & meets_rank_full

    set_sizes_map   = np.sum(pred_sets_full, axis=2)
    pred_class_map  = np.argmax(prob_full, axis=2)
    combined_map    = np.where(set_sizes_map == 1, pred_class_map, n_classes)
    pixel_counts_df = build_pixel_counts_df(combined_map, n_classes)

    plot_buffers = {
        'Per-Class Coverage': make_per_class_coverage_plot(
            per_cls, alpha=alpha, title='RC3P: Per-Class Coverage',
        ),
        'Certain vs Uncertain Map': make_certain_uncertain_map_plot(
            set_sizes_map, title='Predictions with 95% Uncertainty Map\n(RC3P)',
        ),
        'Class Map with Uncertain Mask': make_class_uncertain_mask_plot(
            combined_map, n_classes=n_classes,
            title='Predictions with 95% Uncertainty Mask\n(RC3P)',
        ),
        'Pixel Counts': make_pixel_counts_plot(
            pixel_counts_df, title='Pixel Count per Class (RC3P)', n_classes=n_classes,
        ),
    }

    runtime = time.perf_counter() - t0
    summary = {
        'model_name': model_name, 'method': 'RC3P',
        'target_coverage': float(1.0 - alpha),
        'empirical_coverage': metrics['empirical_coverage'],
        'avg_set_size': metrics['avg_set_size'],
        'median_set_size': metrics['median_set_size'],
        'singleton_rate': metrics['singleton_rate'],
        'empty_set_rate': metrics['empty_set_rate'],
        'runtime_sec': float(runtime),
        'alpha': float(alpha), 'lam': np.nan, 'n_clusters': np.nan,
        'mean_per_class_coverage': float(per_cls['class_coverage'].mean(skipna=True)),
    }
    qhat_df = pd.DataFrame({
        'class_id': np.arange(n_classes),
        'q_hat_rc3p': q_hats,
        'suit_index_limit': suit_indices,
    })

    tables = {
        'Summary':                  pd.DataFrame([summary]),
        'Per-Class Coverage Values': per_cls,
        'Pixel Counts':             pixel_counts_df,
        'RC3P Thresholds & Limits': qhat_df,
    }

    return {
        'model_name': model_name, 'method': 'RC3P',
        'summary': summary, 'per_class_df': per_cls,
        'plot_buffers': plot_buffers, 'tables': tables,
    }
```

## 8.4 — Clustered Conformal Prediction (ClCP) — DAPM latent-space embeddings

Adapted from `build_clustered_outputs_for_model`: instead of slicing the penultimate layer of
a plain `.keras` model, embeddings are the DAPM encoder's mean latent vector `z_mu` — the same
representation the DAPM classifier itself operates on, obtained via the bundle's
`feature_extractor` -> `encoder` path. Every other step (class-mean clustering, per-cluster
calibration, coverage metrics, plots) is unchanged.
# DRY: same clustering / calibration logic as the baseline version, parameterised on the embedding source

```python
def get_dapm_embeddings(bundle, x_data, batch_size=128):
    """Return DAPM latent-space embeddings (encoder mean z_mu) for a patch array."""
    feature_extractor = bundle['feature_extractor']
    encoder            = bundle['encoder']

    ds = tf.data.Dataset.from_tensor_slices(x_data.astype(np.float32)).batch(batch_size)
    all_emb = []
    for xb in ds:
        feat       = feature_extractor(xb, training=False)
        z_mu, _, _ = encoder(feat, training=False)
        all_emb.append(z_mu.numpy())
    emb = np.concatenate(all_emb, axis=0)
    return np.nan_to_num(emb)


def build_clustered_outputs_for_model_dapm(
    model_name, bundle, x_cal, y_cal, x_eval, y_eval,
    prob_cal, prob_eval, alpha=0.05, n_clusters=4, batch_size=128,
):
    """Run Clustered Conformal Prediction on DAPM latent embeddings, cluster classes, calibrate per cluster."""
    t0 = time.perf_counter()
    n_classes = prob_cal.shape[1]
    k = int(min(max(1, n_clusters), n_classes))

    # Build embeddings from the DAPM latent space (z_mu)
    emb_cal = get_dapm_embeddings(bundle, x_cal, batch_size=batch_size)

    # Cluster classes by mean embedding
    global_mean = emb_cal.mean(axis=0)
    class_means = []
    for c in range(n_classes):
        m = (y_cal == c)
        class_means.append(emb_cal[m].mean(axis=0) if m.sum() > 0 else global_mean)
    class_means = np.vstack(class_means)

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    cluster_assignments = km.fit_predict(class_means)

    # Per-cluster calibration
    q_hats_per_cluster = np.ones(k, dtype=np.float64)
    for cluster_id in range(k):
        cls  = np.where(cluster_assignments == cluster_id)[0]
        mask = np.isin(y_cal, cls)
        if mask.sum() == 0:
            continue
        scores = 1.0 - prob_cal[mask, y_cal[mask]]
        q_hats_per_cluster[cluster_id] = conformal_qhat(scores, alpha)

    q_per_class = q_hats_per_cluster[cluster_assignments]
    pred_sets   = prob_eval >= (1.0 - q_per_class.reshape(1, -1))

    metrics = compute_set_metrics(pred_sets, y_eval)
    per_cls = per_class_coverage_df(pred_sets, y_eval, n_classes)

    # Uncertainty analysis
    set_sizes       = pred_sets.sum(axis=1)
    norm_unc        = normalize_vector(set_sizes)
    pred_class      = np.argmax(prob_eval, axis=1)
    sample_clusters = cluster_assignments[pred_class]

    cluster_mean_unc_df = (
        pd.DataFrame({'cluster': sample_clusters, 'uncertainty': norm_unc})
        .groupby('cluster', as_index=False)['uncertainty'].mean()
        .sort_values('cluster').reset_index(drop=True)
    )

    class_rows = []
    for c in range(n_classes):
        m = (pred_class == c)
        if m.sum() == 0:
            continue
        class_rows.append({'class_id': c, 'uncertainty': float(np.mean(norm_unc[m]))})
    class_mean_unc_df = pd.DataFrame(class_rows)

    # Plots
    plot_buffers = {}
    plot_buffers['Per-Class Coverage'] = make_per_class_coverage_plot(
        per_cls, alpha=alpha,
        title='Clustered Conformal Prediction (DAPM latent space) — Per-Class Coverage',
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(norm_unc, bins=30, color='steelblue', edgecolor='black')
    ax.set_xlabel('Normalized Predictive Set Size (Uncertainty)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Uncertainty Distribution — Clustered Conformal Prediction (DAPM ClCP)')
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    plot_buffers['Uncertainty Distribution'] = fig_to_buffer(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cluster_mean_unc_df['cluster'].astype(str), cluster_mean_unc_df['uncertainty'],
           color='darkorange', edgecolor='black')
    ax.set_title('Average Uncertainty per Cluster (DAPM ClCP)')
    ax.set_ylabel('Mean Normalized Uncertainty')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    add_bar_labels(ax, fmt='{:.2f}', y_pad=0.01)
    fig.tight_layout()
    plot_buffers['Avg Uncertainty per Cluster'] = fig_to_buffer(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    if len(class_mean_unc_df) > 0:
        ax.bar(class_mean_unc_df['class_id'].astype(str), class_mean_unc_df['uncertainty'],
               color='#1f77b4', edgecolor='black')
        ax.set_xticklabels([f'Class {x}' for x in class_mean_unc_df['class_id']], rotation=45, ha='right')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Normalized Uncertainty')
    ax.set_title('Average Uncertainty per Class — Clustered Conformal Prediction (DAPM ClCP)')
    ax.grid(axis='y', alpha=0.4)
    add_bar_labels(ax, fmt='{:.2f}', y_pad=0.01)
    fig.tight_layout()
    plot_buffers['Avg Uncertainty per Class'] = fig_to_buffer(fig)

    runtime = time.perf_counter() - t0
    summary = {
        'model_name': model_name, 'method': 'ClusteredConformal',
        'target_coverage': float(1.0 - alpha),
        'empirical_coverage': metrics['empirical_coverage'],
        'avg_set_size': metrics['avg_set_size'],
        'median_set_size': metrics['median_set_size'],
        'singleton_rate': metrics['singleton_rate'],
        'empty_set_rate': metrics['empty_set_rate'],
        'runtime_sec': float(runtime),
        'alpha': float(alpha), 'lam': np.nan, 'n_clusters': int(k),
        'mean_per_class_coverage': float(per_cls['class_coverage'].mean(skipna=True)),
    }

    qhat_cluster_df  = pd.DataFrame({'cluster_id': np.arange(k), 'q_hat_cluster': q_hats_per_cluster})
    class_cluster_df = pd.DataFrame({'class_id': np.arange(n_classes), 'cluster_id': cluster_assignments})

    tables = {
        'Summary':                    pd.DataFrame([summary]),
        'Per-Class Coverage Values':  per_cls,
        'Class-to-Cluster Assignment': class_cluster_df,
        'Cluster q_hat':              qhat_cluster_df,
        'Cluster Mean Uncertainty':   cluster_mean_unc_df,
        'Class Mean Uncertainty':     class_mean_unc_df,
    }

    return {
        'model_name': model_name, 'method': 'ClusteredConformal',
        'summary': summary, 'per_class_df': per_cls,
        'plot_buffers': plot_buffers, 'tables': tables,
    }
```

## 8.5 — RAPS (Regularised Adaptive Prediction Sets)

```python
def raps_score_single(prob_row, true_label, lam=0.01, k_reg=1):
    """Compute the RAPS non-conformity score for one sample."""
    order      = np.argsort(prob_row)[::-1]
    rank       = int(np.where(order == true_label)[0][0])
    cumulative = float(np.sum(prob_row[order[:rank]]))
    penalty    = float(lam) * max(rank - int(k_reg), 0)
    return cumulative + penalty


def raps_set_single(prob_row, q_hat, lam=0.01, k_reg=1):
    """Build the RAPS prediction set for one sample given a calibrated q_hat threshold."""
    order    = np.argsort(prob_row)[::-1]
    pred_set = np.zeros_like(prob_row, dtype=bool)

    cumulative = 0.0
    for k, cls in enumerate(order):
        reg_penalty = float(lam) * max(k - int(k_reg), 0)
        if cumulative + reg_penalty <= q_hat:
            pred_set[cls] = True
            cumulative   += float(prob_row[cls])
        else:
            break

    # Guarantee at least the top-1 class is included
    if not pred_set.any():
        pred_set[order[0]] = True

    return pred_set


def build_raps_outputs_for_model(model_name, y_cal, prob_cal, y_eval, prob_eval, alpha=0.05, lam=0.01, k_reg=1):
    """Run RAPS and return a standardised output dict (no full-scene map for RAPS)."""
    t0    = time.perf_counter()
    n_cal = len(y_cal)

    raps_scores = np.array([
        raps_score_single(prob_cal[i], int(y_cal[i]), lam=lam, k_reg=k_reg)
        for i in range(n_cal)
    ], dtype=np.float64)

    q_hat = conformal_qhat(raps_scores, alpha)

    pred_sets = np.array([
        raps_set_single(prob_eval[i], q_hat=q_hat, lam=lam, k_reg=k_reg)
        for i in range(len(y_eval))
    ], dtype=bool)

    metrics = compute_set_metrics(pred_sets, y_eval)
    per_cls = per_class_coverage_df(pred_sets, y_eval, prob_eval.shape[1])

    plot_buffers = {
        'Per-Class Coverage': make_per_class_coverage_plot(
            per_cls, alpha=alpha, title='RAPS: Per-Class Coverage',
        ),
    }

    runtime = time.perf_counter() - t0
    summary = {
        'model_name': model_name, 'method': 'RAPS',
        'target_coverage': float(1.0 - alpha),
        'empirical_coverage': metrics['empirical_coverage'],
        'avg_set_size': metrics['avg_set_size'],
        'median_set_size': metrics['median_set_size'],
        'singleton_rate': metrics['singleton_rate'],
        'empty_set_rate': metrics['empty_set_rate'],
        'runtime_sec': float(runtime),
        'alpha': float(alpha), 'lam': float(lam), 'n_clusters': np.nan,
        'mean_per_class_coverage': float(per_cls['class_coverage'].mean(skipna=True)),
    }

    tables = {
        'Summary':                 pd.DataFrame([summary]),
        'Per-Class Coverage Values': per_cls,
        'RAPS Parameters':         pd.DataFrame([{'q_hat_raps': float(q_hat), 'lambda': float(lam), 'k_reg': int(k_reg)}]),
    }

    return {
        'model_name': model_name, 'method': 'RAPS',
        'summary': summary, 'per_class_df': per_cls,
        'plot_buffers': plot_buffers, 'tables': tables,
    }
```

# 9.0 — Main Execution Loop

Iterates over all three backbones. For each: loads its DAPM bundle, computes calibration /
evaluation / full-scene probabilities via the encoder→classifier path (Section 7.3), then
runs each of the five conformal prediction methods against those DAPM probabilities and
collects the results into `all_outputs`.

```python
def method_sheet_prefix(method_name):
    """Return the abbreviated sheet-name prefix for a given method identifier."""
    mapping = {
        'SplitConformal':          'Split',
        'ClassConditionalConformal': 'ClassCond',
        'RC3P':                    'RC3P',
        'ClusteredConformal':      'Clustered',
        'RAPS':                    'RAPS',
    }
    return mapping.get(method_name, method_name)


all_outputs           = []
full_scene_prob_cache = {}
dapm_bundles           = {}

for model_key in MODEL_KEYS:
    model_name = MODEL_NAME_MAP.get(model_key, model_key)
    print(f'\n==================== {model_name} ====================')

    bundle = load_dapm_bundle(model_key)
    dapm_bundles[model_key] = bundle

    prob_cal  = predict_dapm_probs(bundle, x_cal,  batch_size=BATCH_SIZE)
    prob_eval = predict_dapm_probs(bundle, x_eval, batch_size=BATCH_SIZE)

    # Cache full-scene probabilities (expensive) so they are computed only once per model
    if model_key not in full_scene_prob_cache:
        print(f'Computing full-scene DAPM probabilities for {model_name} ...')
        full_scene_prob_cache[model_key] = predict_full_scene_dapm_probs(
            bundle=bundle, x_img=x_img, H=H, W=W, B=B,
            patch_size=PATCH_SIZE, batch_size=BATCH_SIZE,
        )
    prob_full = full_scene_prob_cache[model_key]

    # 1. Split CP
    split_out = build_split_outputs_for_model(
        model_name=model_name, y_cal=y_cal, prob_cal=prob_cal,
        y_eval=y_eval, prob_eval=prob_eval, prob_full=prob_full, alpha=ALPHA,
    )
    all_outputs.append(split_out)

    # 2. Class-Conditional CP
    cw_out = build_classconditional_outputs_for_model(
        model_name=model_name, y_cal=y_cal, prob_cal=prob_cal,
        y_eval=y_eval, prob_eval=prob_eval, prob_full=prob_full, alpha=ALPHA,
    )
    all_outputs.append(cw_out)

    # 3. RC3P
    rc3p_out = build_rc3p_outputs_for_model(
        model_name=model_name, y_cal=y_cal, prob_cal=prob_cal,
        y_eval=y_eval, prob_eval=prob_eval, prob_full=prob_full,
        alpha=ALPHA, truncated_gap=0.1,
    )
    all_outputs.append(rc3p_out)

    # 4. Clustered CP (DAPM latent-space embeddings)
    clcp_out = build_clustered_outputs_for_model_dapm(
        model_name=model_name, bundle=bundle, x_cal=x_cal, y_cal=y_cal,
        x_eval=x_eval, y_eval=y_eval, prob_cal=prob_cal, prob_eval=prob_eval,
        alpha=ALPHA, n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE,
    )
    all_outputs.append(clcp_out)

    # 5. RAPS
    raps_out = build_raps_outputs_for_model(
        model_name=model_name, y_cal=y_cal, prob_cal=prob_cal,
        y_eval=y_eval, prob_eval=prob_eval, alpha=ALPHA, lam=RAPS_LAM, k_reg=RAPS_K_REG,
    )
    all_outputs.append(raps_out)

    for out in [split_out, cw_out, rc3p_out, clcp_out, raps_out]:
        s = out['summary']
        print(f"{s['method']}: coverage={s['empirical_coverage']:.4f}, "
              f"avg_set={s['avg_set_size']:.3f}, runtime={s['runtime_sec']:.2f}s")


# Aggregate summary tables
summary_compact_df = (
    pd.DataFrame([o['summary'] for o in all_outputs])
    .sort_values(['method', 'model_name'])
    .reset_index(drop=True)
)

per_class_all_df = pd.concat([
    o['per_class_df'].assign(model_name=o['model_name'], method=o['method'])
    for o in all_outputs
], ignore_index=True)
```

# 10.0 — Comparative Analysis Plots

Generates grouped bar charts comparing empirical coverage, average set size, runtime, and
mean per-class coverage across all three DAPM-backed models — separately for Clustered CP
and RAPS.

```python
def build_method_comparison_buffers(summary_df, method_name, title_prefix):
    """Build grouped metric and mean per-class coverage plots for a given method across all models."""
    df = summary_df[summary_df['method'] == method_name].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(data=df, x='model_name', y='empirical_coverage', ax=axes[0], palette='Set2')
    axes[0].axhline(1 - ALPHA, linestyle='--', color='red', linewidth=1.8,
                    label=f'Target ({1-ALPHA:.2f})')
    axes[0].set_title(f'{title_prefix}: Empirical Coverage')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Coverage')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True)

    sns.barplot(data=df, x='model_name', y='avg_set_size',  ax=axes[1], palette='Set3')
    axes[1].set_title(f'{title_prefix}: Avg Set Size')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Average Set Size')

    sns.barplot(data=df, x='model_name', y='runtime_sec',   ax=axes[2], palette='Set1')
    axes[2].set_title(f'{title_prefix}: Runtime')
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('Runtime (sec)')

    for ax in axes:
        add_bar_labels(ax, fmt='{:.2f}', y_pad=0.01)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    metric_buf = fig_to_buffer(fig)

    mean_pc = (
        per_class_all_df[per_class_all_df['method'] == method_name]
        .groupby('model_name', as_index=False)['class_coverage']
        .mean()
        .rename(columns={'class_coverage': 'mean_per_class_coverage'})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=mean_pc, x='model_name', y='mean_per_class_coverage', palette='Dark2', ax=ax)
    ax.set_title(f'{title_prefix}: Mean Per-Class Coverage')
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Coverage')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    add_bar_labels(ax, fmt='{:.2f}', y_pad=0.01)
    fig.tight_layout()
    mean_pc_buf = fig_to_buffer(fig)

    return {
        'metrics_plot':       metric_buf,
        'mean_per_class_plot': mean_pc_buf,
        'method_df':          df,
        'mean_per_class_df':  mean_pc,
    }


cluster_compare = build_method_comparison_buffers(summary_compact_df, 'ClusteredConformal', 'Clustered CP')
raps_compare    = build_method_comparison_buffers(summary_compact_df, 'RAPS', 'RAPS')
```

# 11.0 — Excel Export

Writes all results to a single `.xlsx` workbook. Each method+model combination gets its own
sheet; global summary and comparison sheets are added on top.

## 11.1 — Excel I/O Helpers

```python
def sanitize_sheet_name(name):
    """Strip characters that are illegal in Excel sheet names and truncate to 31 chars."""
    for ch in ['\\', '/', '*', '?', ':', '[', ']']:
        name = name.replace(ch, '_')
    return name[:31]


def make_sheet_name(base, used):
    """Return a unique sheet name derived from base, appending a counter if necessary."""
    base      = sanitize_sheet_name(base)
    candidate = base
    i = 1
    while candidate in used:
        suffix    = f'_{i}'
        candidate = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(candidate)
    return candidate


def insert_buffer_image(ws, row, col, img_buf, x_scale=0.8, y_scale=0.8):
    """Insert a PNG buffer image at (row, col) in an xlsxwriter worksheet."""
    img_buf.seek(0)
    ws.insert_image(row, col, 'plot.png', {'image_data': img_buf, 'x_scale': x_scale, 'y_scale': y_scale})


def write_method_model_sheet(writer, workbook, output, sheet_name):
    """Write tables on the left and plot images on the right for a single method+model sheet."""
    ws = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = ws

    row = 0
    ws.write(row, 0, f"{output['method']} - {output['model_name']}")
    row += 2

    for tname, tdf in output['tables'].items():
        ws.write(row, 0, tname)
        tdf.to_excel(writer, sheet_name=sheet_name, startrow=row + 1, startcol=0, index=False)
        row += len(tdf) + 4

    # Stacked images to the right so they do not overlap the tables
    img_row, img_col = 0, 9
    for pname, pbuf in output['plot_buffers'].items():
        ws.write(img_row, img_col, pname)
        insert_buffer_image(ws, img_row + 1, img_col, pbuf, x_scale=0.75, y_scale=0.75)
        img_row += 24


def write_comparison_sheet(writer, workbook, sheet_name, compare_obj, title):
    """Write a comparison sheet with summary tables and grouped metric/per-class coverage plots."""
    ws = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = ws

    ws.write(0, 0, title)
    compare_obj['method_df'].to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=0, index=False)
    compare_obj['mean_per_class_df'].to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=8, index=False)

    ws.write(12, 0, 'Grouped Metrics Comparison')
    insert_buffer_image(ws, 13, 0, compare_obj['metrics_plot'], x_scale=0.78, y_scale=0.78)

    ws.write(40, 0, 'Mean Per-Class Coverage Comparison')
    insert_buffer_image(ws, 41, 0, compare_obj['mean_per_class_plot'], x_scale=0.85, y_scale=0.85)
```

## 11.2 — Save CSVs, Run Config, and Workbook

```python
run_config = {
    'seed': SEED, 'data_file': str(DATA_FILE), 'label_file': str(LABEL_FILE),
    'model_dir': str(MODEL_DIR), 'dapm_dir': str(DAPM_DIR),
    'output_dir': str(OUTPUT_DIR), 'results_dir': str(RESULTS_DIR),
    'h': H, 'w': W, 'b': B, 'patch_size': PATCH_SIZE,
    'train_percent': TRAIN_PERCENT, 'calib_fraction_of_test': CALIB_FRACTION_OF_TEST,
    'alpha': ALPHA, 'raps_lam': RAPS_LAM, 'raps_k_reg': RAPS_K_REG,
    'n_clusters': N_CLUSTERS, 'batch_size': BATCH_SIZE,
    'probability_source': 'DAPM encoder(z_mu) -> classifier (no diffusion sampling)',
    'timestamp_utc': pd.Timestamp.utcnow().isoformat(),
}

# CSV / JSON side-outputs
summary_compact_df.to_csv(SUMMARY_CSV_PATH, index=False)
per_class_all_df.to_csv(PER_CLASS_CSV_PATH, index=False)
with open(RUN_CONFIG_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(run_config, f, indent=2)

# Excel workbook
used = set()
with pd.ExcelWriter(EXCEL_PATH, engine='xlsxwriter') as writer:
    workbook = writer.book

    s_summary = make_sheet_name('Summary_Compact', used)
    s_config  = make_sheet_name('Run_Config', used)
    summary_compact_df.to_excel(writer, sheet_name=s_summary, index=False)
    pd.DataFrame(list(run_config.items()), columns=['key', 'value']).to_excel(
        writer, sheet_name=s_config, index=False,
    )

    for out in all_outputs:
        method_prefix = method_sheet_prefix(out['method'])
        sheet = make_sheet_name(f"{method_prefix}_{out['model_name']}", used)
        write_method_model_sheet(writer, workbook, out, sheet)

    s_cmp_cluster = make_sheet_name('Compare_Clustered', used)
    s_cmp_raps    = make_sheet_name('Compare_RAPS',      used)
    write_comparison_sheet(writer, workbook, s_cmp_cluster, cluster_compare, 'Clustered CP (DAPM): 3-Model Comparison')
    write_comparison_sheet(writer, workbook, s_cmp_raps,    raps_compare,    'RAPS (DAPM): 3-Model Comparison')

print('Saved workbook:     ', EXCEL_PATH)
print('Saved summary CSV:  ', SUMMARY_CSV_PATH)
print('Saved per-class CSV:', PER_CLASS_CSV_PATH)
print('Saved run config:   ', RUN_CONFIG_JSON_PATH)
```

# 12.0 — Final Validation

Verifies the workbook was written correctly: checks all required sheet names are present,
confirms the summary has the expected number of rows (3 models × 5 methods = 15), and
validates pixel accounting for map-based methods.

```python
assert EXCEL_PATH.exists(), f'Workbook not found: {EXCEL_PATH}'

wb              = load_workbook(EXCEL_PATH, read_only=True)
existing_sheets = set(wb.sheetnames)

required_prefixes = {
    'Summary_Compact', 'Run_Config',
    'Split_AlexNet',    'Split_GFNet',    'Split_ViT',
    'ClassCond_AlexNet','ClassCond_GFNet','ClassCond_ViT',
    'RC3P_AlexNet',     'RC3P_GFNet',     'RC3P_ViT',
    'Clustered_AlexNet','Clustered_GFNet','Clustered_ViT',
    'RAPS_AlexNet',     'RAPS_GFNet',     'RAPS_ViT',
    'Compare_Clustered','Compare_RAPS',
}

for req in required_prefixes:
    assert any(s.startswith(req) for s in existing_sheets), f'Missing sheet: {req}'

# 3 models × 5 methods
expected_rows = 3 * 5
assert len(summary_compact_df) == expected_rows, (
    f'Expected {expected_rows} rows, got {len(summary_compact_df)}'
)

assert ((summary_compact_df['empirical_coverage'] >= 0) &
        (summary_compact_df['empirical_coverage'] <= 1)).all()
assert (summary_compact_df['avg_set_size'] >= 0).all()

# Full-scene pixel accounting for map-based methods
map_methods = {'SplitConformal', 'ClassConditionalConformal', 'RC3P'}
for out in all_outputs:
    if out['method'] in map_methods:
        px = out['tables']['Pixel Counts']['pixel_count'].sum()
        assert int(px) == H * W, (
            f"Pixel count mismatch for {out['method']} {out['model_name']}: {px} vs {H*W}"
        )

print('Validation passed.')
print('Sheets:', len(existing_sheets))
print('Summary rows:', len(summary_compact_df))
summary_compact_df
```

