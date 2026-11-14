#!/usr/bin/env python3
"""
In-place config and symbol-flow fixer across all 4 benchmark suites.

Restores missing config constants, fixes keras unsafe deserialization import
ordering, and restores split/padded tensors in-place without inserting or
deleting any notebook cells (preserving Tier-9 cell-index assertions in
scripts/verify_suites.py).
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

SUITES = {
    "indian_pines_uncertainty_quantification": {
        "dataset_name": "Indian Pines",
        "hf_model_dir": "indian-pines-dataset-models",
    },
    "pavia_university_uncertainty_quantification": {
        "dataset_name": "Pavia University",
        "hf_model_dir": "pavia-university-dataset-models",
    },
    "hisar_3mts_uncertainty_quantification": {
        "dataset_name": "Hisar-3",
        "hf_model_dir": "hisar3-dataset-models",
    },
    "hissar_25_uncertainty_quantification": {
        "dataset_name": "Hissar-25",
        "hf_model_dir": "hissar25-dataset-models",
    },
}


def get_src(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


def set_src(nb: dict, idx: int, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    nb["cells"][idx]["source"] = lines


def append_block_if_missing(nb: dict, idx: int, marker: str, block: str) -> bool:
    src = get_src(nb, idx)
    if marker in src:
        return False
    new_src = src.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n"
    set_src(nb, idx, new_src)
    return True


def move_enable_unsafe_after_keras_import(nb: dict, idx: int) -> bool:
    src = get_src(nb, idx)
    unsafe_pattern = re.compile(
        r"^try:\s*\n\s*keras\.config\.enable_unsafe_deserialization\(\)\s*\nexcept Exception:\s*\n\s*pass\s*\n*",
        re.MULTILINE,
    )
    if not unsafe_pattern.search(src):
        return False
    # Only move if it appears BEFORE `from tensorflow import keras`
    m = unsafe_pattern.search(src)
    keras_idx = src.find("from tensorflow import keras")
    if keras_idx == -1 or m.start() > keras_idx:
        return False
    cleaned = unsafe_pattern.sub("", src, count=1).lstrip("\n")
    target = "from tensorflow.keras.models import load_model, Model"
    unsafe_block = (
        "try:\n"
        "    keras.config.enable_unsafe_deserialization()\n"
        "except Exception:\n"
        "    pass\n"
    )
    if target in cleaned:
        cleaned = cleaned.replace(target, target + "\n\n" + unsafe_block.rstrip(), 1)
    else:
        cleaned = cleaned.replace(
            "from tensorflow import keras",
            "from tensorflow import keras\n" + unsafe_block.rstrip(),
            1,
        )
    set_src(nb, idx, cleaned)
    return True


def fix_credal_ensemble(suite_name: str, nb: dict) -> bool:
    if suite_name not in (
        "indian_pines_uncertainty_quantification",
        "pavia_university_uncertainty_quantification",
    ):
        return False
    changed = False
    src = get_src(nb, 7)
    hf_dir = SUITES[suite_name]["hf_model_dir"]
    old_line = (
        f'ens_candidates = list(Path(\'/kaggle/input\').glob(f"**/{hf_dir}/**/ensembles"))'
    )
    new_lines = (
        f'ens_candidates = (\n'
        f'        list(Path(\'/kaggle/working/models\').glob("**/ensembles"))\n'
        f'        or list(Path(\'/kaggle/input\').glob(f"**/{hf_dir}/**/ensembles"))\n'
        f'        or list(Path(\'/kaggle/input\').glob("**/ensembles"))\n'
        f'    )'
    )
    if old_line in src:
        src = src.replace(old_line, new_lines, 1)
        set_src(nb, 7, src)
        changed = True

    old_model_dir = (
        f'MODEL_DIR = ens_candidates[0] if ens_candidates else Path(f"/kaggle/input/{hf_dir}/results/models/ensembles")'
    )
    new_model_dir = (
        'MODEL_DIR = ens_candidates[0] if ens_candidates else ('
        'Path("/kaggle/working/models/ensembles") if Path("/kaggle/working/models/ensembles").exists() '
        'else Path("/kaggle/working/models"))'
    )
    if old_model_dir in get_src(nb, 7):
        set_src(nb, 7, get_src(nb, 7).replace(old_model_dir, new_model_dir, 1))
        changed = True

    block = (
        "AU_THRESH = 0.5\n"
        "EU_THRESH = 0.2\n"
        "TU_THRESH = 0.7"
    )
    if append_block_if_missing(nb, 7, "# ── CreDE Uncertainty Thresholds (auto-restored) ──", block):
        changed = True
    return changed


def fix_baseline(suite_name: str, nb: dict) -> bool:
    src = get_src(nb, 4)
    lines_to_add = []
    if "MODEL_NAME_MAP" not in src:
        lines_to_add.append(
            "MODEL_NAME_MAP = {'AlexNet_CNN': 'AlexNet', 'GFNet': 'GFNet', 'ViT_UNet': 'ViT'}"
        )
    if "TRUSTED_MODEL_ROOTS" not in src:
        lines_to_add.append(
            "TRUSTED_MODEL_ROOTS = [\n"
            "    Path('/kaggle/input'),\n"
            "    Path('/kaggle/working'),\n"
            "    MODEL_DIR,\n"
            "    Path('baseline/results/models'),\n"
            "    Path('models'),\n"
            "]"
        )
    if re.search(r"^\s*EPS\s*=", src, re.MULTILINE) is None:
        lines_to_add.append("EPS = 1e-12")
    if "SUMMARY_CSV_PATH" not in src:
        lines_to_add.append("SUMMARY_CSV_PATH = RESULTS_DIR / 'summary_metrics.csv'")
    if "PER_CLASS_CSV_PATH" not in src:
        lines_to_add.append("PER_CLASS_CSV_PATH = RESULTS_DIR / 'per_class_coverage.csv'")
    if "RUN_CONFIG_JSON_PATH" not in src:
        lines_to_add.append("RUN_CONFIG_JSON_PATH = RESULTS_DIR / 'run_config.json'")

    if not lines_to_add:
        return False
    block = "\n".join(lines_to_add)
    return append_block_if_missing(
        nb, 4, "# ── Restored Baseline UQ Constants (auto-restored) ──", block
    )


def fix_sacp(suite_name: str, nb: dict) -> bool:
    block = (
        "MODEL_NAME_MAP = {'AlexNet_CNN': 'AlexNet', 'GFNet': 'GFNet', 'ViT_UNet': 'ViT'}\n"
        "TRAIN_PERCENT = 0.75\n"
        "CALIB_FRACTION_OF_TEST = 0.50\n"
        "BATCH_SIZE = 128\n"
        "EPS = 1e-12"
    )
    return append_block_if_missing(
        nb, 7, "# ── Restored SACP Constants (auto-restored) ──", block
    )


def fix_credit(suite_name: str, nb: dict) -> bool:
    if suite_name not in (
        "indian_pines_uncertainty_quantification",
        "pavia_university_uncertainty_quantification",
    ):
        return False
    changed = False
    src = get_src(nb, 4)
    hf_dir = SUITES[suite_name]["hf_model_dir"]
    old_ens = (
        f'ens_candidates = (glob.glob(f"/kaggle/input/{hf_dir}/**/*_ens_*_final.keras", recursive=True) '
        'or glob.glob("models/**/*_ens_*_final.keras", recursive=True))'
    )
    new_ens = (
        'ens_candidates = (\n'
        '    glob.glob("/kaggle/working/models/**/*_ens_*_final.keras", recursive=True)\n'
        f'    or glob.glob("/kaggle/input/{hf_dir}/**/*_ens_*_final.keras", recursive=True)\n'
        '    or glob.glob("/kaggle/input/**/*_ens_*_final.keras", recursive=True)\n'
        '    or glob.glob("models/**/*_ens_*_final.keras", recursive=True)\n'
        ')'
    )
    if old_ens in src:
        src = src.replace(old_ens, new_ens, 1)
        set_src(nb, 4, src)
        changed = True

    block = (
        "PLOT_DIR = RESULTS_DIR / 'training_plots'\n"
        "PLOT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "VAL_SPLIT_FROM_TRAIN = 0.20\n"
        "EPOCHS = 100\n"
        "DROPOUT_RATE = 0.25\n"
        "TRAIN_CFG = {\n"
        "    'label_smoothing': 0.05,\n"
        "    'weight_decay': 1e-4,\n"
        "    'clipnorm': 1.0,\n"
        "    'cosine_alpha': 0.05,\n"
        "}\n"
        "ALEXNET_CFG = {\n"
        "    'conv_filters': [96, 256, 384, 384, 256],\n"
        "    'dense_units': [4096, 1024, 256, 32],\n"
        "}\n"
        "GFNET_CFG = {'hidden_dim': 512, 'num_blocks': 5, 'mlp_ratio': 4}\n"
        "VIT_CFG = {\n"
        "    'projection_dim': 256,\n"
        "    'num_heads': 4,\n"
        "    'transformer_layers': 12,\n"
        "    'mlp_multiplier': 2,\n"
        "    'head_units': [512, 256, 128, 64],\n"
        "}\n"
        "ALEXNET_LEGACY_SPLIT_SEED = 10\n"
        "ALEXNET_LEGACY_TRAIN_PERCENT = 0.75\n"
        "ALEXNET_LR_START = 0.01\n"
        "ALEXNET_LR_MAX = 0.02\n"
        "ALEXNET_LR_MIN = 0.005"
    )
    if append_block_if_missing(nb, 4, "# ── Restored CREDIT Hyperparameters (auto-restored) ──", block):
        changed = True
    return changed


def fix_multi_training(suite_name: str, nb: dict) -> bool:
    ds_name = SUITES[suite_name]["dataset_name"]
    block = (
        "TRAINING_RESULTS_DIR = OUTPUT_ROOT\n"
        'TRAINING_WORKBOOK = TRAINING_RESULTS_DIR / "multihead_training_report.xlsx"\n'
        f"Data_to_use = {ds_name!r}\n"
        "use_pearson_corr = 0\n"
        "P_S = 9\n"
        "PATCH_SIZE = 9\n"
        "PERCENTILE_CLIP = (2.0, 98.0)\n"
        "Targeted_accuracy = 99\n"
        "Min_trainable_epoch = 45\n"
        "train_percent = 75\n"
        "epoch = 100\n"
        "BATCH_SIZE = 128\n"
        "dropout_rate = 0.25\n"
        "shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]"
    )
    return append_block_if_missing(
        nb, 4, "# ── Restored Multi-Head Training Config (auto-restored) ──", block
    )


def fix_multi_uncertainty(suite_name: str, nb: dict) -> bool:
    c1 = move_enable_unsafe_after_keras_import(nb, 2)
    block = (
        "pad_width = (PATCH_SIZE - 1) // 2\n"
        "padded_x = np.pad(x_img, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], mode='edge')\n"
        "Y = y\n"
        "y = y_img\n"
        "try:\n"
        "    x_train, x_test, y_train, y_test = train_test_split(\n"
        "        X, Y, train_size=TRAIN_PERCENT, stratify=Y, random_state=10\n"
        "    )\n"
        "except ValueError:\n"
        "    x_train, x_test, y_train, y_test = train_test_split(\n"
        "        X, Y, train_size=TRAIN_PERCENT, stratify=None, random_state=10\n"
        "    )\n"
        "try:\n"
        "    x_cal, x_test_new, y_cal, y_test_new = train_test_split(\n"
        "        x_test, y_test, test_size=0.5, stratify=y_test, random_state=10\n"
        "    )\n"
        "except ValueError:\n"
        "    x_cal, x_test_new, y_cal, y_test_new = train_test_split(\n"
        "        x_test, y_test, test_size=0.5, stratify=None, random_state=10\n"
        "    )\n"
        "print('padded_x:', padded_x.shape)\n"
        "print('x_cal:', x_cal.shape, 'y_cal:', y_cal.shape)\n"
        "print('x_test_new:', x_test_new.shape, 'y_test_new:', y_test_new.shape)"
    )
    c2 = append_block_if_missing(
        nb, 10, "# ── Restored MultiCP Split & Padded Tensor (auto-restored) ──", block
    )
    return c1 or c2


def fix_dapm_training(suite_name: str, nb: dict) -> bool:
    src = get_src(nb, 3)
    batch_line = "" if re.search(r"^\s*BATCH_SIZE\s*=", src, re.MULTILINE) else "BATCH_SIZE = 128\n"
    block = (
        "DAPM_DIR = MODEL_DIR\n"
        "DAPM_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "def _resolve_dapm_backbone(model_key):\n"
        "    cands = [\n"
        "        MODEL_DIR / f'{model_key}_best.keras',\n"
        "        Path(f'/kaggle/working/models/{model_key}_best.keras'),\n"
        "        Path(f'models/{model_key}_best.keras'),\n"
        "    ]\n"
        "    for c in cands:\n"
        "        if c.exists():\n"
        "            return c\n"
        "    for p in sorted(Path('/kaggle/input').glob(f'**/{model_key}_best.keras')):\n"
        "        return p\n"
        "    return cands[0]\n\n"
        "MODEL_KEYS = ['AlexNet_CNN', 'GFNet', 'ViT_UNet']\n"
        "MODEL_FILES = {k: _resolve_dapm_backbone(k) for k in MODEL_KEYS}\n\n"
        "FREEZE_BACKBONE = True\n"
        "MAX_TARGET_UNLABELED = 20000\n"
        "TARGET_VAL_FRACTION = 0.10\n\n"
        "LATENT_DIM = 64\n"
        "DECODER_HIDDEN_DIM = 256\n"
        "DISCRIM_HIDDEN_DIM = 128\n"
        "CLASSIFIER_HIDDEN_DIM = 128\n"
        "DIFFUSION_HIDDEN_DIM = 256\n"
        "T_EMBED_DIM = 32\n\n"
        "STAGE1_EPOCHS = 20\n"
        "STAGE2_EPOCHS = 20\n"
        + batch_line +
        "STAGE1_LR = 1e-3\n"
        "STAGE2_LR = 1e-3\n\n"
        "LAMBDA_SRC_RECON = 1.0\n"
        "LAMBDA_TGT_RECON = 1.0\n"
        "LAMBDA_KL = 0.01\n"
        "LAMBDA_CE = 1.0\n"
        "LAMBDA_DOMAIN = 0.2\n"
        "LAMBDA_TGT_DIFF = 0.5\n\n"
        "DIFFUSION_T = 100\n"
        "BETA_START = 1e-4\n"
        "BETA_END = 2e-2"
    )
    return append_block_if_missing(
        nb, 3, "# ── Restored DAPM Training Config (auto-restored) ──", block
    )


def fix_dapm_uncertainty(suite_name: str, nb: dict) -> bool:
    block_c3 = (
        "CHUNK_DIR = OUTPUT_DIR / 'chunk_outputs'\n"
        "MAP_DIR = OUTPUT_DIR / 'maps'\n"
        "CHUNK_DIR.mkdir(parents=True, exist_ok=True)\n"
        "MAP_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "N_SAMPLES = 30\n"
        "BATCH_SIZE = 128\n"
        "SCENE_CHUNK_SIZE = 1000\n"
        "P_THRESH = 0.05\n"
        "MODEL_NAME_MAP = {'AlexNet_CNN': 'AlexNet', 'GFNet': 'GFNet', 'ViT_UNet': 'ViT'}"
    )
    c1 = append_block_if_missing(
        nb, 3, "# ── Restored DAPM UQ Config (auto-restored) ──", block_c3
    )
    c2 = append_block_if_missing(
        nb, 5, "# ── Restored LOCAL_DAPM_DIR_ORIG (auto-restored) ──", "LOCAL_DAPM_DIR_ORIG = DAPM_DIR"
    )
    return c1 or c2


def fix_multicp_sacp(suite_name: str, nb: dict) -> bool:
    c1 = move_enable_unsafe_after_keras_import(nb, 5)
    block = (
        "TRAIN_PERCENT = 0.75\n"
        "CALIB_FRACTION_OF_TEST = 0.50\n"
        "BATCH_SIZE = 128\n"
        "EPS = 1e-12"
    )
    c2 = append_block_if_missing(
        nb, 8, "# ── Restored MultiCP-SACP Config (auto-restored) ──", block
    )
    return c1 or c2


FIXERS = {
    "ensemble/model_uncertainty_credal_ensemble.ipynb": fix_credal_ensemble,
    "baseline/model_uncertainty_comparison.ipynb": fix_baseline,
    "sacp/model_sacp_comparison.ipynb": fix_sacp,
    "credit/model_training_credit.ipynb": fix_credit,
    "multi_cp/model_training_multihead.ipynb": fix_multi_training,
    "multi_cp/model_uncertainty_multicp.ipynb": fix_multi_uncertainty,
    "dapm/model_training_dapm_full.ipynb": fix_dapm_training,
    "dapm/model_uncertainty_dapm_full.ipynb": fix_dapm_uncertainty,
    "multicp_sacp/Model_uncertainty_multicp_sacp.ipynb": fix_multicp_sacp,
}


def main():
    modified = 0
    for suite_name in SUITES:
        suite_dir = EXAMPLES_DIR / suite_name
        for rel_nb, fixer in FIXERS.items():
            nb_path = suite_dir / rel_nb
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            orig_len = len(nb["cells"])
            if fixer(suite_name, nb):
                assert len(nb["cells"]) == orig_len, f"Cell count changed in {nb_path}"
                nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
                modified += 1
                print(f"[FIXED] {suite_name}/{rel_nb} (cells={orig_len})")
            else:
                print(f"[OK]    {suite_name}/{rel_nb}")
    print(f"\nDone. Modified {modified} notebook(s).")


if __name__ == "__main__":
    main()
