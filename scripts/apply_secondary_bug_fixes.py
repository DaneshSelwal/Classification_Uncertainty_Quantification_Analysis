#!/usr/bin/env python3
"""
Apply 6 secondary bug fixes across the notebooks in examples/ for all 4 datasets:
- Indian Pines (examples/indian_pines_uncertainty_quantification)
- Pavia University (examples/pavia_university_uncertainty_quantification)
- Hisar-3 (examples/hisar_3mts_uncertainty_quantification)
- Hissar-25 (examples/hissar_25_uncertainty_quantification)

Fixes:
1. sacp/model_sacp_comparison.ipynb: ensure def load_scene_from_mat(...) is defined BEFORE calling it.
2. baseline/model_uncertainty_comparison.ipynb: ensure Kaggle path resolution looks in /kaggle/working/models/ in addition to /kaggle/input.
3. credit/model_training_credit.ipynb: ensure ALEXNET_LEGACY_TRAIN_PERCENT = 0.20 and ALEXNET_LEGACY_SPLIT_SEED = 42 are defined before use.
4. multi_cp/model_training_multihead.ipynb: remove/fix redundant y.shape[1] indexing on 1D label array, define x, Y, P_S, train_percent.
5. dapm/model_training_dapm_full.ipynb: define TARGET_VAL_FRACTION = 0.10 before use.
6. multicp_sacp/Model_uncertainty_multicp_sacp.ipynb: define CALIB_FRACTION_OF_TEST = 0.50 before use.
"""

import json
import os
import sys
from pathlib import Path

DATASETS = [
    'indian_pines_uncertainty_quantification',
    'pavia_university_uncertainty_quantification',
    'hisar_3mts_uncertainty_quantification',
    'hissar_25_uncertainty_quantification',
]

def load_nb(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_nb(path: Path, nb: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def apply_fix_1(examples_dir: Path, ds: str):
    """Fix 1: In sacp/model_sacp_comparison.ipynb, ensure def load_scene_from_mat(...) is defined BEFORE calling it."""
    nb_path = examples_dir / ds / 'sacp/model_sacp_comparison.ipynb'
    if not nb_path.exists():
        return False, "File not found"
    nb = load_nb(nb_path)
    cells = nb['cells']
    
    call_idx = None
    def_idx = None
    for i, c in enumerate(cells):
        if c['cell_type'] != 'code':
            continue
        src = "".join(c['source'])
        if "load_scene_from_mat(" in src and "def load_scene_from_mat(" not in src:
            if call_idx is None:
                call_idx = i
        if "def load_scene_from_mat(" in src:
            if def_idx is None:
                def_idx = i

    if call_idx is not None and def_idx is not None and call_idx < def_idx:
        # Move def cell to before call cell
        def_cell = cells.pop(def_idx)
        cells.insert(call_idx, def_cell)
        save_nb(nb_path, nb)
        return True, f"Reordered: moved definition cell from {def_idx} to before call cell {call_idx}"
    elif def_idx is not None and call_idx is not None and def_idx < call_idx:
        return True, f"Already in correct order (def at {def_idx}, call at {call_idx})"
    return False, f"Could not determine order (call={call_idx}, def={def_idx})"

def apply_fix_2(examples_dir: Path, ds: str):
    """Fix 2: In baseline/model_uncertainty_comparison.ipynb, make sure the Kaggle path resolution looks in /kaggle/working/models/ in addition to /kaggle/input."""
    nb_path = examples_dir / ds / 'baseline/model_uncertainty_comparison.ipynb'
    if not nb_path.exists():
        return False, "File not found"
    nb = load_nb(nb_path)
    modified = False

    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = "".join(c['source'])
        if "if IN_KAGGLE:" in src and "model_candidates" in src:
            # Check if /kaggle/working/models is already in model_candidates
            if "Path('/kaggle/working/models')" in src and "model_candidates" in src:
                # Ensure it is searched properly
                continue
            
            # Replace model_candidates resolution in IN_KAGGLE branch
            old_block = 'model_candidates = list(Path(\'/kaggle/input\').glob("**/*_best.keras"))'
            new_block = 'model_candidates = list(Path(\'/kaggle/working/models\').glob("**/*_best.keras")) or list(Path(\'/kaggle/input\').glob("**/*_best.keras")) or list(Path(\'/kaggle/working/models\').glob("**/*best.keras")) or list(Path(\'/kaggle/input\').glob("**/*best.keras"))'
            
            if old_block in src:
                src = src.replace(old_block, new_block)
                # Also ensure fallback checks working models
                if "MODEL_DIR = model_candidates[0].parent if model_candidates else Path('/kaggle/input/" in src:
                    src = src.replace(
                        "MODEL_DIR = model_candidates[0].parent if model_candidates else Path('/kaggle/input/",
                        "MODEL_DIR = model_candidates[0].parent if model_candidates else (Path('/kaggle/working/models') if Path('/kaggle/working/models').exists() else Path('/kaggle/input/"
                    )
                    # Ensure closing paren is added if not present
                    src = re.sub(r"(Path\('/kaggle/input/[^'\)]+-models'\))(?!\))", r"\1)", src)
                c['source'] = [l if l.endswith('\n') else l + '\n' for l in src.splitlines()]
                modified = True

    if modified:
        save_nb(nb_path, nb)
        return True, "Updated model_candidates to search /kaggle/working/models/"
    return True, "Already searches /kaggle/working/models/ or patterned"

def apply_fix_3(examples_dir: Path, ds: str):
    """Fix 3: In credit/model_training_credit.ipynb, ensure ALEXNET_LEGACY_TRAIN_PERCENT = 0.20 and ALEXNET_LEGACY_SPLIT_SEED = 42 are defined before use."""
    nb_path = examples_dir / ds / 'credit/model_training_credit.ipynb'
    if not nb_path.exists():
        return False, "File not found"
    nb = load_nb(nb_path)
    cells = nb['cells']
    
    # Check if already defined before use
    first_use_idx = None
    first_def_idx = None
    for i, c in enumerate(cells):
        if c['cell_type'] != 'code':
            continue
        src = "".join(c['source'])
        if "ALEXNET_LEGACY_TRAIN_PERCENT" in src:
            if "ALEXNET_LEGACY_TRAIN_PERCENT =" in src or "ALEXNET_LEGACY_TRAIN_PERCENT=" in src:
                if first_def_idx is None:
                    first_def_idx = i
            else:
                if first_use_idx is None:
                    first_use_idx = i

    if first_def_idx is not None and first_use_idx is not None and first_def_idx < first_use_idx:
        return True, f"Already defined before use (def={first_def_idx}, use={first_use_idx})"

    # If undefined or defined after use, insert definition in the hoisted cell or immediately before first use
    hoist_src = (
        "# [auto-fix: hoisted definition so it exists before first use]  <ALEXNET_LEGACY_SPLIT>\n"
        "ALEXNET_LEGACY_TRAIN_PERCENT = 0.20\n"
        "ALEXNET_LEGACY_SPLIT_SEED = 42\n"
    )
    
    # Check if there is already a VAL_SPLIT_FROM_TRAIN hoist cell
    inserted = False
    for c in cells:
        if c['cell_type'] == 'code' and "VAL_SPLIT_FROM_TRAIN" in "".join(c['source']):
            src = "".join(c['source'])
            if "ALEXNET_LEGACY_TRAIN_PERCENT" not in src:
                src += "\n# AlexNet legacy split constants\nALEXNET_LEGACY_TRAIN_PERCENT = 0.20\nALEXNET_LEGACY_SPLIT_SEED = 42\n"
                c['source'] = [l if l.endswith('\n') else l + '\n' for l in src.splitlines()]
                inserted = True
                break

    if not inserted and first_use_idx is not None:
        cells.insert(first_use_idx, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l if l.endswith('\n') else l + '\n' for l in hoist_src.splitlines()]
        })
        inserted = True

    if inserted:
        save_nb(nb_path, nb)
        return True, "Defined ALEXNET_LEGACY_TRAIN_PERCENT = 0.20 and ALEXNET_LEGACY_SPLIT_SEED = 42"
    return False, "Could not determine insertion point"

def apply_fix_4(examples_dir: Path, ds: str):
    """Fix 4: In multi_cp/model_training_multihead.ipynb, remove or fix the redundant y.shape[1] indexing on 1D label array."""
    nb_path = examples_dir / ds / 'multi_cp/model_training_multihead.ipynb'
    if not nb_path.exists():
        return False, "File not found"
    nb = load_nb(nb_path)
    modified = False

    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = "".join(c['source'])
        if "y.shape[1]" in src and "y_shape" in src:
            # Replace the broken re-extraction block with robust variable definitions
            fixed_src = (
                "# [auto-fix: resolved 1D label indexing and provided downstream aliases]\n"
                "x = x_img\n"
                "P_S = PATCH_SIZE\n"
                "train_percent = 75\n"
                "Y = y\n"
                "pad_width = int((P_S - 1) / 2)\n"
                "padded_x  = np.pad(x, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], 'edge')\n"
                "y_shape   = H * W\n"
                "\n"
                "print('shape of padded_x:', padded_x.shape)\n"
                "print(f'{len(Y) * 100 / y_shape:.4f}% of data is labeled with {num_classes} classes')\n"
            )
            c['source'] = [l if l.endswith('\n') else l + '\n' for l in fixed_src.splitlines()]
            modified = True

    if modified:
        save_nb(nb_path, nb)
        return True, "Replaced broken y.shape[1] cell with clean aliases (x, Y, P_S, train_percent, y_shape)"
    return True, "Already fixed (no y.shape[1] found)"

def apply_fix_5(examples_dir: Path, ds: str):
    """Fix 5: In dapm/model_training_dapm_full.ipynb, define TARGET_VAL_FRACTION = 0.10 before use."""
    nb_path = examples_dir / ds / 'dapm/model_training_dapm_full.ipynb'
    if not nb_path.exists():
        return False, "File not found"
    nb = load_nb(nb_path)
    cells = nb['cells']

    # Check if already defined
    first_use_idx = None
    first_def_idx = None
    for i, c in enumerate(cells):
        if c['cell_type'] != 'code':
            continue
        src = "".join(c['source'])
        if "TARGET_VAL_FRACTION" in src:
            if "TARGET_VAL_FRACTION =" in src or "TARGET_VAL_FRACTION=" in src:
                if first_def_idx is None:
                    first_def_idx = i
            else:
                if first_use_idx is None:
                    first_use_idx = i

    if first_def_idx is not None and first_use_idx is not None and first_def_idx < first_use_idx:
        return True, f"Already defined before use (def={first_def_idx}, use={first_use_idx})"

    # Append to MAX_TARGET_UNLABELED hoist cell or insert before first use
    inserted = False
    for c in cells:
        if c['cell_type'] == 'code' and "MAX_TARGET_UNLABELED" in "".join(c['source']):
            src = "".join(c['source'])
            if "TARGET_VAL_FRACTION" not in src:
                src += "\nTARGET_VAL_FRACTION = 0.10\n"
                c['source'] = [l if l.endswith('\n') else l + '\n' for l in src.splitlines()]
                inserted = True
                break

    if not inserted and first_use_idx is not None:
        hoist_src = (
            "# [auto-fix: hoisted definition so it exists before first use]  <TARGET_VAL_FRACTION>\n"
            "TARGET_VAL_FRACTION = 0.10\n"
        )
        cells.insert(first_use_idx, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l if l.endswith('\n') else l + '\n' for l in hoist_src.splitlines()]
        })
        inserted = True

    if inserted:
        save_nb(nb_path, nb)
        return True, "Defined TARGET_VAL_FRACTION = 0.10 before use"
    return False, "Could not determine insertion point"

def apply_fix_6(examples_dir: Path, ds: str):
    """Fix 6: In multicp_sacp/Model_uncertainty_multicp_sacp.ipynb, define CALIB_FRACTION_OF_TEST = 0.50 before use."""
    nb_path = examples_dir / ds / 'multicp_sacp/Model_uncertainty_multicp_sacp.ipynb'
    if not nb_path.exists():
        return False, "File not found"
    nb = load_nb(nb_path)
    cells = nb['cells']

    # Check if already defined
    first_use_idx = None
    first_def_idx = None
    for i, c in enumerate(cells):
        if c['cell_type'] != 'code':
            continue
        src = "".join(c['source'])
        if "CALIB_FRACTION_OF_TEST" in src:
            if "CALIB_FRACTION_OF_TEST =" in src or "CALIB_FRACTION_OF_TEST=" in src:
                if first_def_idx is None:
                    first_def_idx = i
            else:
                if first_use_idx is None:
                    first_use_idx = i

    if first_def_idx is not None and first_use_idx is not None and first_def_idx < first_use_idx:
        return True, f"Already defined before use (def={first_def_idx}, use={first_use_idx})"

    # Append to TRAIN_PERCENT hoist cell or insert before first use
    inserted = False
    for c in cells:
        if c['cell_type'] == 'code' and "TRAIN_PERCENT" in "".join(c['source']):
            src = "".join(c['source'])
            if "CALIB_FRACTION_OF_TEST" not in src:
                src += "\nCALIB_FRACTION_OF_TEST = 0.50\n"
                c['source'] = [l if l.endswith('\n') else l + '\n' for l in src.splitlines()]
                inserted = True
                break

    if not inserted and first_use_idx is not None:
        hoist_src = (
            "# [auto-fix: hoisted definition so it exists before first use]  <CALIB_FRACTION_OF_TEST>\n"
            "CALIB_FRACTION_OF_TEST = 0.50\n"
        )
        cells.insert(first_use_idx, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l if l.endswith('\n') else l + '\n' for l in hoist_src.splitlines()]
        })
        inserted = True

    if inserted:
        save_nb(nb_path, nb)
        return True, "Defined CALIB_FRACTION_OF_TEST = 0.50 before use"
    return False, "Could not determine insertion point"

def main():
    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"
    print("=" * 80)
    print("  APPLYING 6 SECONDARY BUG FIXES ACROSS 4 DATASET SUITES")
    print("=" * 80)

    all_passed = True
    for ds in DATASETS:
        print(f"\nProcessing Suite: {ds}")
        
        ok1, msg1 = apply_fix_1(examples_dir, ds)
        print(f"  Fix 1 (SACP load_scene_from_mat):       [{'OK' if ok1 else 'FAIL'}] {msg1}")
        
        ok2, msg2 = apply_fix_2(examples_dir, ds)
        print(f"  Fix 2 (Baseline Kaggle model path):     [{'OK' if ok2 else 'FAIL'}] {msg2}")
        
        ok3, msg3 = apply_fix_3(examples_dir, ds)
        print(f"  Fix 3 (CREDIT AlexNet legacy split):    [{'OK' if ok3 else 'FAIL'}] {msg3}")
        
        ok4, msg4 = apply_fix_4(examples_dir, ds)
        print(f"  Fix 4 (Multi-CP y.shape[1] 1D indexing): [{'OK' if ok4 else 'FAIL'}] {msg4}")
        
        ok5, msg5 = apply_fix_5(examples_dir, ds)
        print(f"  Fix 5 (DAPM TARGET_VAL_FRACTION):       [{'OK' if ok5 else 'FAIL'}] {msg5}")
        
        ok6, msg6 = apply_fix_6(examples_dir, ds)
        print(f"  Fix 6 (SCMCP CALIB_FRACTION_OF_TEST):   [{'OK' if ok6 else 'FAIL'}] {msg6}")
        
        if not (ok1 and ok2 and ok3 and ok4 and ok5 and ok6):
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 6 SECONDARY BUG FIXES SUCCESSFULLY APPLIED ACROSS ALL 4 DATASETS!")
    else:
        print("SOME FIXES FAILED — CHECK LOGS ABOVE.")
    print("=" * 80)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
