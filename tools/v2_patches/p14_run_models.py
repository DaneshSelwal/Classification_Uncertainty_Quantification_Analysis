# FIX K5: DIAS consumer notebooks read models from separate v1 Kaggle datasets (dias-baseline-models,
# dias-ensemble-*) under coded names whose meaning differs between notebooks (k005 = AlexNet in SACP, ViT in
# DAPM). In a v2.0 pipeline run the models of the earlier stages of the same run (original names under
# /kaggle/working) must be used. A cell is inserted after the cell that defines the path variables.
# usage: python3 -I p14_run_models.py <suite_dir>
import json, sys
from pathlib import Path

HEAD = '''# FIX K5 - v2.0 pipeline run: use the models trained by the earlier stages of THIS run.
# The v1 notebook read them from separate Kaggle datasets under coded names (k0NN) whose meaning differs
# between notebooks. When the original file names exist under /kaggle/working they take precedence.
from pathlib import Path as _Path
def _this_run(name):
    root = _Path("/kaggle/working")
    hits = sorted(root.rglob(name)) if root.exists() else []
    return hits[0] if hits else None
_MODELS = ("AlexNet_CNN", "GFNet", "ViT_UNet")
_RUN_BASE = {m: _this_run(f"{m}_best.keras") for m in _MODELS}
_RUN_MH   = {m: _this_run(f"{m}_MultiHead_best.keras") for m in _MODELS}
_RUN_ENS  = next((p.parent for m in _MODELS for p in [_this_run(f"{m}_ens_1_final.keras")] if p), None)
_RUN_DAPM = next((p.parent for m in _MODELS for p in [_this_run(f"{m}_dapm_full_encoder.weights.h5")] if p), None)
_RUN_REG  = _this_run("model_registry_multihead.json")
def _swap(d, new, suffix=""):
    out = {}
    for k, v in d.items():
        base = k[:-len(suffix)] if suffix and k.endswith(suffix) else k
        out[k] = type(v)(new[base]) if new.get(base) else v
    return out
'''
ROLE = {
    "baseline_files": '''if all(_RUN_BASE.values()):
    MODEL_FILES = _swap(MODEL_FILES, _RUN_BASE)
    MODEL_DIR = _RUN_BASE["GFNet"].parent
    print("FIX K5: baseline models of this run ->", MODEL_DIR)
''',
    "ensemble_dir_credit": '''if _RUN_ENS is not None:
    ENSEMBLE_DIR = _RUN_ENS
    print("FIX K5: ensemble members of this run ->", ENSEMBLE_DIR)
''',
    "ensemble_dir_model_dir": '''if _RUN_ENS is not None:
    MODEL_DIR = _RUN_ENS
    print("FIX K5: ensemble members of this run ->", MODEL_DIR)
''',
    "multihead_resolved": '''if all(_RUN_MH.values()):
    RESOLVED_MULTIHEAD_MODELS = _swap(RESOLVED_MULTIHEAD_MODELS, _RUN_MH, "_MultiHead")
    MODEL_DIR = _RUN_MH["GFNet"].parent
    if _RUN_REG is not None: MODEL_REGISTRY_PATH = _RUN_REG
    print("FIX K5: multi-head models of this run ->", MODEL_DIR)
''',
    "multihead_dir": '''if all(_RUN_MH.values()):
    MODEL_DIR = MULTICP_DIR = _RUN_MH["GFNet"].parent
    if _RUN_REG is not None: MODEL_REGISTRY_PATH = _RUN_REG
    print("FIX K5: multi-head models of this run ->", MODEL_DIR)
''',
    "dapm_dir": '''if _RUN_DAPM is not None:
    DAPM_DIR = _RUN_DAPM
    print("FIX K5: DAPM artefacts of this run ->", DAPM_DIR)
''',
}
# suite -> notebook -> (index of the cell that defines the path variables, roles)
PLANS = {"dias": {
    "baseline/model_uncertainty_comparison.ipynb": (4, ["baseline_files"]),
    "sacp/model_sacp_comparison.ipynb": (7, ["baseline_files"]),
    "credit/model_training_credit.ipynb": (4, ["ensemble_dir_credit"]),
    "ensemble/model_uncertainty_credal_ensemble.ipynb": (7, ["ensemble_dir_model_dir"]),
    "multi_cp/model_uncertainty_multicp.ipynb": (4, ["multihead_resolved"]),
    "dapm/model_training_dapm_full.ipynb": (3, ["baseline_files"]),
    "dapm/model_uncertainty_dapm_full.ipynb": (5, ["baseline_files", "dapm_dir"]),
    "dapm/Model_uncertainty_dapm_cp.ipynb": (4, ["baseline_files", "dapm_dir"]),
    "multicp_sacp/Model_uncertainty_multicp_sacp.ipynb": (8, ["multihead_dir"]),
}, "6_band": {   # DAPM training saves to dapm/dapm_full_artifacts, v1 consumers read dapm/models/dapm_full_artifacts
    "dapm/model_uncertainty_dapm_full.ipynb": (3, ["dapm_dir"]),
    "dapm/Model_uncertainty_dapm_cp.ipynb": (4, ["dapm_dir"]),
}}
suite = Path(sys.argv[1])
PLAN = PLANS[suite.name.replace("_uncertainty_quantification", "")]
for rel, (idx, roles) in PLAN.items():
    p = suite / rel
    nb = json.loads(p.read_text())
    if any("FIX K5" in "".join(c["source"]) for c in nb["cells"]):
        print("already patched", rel); continue
    assert nb["cells"][idx]["cell_type"] == "code", (rel, idx)
    if rel.startswith("credit/"):   # the credit cell uses ENSEMBLE_DIR before the inserted cell runs
        c = nb["cells"][idx]; s = "".join(c["source"])
        old = "if ENSEMBLE_DIR.exists():"
        assert old in s, rel
        s = s.replace(old, "if ENSEMBLE_DIR is not None and ENSEMBLE_DIR.exists():  # FIX K5", 1)
        c["source"] = s.splitlines(keepends=True)
    src = HEAD + "".join(ROLE[r] for r in roles)
    cell = {"cell_type": "code", "execution_count": None, "id": f"fixk5{idx:03d}", "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}
    nb["cells"].insert(idx + 1, cell)
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("patched", rel, "after cell", idx, roles)
