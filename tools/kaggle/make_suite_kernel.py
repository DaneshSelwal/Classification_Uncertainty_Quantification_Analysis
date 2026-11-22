"""Build a Kaggle GPU *script* job that runs the 12 v2.0 notebooks of one dataset suite in order
with papermill (no pre-trained weights are downloaded: every model is trained in the run).

usage: make_suite_kernel.py <suite_dir> <out_dir> <dataset_source> [--smoke] [--stages 0,1,...]

--smoke : shrink epochs / ensemble size / samples so every stage runs end-to-end in minutes.
          (Only run-length settings change; no maths is touched.)
Every executed notebook (with outputs, also when a cell fails) and stage_status.json are saved
to /kaggle/working/output. A stage that fails does not stop the later stages.
"""
import base64, io, json, os, sys, tarfile

PIPELINE = [
    (0, "baseline/model_training.ipynb"),
    (1, "baseline/model_uncertainty_comparison.ipynb"),
    (2, "sacp/model_sacp_comparison.ipynb"),
    (3, "ensemble/model_training_ensembles.ipynb"),
    (4, "credit/model_training_credit.ipynb"),
    (5, "ensemble/model_uncertainty_credal_ensemble.ipynb"),
    (6, "multi_cp/model_training_multihead.ipynb"),
    (7, "multi_cp/model_uncertainty_multicp.ipynb"),
    (8, "dapm/model_training_dapm_full.ipynb"),
    (9, "dapm/model_uncertainty_dapm_full.ipynb"),
    (10, "dapm/Model_uncertainty_dapm_cp.ipynb"),
    (11, "multicp_sacp/Model_uncertainty_multicp_sacp.ipynb"),
]
# run-length settings only (regex on a whole line at top level) -> smoke values
SMOKE_RULES = [
    (r"^EPOCHS(\s*)=\s*\d+", "EPOCHS = 2"),
    (r"^STUDENT_EPOCHS(\s*)=\s*\d+", "STUDENT_EPOCHS = 2"),
    (r"^STAGE1_EPOCHS(\s*)=\s*\d+", "STAGE1_EPOCHS = 1"),
    (r"^STAGE2_EPOCHS(\s*)=\s*\d+", "STAGE2_EPOCHS = 1"),
    (r"^MAX_TARGET_UNLABELED(\s*)=\s*\d+", "MAX_TARGET_UNLABELED = 2000"),
    (r"^N_SAMPLES(\s*)=\s*\d+", "N_SAMPLES = 5"),
    (r"^M(\s*)=\s*\d+", "M = 2"),
    (r"^epoch(\s*)=\s*\d+", "epoch = 2"),
    (r"^Min_trainable_epoch(\s*)=\s*\d+", "Min_trainable_epoch = 1"),
    (r"^SACP_WINDOW_SIZES(\s*)=\s*\[[^\]]*\]", "SACP_WINDOW_SIZES = [3, 5]"),
]

args = sys.argv[1:]
smoke = "--smoke" in args; args = [a for a in args if a != "--smoke"]
stages = list(range(12))
if "--stages" in args:
    i = args.index("--stages"); stages = [int(s) for s in args[i + 1].split(",")]; del args[i:i + 2]
suite_dir, out_dir, dataset = args
os.makedirs(out_dir, exist_ok=True)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for _, rel in PIPELINE:
        tar.add(os.path.join(suite_dir, rel), arcname=rel)
payload = base64.b64encode(buf.getvalue()).decode()
suite = os.path.basename(os.path.normpath(suite_dir))

run_py = f'''# v2.0 suite runner: {suite} | smoke={smoke} | stages={stages}
import base64, io, json, os, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path
SUITE, SMOKE, STAGES = {suite!r}, {smoke!r}, {stages!r}
PIPELINE = {PIPELINE!r}
SMOKE_RULES = {SMOKE_RULES!r}
W = Path("/kaggle/working"); SRC = Path("/kaggle/tmp_suite"); OUT = W / "output"
OUT.mkdir(parents=True, exist_ok=True); SRC.mkdir(parents=True, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})), mode="r:gz").extractall(SRC)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "papermill"], check=True)
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)

def smoke_patch(path):
    nb = json.load(open(path)); n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code": continue
        s = "".join(c["source"])
        for pat, new in SMOKE_RULES:
            s, k = re.subn(pat, new, s, flags=re.M); n += k
        c["source"] = s.splitlines(keepends=True)
    json.dump(nb, open(path, "w"), indent=1)
    return n

status, T0 = [], time.time()
for st, rel in PIPELINE:
    if st not in STAGES: continue
    src = SRC / rel
    n = smoke_patch(src) if SMOKE else 0
    out = OUT / ("EXECUTED_%02d_" % st + Path(rel).name)
    t0 = time.time()
    print("\\n" + "=" * 70 + f"\\nSTAGE {{st}}: {{rel}}  (smoke overrides: {{n}})\\n" + "=" * 70, flush=True)
    r = subprocess.run([sys.executable, "-m", "papermill", str(src), str(out), "--cwd", str(W),
                        "-k", "python3", "--log-output", "--request-save-on-cell-execute",
                        "--autosave-cell-every", "60"])
    mins = (time.time() - t0) / 60
    status.append({{"stage": st, "notebook": rel, "ok": r.returncode == 0, "minutes": round(mins, 2)}})
    print(f"STAGE {{st}} {{'OK' if r.returncode == 0 else 'FAILED'}} in {{mins:.1f}} min", flush=True)
    json.dump({{"suite": SUITE, "smoke": SMOKE, "stages": status,
               "total_minutes": round((time.time() - T0) / 60, 2)}}, open(OUT / "stage_status.json", "w"), indent=1)

if SMOKE:   # smoke runs keep notebooks + tables only (no model weights) to keep the output small
    for p in W.rglob("*.keras"): p.unlink()
    for p in W.rglob("*.h5"): p.unlink()
print("\\nSUMMARY"); [print(s) for s in status]
print("RUN_STATUS:", "OK" if all(s["ok"] for s in status) else "FAILED", f"total {{(time.time() - T0) / 60:.1f}} min")
'''
open(os.path.join(out_dir, "run.py"), "w").write(run_py)
meta = {"id": "PLACEHOLDER/placeholder", "title": "placeholder", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [dataset], "kernel_sources": [], "competition_sources": [], "model_sources": []}
json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=1)
print("built", out_dir, "| suite", suite, "| smoke", smoke, "| stages", stages, "|", len(payload) // 1024, "KB")
