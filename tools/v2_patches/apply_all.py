import os, subprocess, sys
F = sys.argv[1]
P = {"baseline/model_training":"p01_baseline_training","baseline/model_uncertainty_comparison":"p02_baseline_cp",
"sacp/model_sacp_comparison":"p03_sacp","ensemble/model_training_ensembles":"p04_ensemble","credit/model_training_credit":"p05b_credit_generic",
"ensemble/model_uncertainty_credal_ensemble":"p06_crede","multi_cp/model_training_multihead":"p07_multihead",
"multi_cp/model_uncertainty_multicp":"p08_multicp","dapm/Model_uncertainty_dapm_cp":"p09_dapm_cp",
"dapm/model_training_dapm_full":"p10_dapm_train","dapm/model_uncertainty_dapm_full":"p11_dapm_uq",
"multicp_sacp/Model_uncertainty_multicp_sacp":"p12_multicp_sacp"}
BASE = "237ab65"   # v1 (origin/main) originals
bad = 0
for s in sorted(x for x in os.listdir("examples") if not x.startswith("indian")):
    for nb, p in P.items():
        path = f"examples/{s}/{nb}.ipynb"
        subprocess.run(["git", "checkout", BASE, "--", path], check=True)   # always patch from the v1 original
        r = subprocess.run([sys.executable, f"{F}/{p}.py", path, F], capture_output=True, text=True)
        ok = r.returncode == 0; bad += not ok
        print(("OK  " if ok else "FAIL"), s, nb, (r.stdout + r.stderr).strip().splitlines()[-1][:160])
print("FAILED:", bad)
