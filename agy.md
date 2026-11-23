# agy.md: instructions for agy (Gemini) to finish the v2.0 classification runs

Read this whole file before doing anything. Everything needed is already set up; your job is to **run the
plan, watch it, fix generic bugs if a smoke run fails, and commit the results**. Do not redesign anything.

## 1. Where things are

| What | Where |
|---|---|
| Repository | https://github.com/DaneshSelwal/Classification_Uncertainty_Quantification_Analysis |
| Code branch (fixed notebooks + tools) | `version2.0` (local worktree `/Users/danesh/Desktop/Reports_dias/Classification_v2.0`) |
| Results so far | branch `temp-results-v2.0` (= `version2.0` + computed results) |
| Fix list (what changed vs v1, why) | `CHANGELOG_FIXES.md` |
| Run plan (what still has to run) | `docs/run_plan.tsv` |
| Every Kaggle run ever pushed | `docs/run_registry.tsv` |
| GPU hours per account | `docs/gpu_ledger.tsv` |
| Kaggle conventions | `docs/KAGGLE_RUNS.md` |

The 11 datasets ("suites") are `examples/<suite>_uncertainty_quantification/`. Each has **12 pipeline
notebooks** that run in this order (stage 0 to 11) in ONE Kaggle GPU kernel:

```
0 baseline/model_training             4 credit/model_training_credit                 8 dapm/model_training_dapm_full
1 baseline/model_uncertainty_comparison 5 ensemble/model_uncertainty_credal_ensemble 9 dapm/model_uncertainty_dapm_full
2 sacp/model_sacp_comparison          6 multi_cp/model_training_multihead             10 dapm/Model_uncertainty_dapm_cp
3 ensemble/model_training_ensembles   7 multi_cp/model_uncertainty_multicp            11 multicp_sacp/Model_uncertainty_multicp_sacp
```
Other notebooks in a suite (root copies such as `Model_training.ipynb`, `ensemble/Model_uncertainty_CreDE.ipynb`,
the DIAS / 6-Band folders `dofa`, `georsclip`, `mambahsi`, `spatialgcn`, `trials`) are **not** part of v2.0.

## 2. Rules (must follow)

1. Kaggle accounts: **only** `daneshselwal` (token dir `~/.kaggle`) and `daneshjangra` (token dir `~/.kaggle_alt`).
   Never use geminicli, daneshselwal1, daneshselwal2. Never print, copy or commit a token.
2. Max **2 GPU sessions per account** (4 in total). Every run is named `CLA<DD>_<MM>_<YYYY>_<HH>_<MM>`.
   At most **one push per minute**, shared with the regression project: always push through
   `tools/kaggle/queue_push.sh` (or `run_plan.py`, which calls it). Never push with `kaggle kernels push` directly.
3. Weekly GPU limit is **30 h per account**; it resets on the date in `docs/gpu_ledger.tsv` (column refresh).
   The API's "used" number is NOT what Kaggle enforces: on 2026-11-23 both accounts were refused with
   "Maximum weekly GPU quota of 30.00 hours reached" while the API showed ~10 h. Trust the push error.
4. Do not change any maths (conformal quantiles, scores, smoothing, uncertainty formulas, split design) without
   the owner's approval. Non-maths fixes (paths, file names, plotting, label lists, crashes) are allowed, must
   be **generic** (same fix in every suite where it applies) and must be logged in `CHANGELOG_FIXES.md`.
5. Do not modify other branches (`main`, old version branches). Do not touch files outside the worktree.
6. No heavy local compute (no local training). The local checks below take seconds to minutes.
7. Commit only final results; never commit model weights (`.keras`, `.h5`).

## 3. Current status (2026-11-23)

| Dataset (suite) | Smoke run (short epochs) | Full run |
|---|---|---|
| Indian Pines | OK (CLA22_11_2026_21_07) | **DONE, 12/12 OK** (CLA23_11_2026_01_04), results in `examples/indian_pines_uncertainty_quantification/logs/v2.0_CLA23_11_2026_01_04/` |
| Pavia University | OK, 12/12 (CLA23_11_2026_01_05) | pending |
| Achla multi-temporal | OK, 12/12 (CLA23_11_2026_01_06) | pending (daneshjangra only) |
| Planet Hisar | **12 h timeout** (CLA23_11_2026_01_07): stages 0-8 OK, no error; 9-11 not reached | **blocked: owner decision** (see 5, "Long kernels") |
| Hisar-3 MTS | pending | pending |
| Hissar-25 | pending | pending |
| Pavia Centre | pending | pending |
| ETM multispectral | pending | pending |
| DIAS | pending | pending |
| 6-Band | pending | pending |
| 372-Band | pending | pending |

Local checks already passed for all suites: `tools/check_notebooks.py` (all 132 notebooks) and
`tools/dry_run_data.py` (data loading + shared split + alignment for every suite with data available locally;
DIAS and 6-Band also with `--fake-models`, i.e. every stage finds the models of the earlier stages).
372-Band (1.1 GB) and Achla / Planet Hisar (data only on Kaggle) were checked statically only.

Measured run times: smoke 2.4 h (Pavia University) to 4.3 h (Achla), Planet Hisar smoke > 12 h (DAPM training alone 4 h); full 3.6 h (Indian Pines). Larger scenes
(372-Band, Hisar-3, Pavia Centre) will be longer. Expect roughly 20 h for the 7 smokes and 40-60 h for the 10
fulls, i.e. **more than one week of quota (2 x 30 h)**. `run_plan.py` simply continues after the reset.

## 4. What to do, step by step

### Step 0: get the code
```bash
cd /Users/danesh/Desktop/Reports_dias/Classification_v2.0      # existing worktree on branch version2.0
git status                                                      # must be clean
git pull --ff-only origin version2.0
```
(If the worktree is missing: `git clone -b version2.0 <repo> Classification_v2.0`.)

### Step 1: check quota and run the local checks
```bash
zsh tools/kaggle/gpu_ledger.sh            # appends both accounts' GPU hours to docs/gpu_ledger.tsv
python3 tools/check_notebooks.py          # must end with "RESULT: OK"
python3 tools/kaggle/run_plan.py --status # shows what is done / pending
```

### Step 2: start the plan (it runs unattended)
```bash
nohup python3 tools/kaggle/run_plan.py > docs/run_plan.log 2>&1 &
```
`run_plan.py` loops every 5 minutes. It:
* collects finished runs: downloads stage status, log, tables and executed notebooks, and marks the row
  `DONE_OK` or `DONE_FAILED`;
* starts the next allowed row: smokes first, then a suite's `full` only after its smoke is `DONE_OK`;
* keeps at most 2 sessions per account, and stops using an account once Kaggle says its weekly quota is used up;
* updates `docs/gpu_ledger.tsv` every hour;
* exits when nothing is left, or nothing can start because the quota is used up. After the weekly reset,
  just start it again.

Progress: `python3 tools/kaggle/run_plan.py --status` and `tail docs/run_plan.log`.
To poll and collect without starting anything new: `python3 tools/kaggle/run_plan.py --once --no-push`.

### Step 3: when a smoke run is DONE_FAILED
1. Open the failing notebook: `docs/smoke_runs/<CODE>/output/EXECUTED_<stage>_<name>.ipynb` and find the
   cell with the error (the log is `docs/smoke_runs/<CODE>/<slug>.log`).
2. Decide whether it is a maths issue (stop, ask the owner) or a generic non-maths bug (path, file name,
   shape, plotting, label list).
3. Fix it in **every suite where the same code exists**, mark the line with `# FIX <ID>`, and add the ID to
   `CHANGELOG_FIXES.md`. Helpers: `tools/v2_patches/nbpatch.py` (class `NB`, `common_fixes`, ...) shows
   how earlier fixes were applied. Those scripts were already applied, so do NOT re-run them blindly.
4. Re-check locally:
   ```bash
   python3 tools/check_notebooks.py
   # data dry run (download the dataset once into its own folder):
   mkdir -p /tmp/agy_in/<suite>/datasets/<owner>/<slug>
   kaggle datasets download <owner>/<slug> -p /tmp/agy_in/<suite>/datasets/<owner>/<slug> --unzip
   python3 tools/dry_run_data.py examples/<suite> /tmp/agy_in/<suite> --fake-models
   ```
   (`<owner>/<slug>` is in `tools/kaggle/suites.json`. For the download, set
   `KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)` in the same command and never echo it.)
5. Commit the fix, set the smoke row back to `PENDING` in `docs/run_plan.tsv` (clear code_name, slug and
   note), and restart `run_plan.py`.

### Step 4: when a full run is DONE_OK, verify it
Results are in `examples/<suite>/logs/v2.0_<CODE>/` (tables `*.csv`, `stage_status.json`, executed notebooks).
Check that:
* all 12 stages are ok (`stage_status.json`) and no executed notebook contains an error output;
* `summary_metrics.csv`: SplitConformal / RAPS / ClusteredConformal empirical coverage is close to the target
  0.95 (roughly 0.93-0.98). Class-conditional CP / RC3P can be lower for a weak model (Indian Pines AlexNet:
  0.885); report it, do not "fix" it;
* `sacp_all_windows_summary.csv` and `multicp_summary_metrics.csv` look similar (coverage near or above 0.95);
* the log line `split alignment verified against: [...]` appears in the notebooks (shared split is used).
Then add one line per finished suite to the "Results" section of `docs/KAGGLE_RUNS.md` (accuracy, coverage).

### Step 5: commit and push
* Code fixes go to `version2.0`. Results (`examples/*/logs/v2.0_*`, `docs/smoke_runs/`) go to
  `temp-results-v2.0`:
  ```bash
  git add -f tools/ ; git add docs CHANGELOG_FIXES.md examples/*/*/*.ipynb ; git commit -m "..."  # on version2.0
  git push origin version2.0
  git checkout temp-results-v2.0 && git merge --no-edit version2.0
  git add examples/*/logs/v2.0_* docs/smoke_runs ; git commit -m "results: <suites>" ; git push origin temp-results-v2.0
  git checkout version2.0
  ```
* Never commit `.kaggle_jobs/`, model weights, or tokens. (`.keras` / `.h5` files are not downloaded by
  `run_plan.py`.)

### Step 6: final report to the owner
One table, **Dataset | Notebook | Status**, with all 11 datasets x 12 notebooks expanded, plus the GPU
hours used per account and anything that needed the owner's decision.

## 5. Known facts and pitfalls

* **Data access:** Achla (`daneshjangra/achla-dataset-k08`) and Planet Hisar (`daneshjangra/planethisar-dataset-k09`)
  are readable only by daneshjangra; all other datasets are owned by daneshselwal but readable by both.
  `docs/run_plan.tsv` pins those two suites to daneshjangra.
* **Token file names:** Kaggle datasets store files as `k001.mat`, `k002.csv`, ... The kernel runner
  (`tools/kaggle/make_suite_kernel.py`) builds `/kaggle/kin` with links under the real names
  (`tools/kaggle/suites.json` → `files`) and rewrites `/kaggle/input` to it in the notebooks.
* **DIAS and 6-Band** were written for Colab; the runner turns `drive.mount` into a no-op and maps
  `/content/drive/My Drive/Classification` to `/kaggle/working`. DIAS stages pick the models of the same run
  (FIX K5) instead of the old v1 model datasets.
* **Spatial split:** some small classes have no evaluation pixel (Indian Pines: 4 of 16). Their per-class
  coverage is NaN. This is expected under the spatial split, and changing it is a design decision for the owner.
* **Kaggle errors:** "Maximum batch GPU session count of 2 reached" and "Expecting value: line 1 column 1" mean
  the account's sessions are busy (the queue retries). "Maximum weekly GPU quota" means stop until the reset.
* **Long kernels:** Kaggle stops a session after 12 h. If a full run times out, `stage_status.json` shows the
  stages that finished. Re-run only the missing stages with
  `python3 tools/kaggle/make_suite_kernel.py examples/<suite> <job_dir> <account> --stages 8,9,10,11`.
  This only works if the earlier stages' models are present, so normally re-run the whole suite and tell the owner.
* **Planet Hisar does not fit in one 12 h session.** Its smoke run (CLA23_11_2026_01_07, log and
  `stage_status.json` in `docs/smoke_runs/CLA23_11_2026_01_07/`) finished stages 0-8 without any error and was
  stopped by Kaggle at 12 h (694 min) before stages 9-11 (DAPM uncertainty, DAPM CP, Multi-CP+SACP). Its full run
  (more epochs) will not fit either, so `run_plan.py` will not start it (the smoke row is DONE_FAILED). Do NOT
  shorten epochs or change the DAPM settings yourself (that changes results). Ask the owner which option to use:
  (a) split the suite into two kernels (stages 0-8, then 9-11 with the first kernel's `/kaggle/working` output
  attached as input, which needs a small change to `make_suite_kernel.py`), or (b) accept the smoke for 0-8 and
  run only the full flow split as in (a). Watch 372-Band, Hisar-3 MTS and Pavia Centre the same way: if a smoke
  of theirs times out, report it the same way.
* **Regression project:** it shares the push lock and the Kaggle accounts (CPU runs). Do not delete its lock
  directory (`.../Regression_Uncertainty_Quantification_Analysis/docs/run_registry.tsv.lock`) unless no
  push is running for more than 10 minutes.
