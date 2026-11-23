# Kaggle runs (version2.0)

* Every run is named `CLA<DD>_<MM>_<YYYY>_<HH>_<MM>` = local push time (slug `cla<dd>-<mm>-<yyyy>-<hh>-<mm>`).
* At most one push per minute, shared with the regression project (common lock, both registries checked):
  `tools/kaggle/queue_push.sh "<note>" <job_dir>`.
* Job = one dataset suite, 12 notebooks in order, GPU + internet, no pre-trained weights:
  `python3 tools/kaggle/make_suite_kernel.py examples/<suite> <job_dir> <kaggle_dataset> [--smoke]`.
  `--smoke` only shortens run length (epochs, ensemble size M = 2, diffusion samples, SACP windows 3/5).
* Accounts: daneshselwal (`~/.kaggle`) and daneshjangra (`~/.kaggle_alt`), 2 GPU sessions each.
* GPU budget: `tools/kaggle/gpu_ledger.sh` appends the weekly GPU use of both accounts to
  `docs/gpu_ledger.tsv`. (The API reports `totalTimeAllowed` = 6 h, but daneshjangra already shows
  22.8 h used, so the real weekly allowance is higher; the Kaggle settings page is authoritative.)
  The real weekly limit is 30 h per account (Kaggle push error). A push is refused when used hours plus the
  time reserved for running sessions plus the new session would exceed 30 h, so a long full run may have to
  wait until another run on that account finishes.
* All runs: `docs/run_registry.tsv` (code name, slug, suite, job, push time, kind, status).

* `--cpu` builds a CPU-only job (for when the GPU quota is used up).
* DIAS and 6-Band notebooks were written for Colab: the runner turns `drive.mount` into a no-op and maps
  `/content/drive/My Drive/Classification` to `/kaggle/working` with `data/` linking the input files.
* Embedded notebooks are stripped of their old outputs (papermill writes new ones).

Workflow per suite: smoke run -> all 12 stages OK -> full run -> verify tables -> commit results.
