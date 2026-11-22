# version2.0: fixes applied to the 12 pipeline notebooks of every dataset suite

All fixes were reviewed on Indian Pines and approved by the project owner on 2026-11-22. They are
generic: no value is tuned for one dataset. Each changed code line is marked `# FIX <ID>` in the
notebooks.

| ID | Notebook(s) | Problem in v1 | Fix in v2.0 |
|----|-------------|---------------|-------------|
| C1 | all | Each notebook / model drew its own random split; AlexNet used a different "legacy" split, so it was calibrated on its own training pixels (~77 % overlap). | One shared split (`make_shared_split`) built from the label map, identical in every notebook and for every model. A runtime check (`check_split_alignment`) stops the run if a notebook's patch order does not match the split. |
| C2 | ensemble training | Member 1 of the ensemble overwrote the baseline `{model}_best/_final.keras`, so later notebooks silently used an ensemble member as "baseline". | Baseline files are never overwritten; visualisation reads the ensemble member directly. |
| C3 | baseline / ensemble / CREDIT / multi-head training | Checkpoints were selected on the **test** split. | Checkpoint selection and early stopping use the validation split only. |
| C4 | all | Random pixel split: 100 % of evaluation pixels had training pixels inside their 9x9 patch (spatial leakage). DAPM metrics were computed on all labelled pixels (mostly training). | Spatial block split: 10x10 blocks assigned train / val / cal / eval (stratified, rare classes first), and train/val pixels within `patch_size // 2` of a held-out pixel are dropped (0 % leakage). DAPM metrics on held-out pixels only. |
| C5 | every GFNet | `tf.signal.fft2d` / `rfft2d` act on the two innermost axes, i.e. (token, channel) instead of the spatial token grid. | Channels moved first, FFT over the token grid, filter weights shaped per channel, transpose back. |
| H6 | split CP family | Quantile level capped at 1, so for small calibration sets q_hat was the max score instead of +inf (coverage not guaranteed). | `conformal_qhat`: the ceil((n+1)(1-alpha))-th smallest score, +inf when that rank exceeds n. |
| H7 | RAPS | Degenerate (set size always 1). | Randomised RAPS score `above + u * p_y + lambda * max(rank + 1 - k_reg, 0)` with the same u for all candidate labels. |
| H8 | RC3P | Rank-selection criterion computed on the evaluation set. | Computed on calibration data. |
| H9 | baseline CP | Spatial (SACP) code was dead / never called and wrong. | Replaced by the shared correct implementation. |
| H10 | SACP, spatial Multi-CP | Smoothing formula wrong (`(1-lambda) * ori + neigh`); calibration and map used different procedures (cal / eval smoothed in separate sparse volumes). | `(1-lambda) * p + lambda * mean(neighbours)`; the full scene is smoothed once and calibration, evaluation and the map read from that one map. Vectorised (identical numbers to the loop, max diff 1e-16). |
| H11 | Multi-CP, spatial Multi-CP | No Bonferroni correction for the intersection of K heads; no finite-sample (n+1) correction; set size was the per-head average, not the joint set; misaligned coverage indices. | Each head at alpha/K with `conformal_qhat`; joint (intersection) set used for both coverage and size. |
| H12 | Multi-CP maps | Class id 7 hard-coded as "unlabelled" and painted uncertain; full-image coverage computed on all pixels. | No hard-coded class; coverage on evaluation pixels only (others get label -1 and are ignored). |
| H13 | CREDIT | AlexNet split silently overridden (20 % train, seed 42). | Override removed; shared split. |
| H14 | CREDIT, CreDE | Normalised `p_min` used as a lower bound (it is not); TU = AU + EU added an entropy (nats) to a probability width. | CreDE: TU = H(mean)/ln C, AU = mean member entropy / ln C, EU = TU - AU (mutual information). CREDIT student has four heads (p_star, p_low, delta_p, au) trained on the same ensemble quantities; bounds = [p_low, p_low + delta_p]. All three values lie in [0, 1] and AU + EU = TU. |
| H15 | DAPM | Welch independent-samples t-test on paired, dependent diffusion draws (drifts to "certain" as samples grow). | Flip rate: fraction of draws whose argmax differs from the mean-probability argmax; uncertain when > threshold. |
| M16 | baseline training | Reported metrics came from the final-epoch model, but later notebooks load the best checkpoint. | Metrics computed after reloading the best checkpoint. |
| M18 | ensemble training | A reused (skipped) member wrote a dummy history of zeros. | NaN history and NaN training time. |
| M19 | ClCP | Same calibration data used to fit the class clusters and to compute the quantiles. | Half of calibration fits the clusters, the other half calibrates. |
| M20 | Multi-CP | External repo `yamtawa/Multi-CP` cloned at an unpinned HEAD. | Pinned to commit `fc252502d0bd2e2311f872672e3b70f7f64bcb74`. |
| K1 | Excel writers | `xlsxwriter` crashed on inf (now valid for q_hat). | `nan_inf_to_errors` option. |
| K2 | all with confusion matrices / reports | With the spatial split a very small class can have no held-out pixel; `confusion_matrix` / `classification_report` then returned fewer rows than class names and the plot crashed. | `labels=np.arange(num_classes)` everywhere, so every class keeps its row (empty classes show zeros). |
| K3 | spatial Multi-CP | Final table check hard-coded 4 SACP window sizes (24 rows). | Row count derived from `SACP_WINDOW_SIZES`. |

Not changed (reported, owner aware): duplicate notebooks in each suite root (`Model_training.ipynb`
etc.) are v1 copies and are not part of the v2.0 pipeline; the extra model folders of 6-Band / DIAS
(dofa, georsclip, mambahsi, spatialgcn, trials) are out of scope of the 12-notebook pipeline.

## Verification
* Local (numpy only, seconds): split CP coverage 0.949 at alpha = 0.05; RAPS 0.950; SACP 0.951; Multi-CP
  joint coverage 0.938 at alpha = 0.10 (K = 3); CREDIT AU + EU = TU, lower <= upper; vectorised
  smoothing equals the loop (1e-16).
* Kaggle GPU smoke runs (short epochs) and full runs: see `docs/KAGGLE_RUNS.md`.
