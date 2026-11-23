# temp-results-v2.0: computed results of the v2.0 runs so far

This branch = `version2.0` (code) + results. It is updated by agy (see `agy.md`) until all suites are done.

| Suite | Run | Kind | Result | Files |
|---|---|---|---|---|
| Indian Pines | CLA22_11_2026_21_07 | smoke (stages 0,6,8-11 after K2/K3) | all OK | `docs/smoke_runs/CLA22_11_2026_21_07/` |
| Indian Pines | CLA23_11_2026_01_04 | full | 12/12 OK, 216 min | `examples/indian_pines_uncertainty_quantification/logs/v2.0_CLA23_11_2026_01_04/` (executed notebooks + tables) |
| Pavia University | CLA23_11_2026_01_05 | smoke | 12/12 OK, 143 min | `docs/smoke_runs/CLA23_11_2026_01_05/` |
| Achla multi-temporal | CLA23_11_2026_01_06 | smoke | 12/12 OK, 256 min | `docs/smoke_runs/CLA23_11_2026_01_06/` |

Indian Pines full run, key numbers (target coverage 0.95, spatial block split):
* Split CP coverage AlexNet / GFNet / ViT: 0.947 / 0.957 / 0.955; RAPS 0.95-0.98; Clustered CP 0.96.
* Class-conditional CP / RC3P: AlexNet 0.885 / 0.892 (below target), GFNet / ViT 0.94-0.96.
* SACP (windows 3-9): 0.91-0.98. Multi-CP (Bonferroni, joint set): 0.98-1.00.
* CREDIT credal coverage 0.47 / 0.73 / 0.64 (no coverage guarantee). DAPM accuracy 0.38 / 0.72 / 0.64.
* 4 of 16 classes have no evaluation pixel under the spatial split (per-class coverage NaN).
