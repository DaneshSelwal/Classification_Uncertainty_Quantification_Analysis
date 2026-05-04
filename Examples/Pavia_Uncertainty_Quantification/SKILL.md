---
name: classification-uncertainty-analysis
description: Use when running, updating, or validating classification uncertainty experiments and result summaries in this repository.
---

# Classification Uncertainty Analysis Skill

## Purpose

Use this workflow when you need consistent changes across notebooks, results, and reporting assets in this repository.

## Preferred Workflow

1. Identify the target module: Baseline, Credit, DAPM, Ensemble, MultiCP, or SACP.
2. Update the corresponding notebook and configuration inputs.
3. Re-run the uncertainty notebook for the selected module.
4. Validate generated result files in the module `results/` folder.
5. Keep documentation updates in sync in the root `README.md`.

## Repository Conventions

- Keep module-level outputs inside each module's `results/` directory.
- Do not track large trained artifacts under any `models/` directory.
- Preserve naming consistency for summary CSV and run configuration files.

## Validation Checklist

- Notebook runs complete without errors.
- Summary CSV files are regenerated as expected.
- Any changes to experiment settings are reflected in run configuration JSON files.
- Root documentation remains accurate after workflow updates.
