# Classification Uncertainty Quantification Analysis

This repository contains experiments and analysis pipelines for classification uncertainty across multiple workflows:

- Baseline model training and uncertainty comparison
- CREDIT uncertainty workflow
- DAPM full pipeline training and uncertainty
- Ensemble and MultiCP uncertainty methods
- SACP comparison outputs

## Repository Structure

- `Baseline/` - baseline model training and uncertainty notebooks, metrics, and outputs
- `Credit/` - CREDIT training notebook and uncertainty results
- `DAPM/` - DAPM training, uncertainty notebooks, and result summaries
- `Data/` - source datasets used by experiments
- `Ensemble/` - ensemble training and CreDE uncertainty outputs
- `MultiCP/` - multi-head training and uncertainty notebooks
- `SACP/` - SACP comparison metrics and run configurations

## Notes

- Large model artifact directories named `models/` are intentionally excluded from version control.
- Experiment outputs and summaries are provided in each module's `results/` directory.

## Quick Start

1. Create and activate a Python virtual environment.
2. Install notebook dependencies required by your target workflow.
3. Open the corresponding notebook in each module and run cells in sequence.

## License

This project is distributed under the terms in `LICENSE`.