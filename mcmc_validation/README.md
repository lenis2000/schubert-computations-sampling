# MCMC validation bundle

This directory contains the quantitative validation material cited in SI Sec. 4D.

- `data/` contains the committed CSVs and `.tex` snippets used in the SI:
  small-`n` uniformity, within-fiber validation at `n = 4`, acceptance-rate
  summaries, autocorrelation / ESS tables, and the multi-start trace at `n = 60`.
- `img/` contains the committed multi-start figure rebuilt from the CSV.
- `scripts/` contains the regeneration pipeline:
  `uniformity_test.py` lives at the repo root, while the directory here holds
  `within_fiber_test.py`, `measure_acceptance_rates.py`,
  `analyze_mcmc_validation.py`, `compute_null_envelope.py`,
  `plot_multistart_trace.py`, `run_validation_suite.sh`, and the Rivanna
  `n = 100` template.

All commands are intended to be run from the repository root.
