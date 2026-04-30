# n=100 production MCMC run

These are the raw outputs of the MCMC run that produced the permuton, liquid
region, and height-fluctuation figures in the SI (`\S{}4`, subsection
"MCMC parameters and simulation results").

## Exact command

Run on the University of Virginia Rivanna HPC cluster, 40-core node,
2026-03-08 (batch started 2026-03-07 21:55 UTC, collection started
2026-03-08 01:53 UTC). Total wall time approximately 3 hours.

```
./bpd_mcmc batch:100:10000 \
    --start w0 \
    --droop-dist geometric \
    --burnin 10G \
    --thin 500M \
    --seed 9200 \
    --threads 40 \
    --export \
    --export-tile-counts
```

Each of 40 parallel chains performs an independent burn-in of
$10^{10}$ steps and collects 250 samples with thinning
$5 \times 10^8$ steps, for a total of $B = 10\,000$ samples.
Independence of the 40 chains is verified by the multi-start diagnostics
in `../../mcmc_validation/data/multistart_n60_trace.csv` at smaller $n$.

## Files

| File | Description |
|---|---|
| `20260308_015320_perms_n100_B10000_geometric_se_w0_s9200_t500.00M.txt` | B=10,000 permutations in Mathematica list format |
| `20260308_015320_perm_matrix_n100_B10000_geometric_se_w0_s9200_t500.00M.txt` | Average permutation matrix $M_{ij}$ (eq. on average permutation matrix in SI) |
| `20260308_015320_height_avg_n100_B10000_geometric_se_w0_s9200_t500.00M.txt` | Average height function $\bar h(x,y)$ |
| `20260308_015320_height_sum_n100_B10000_geometric_se_w0_s9200_t500.00M.txt` | Sum of height functions (for reconstructing $\bar h$) |
| `20260308_015320_height_dxdy_n100_B10000_geometric_se_w0_s9200_t500.00M.txt` | Discrete mixed derivative $\Delta_x\Delta_y \bar h$ |
| `20260308_015320_burnin_trace_n100_geometric_se_w0_s9200_t500.00M.txt` | $\ell(w_t)$ trace during burn-in (equilibration diagnostic) |
| `20260308_015320_permuton_mcmc_n100_B10000_geometric_se_w0_s9200_t500.00M.png` | SI Fig. `permuton_n100` |
| `20260308_015320_overlay_n100_B10000_geometric_se_w0_s9200_t500.00M.png` | SI Fig. `overlay_n100` left |
| `20260308_015320_overlay_dxdyT_n100_B10000_geometric_se_w0_s9200_t500.00M.png` | Variant of overlay figure |
| `20260308_015320_overlay_permT_n100_B10000_geometric_se_w0_s9200_t500.00M.png` | Variant of overlay figure |
| `20260308_015320_height_dxdy_n100_B10000_geometric_se_w0_s9200_t500.00M.png` | Liquid region visualization |
| `20260308_015320_bpd_mcmc_n100_geometric_se_w0_s9200_t500.00M.png` | Single RBPD sample render |
| `20260308_015320_droop_hist_n100_B10000_geometric_se_w0_s9200_t500.00M.png` | Histogram of accepted droop rectangle sizes |

## Version

Run produced with `code/bpd_mcmc` at the commit recorded in the release tag
of the SI Code availability section. Rebuild with the flags in `CLAUDE.md`.
