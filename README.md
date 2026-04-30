# Computational Tools for Schubert Specializations and RBPD Sampling

Code accompanying the paper:

> D. Anderson, G. Panova, L. Petrov.
> *Computation and sampling for Schubert specializations.*

[https://arxiv.org/abs/2603.20104](https://arxiv.org/abs/2603.20104)

## Overview

This repository contains the Schubert evaluator, the main MCMC sampler for reduced bumpless pipe dreams, and the auxiliary code used for connectivity checks, CFTP diagnostics, and validation experiments.

### Core programs

- **`schubert.cpp`** — Computation of principal specializations $\mathfrak{S}_w(1^n)$ via three recurrence formulas (descent, cotransition, transition), with full and heuristic search for the maximum over $S_n$.

- **`bpd_mcmc.cpp`** — MCMC sampler for uniformly random reduced bumpless pipe dreams (RBPDs). Uses local 2x2 flips and rectangular droop/undroop moves. Outputs permuton data, height functions, and individual permutations.

### Validation and diagnostics

- **`uniformity_test.py`** — Small-$n$ validation of the MCMC stationary distribution using chi-squared and total variation tests. Use `--all-starts --csv=...` to regenerate the small-$n$ matrix used in SI Sec.~4.

- **`cayley_ball.py`** — BFS over the Cayley-distance ball around a permutation, evaluating $\mathfrak{S}_w(1^n)$ at every node. Backs the SI search-protocol claim that no counterexample exists within Cayley distance $4$ of the optimal layered permutation for $n = 14, 15, 16$.

- **`julia/run_n17_check.jl`** — Independent Julia cross-check at $n=17$ via Anderson's [SchubertPolynomials.jl](https://github.com/pseudoeffective/SchubertPolynomials.jl) (function `nschub`). Verifies the certified C++ values for $w^*$, $u^*$, and the layered comparator $w(1,2,4,10)$ digit-for-digit. See `julia/README.md`.

- **`bpd_connectivity_check.cpp`** — Verifies connectivity of the RBPD graph under reducedness-preserving flips (Conjecture 4.1 in the paper, verified for $n \le 8$).

- **`explore_allowed_moves.cpp`** — Enumerates allowed flip/droop moves in the RBPD state space and supports the connectivity proof experiments.

- **`connectivity_proof_check.cpp`** — Verifies components of the connectivity proof computationally.

### CFTP failure experiments

- **`bpd_cftp_sampler.cpp`** — Experimental backward-CFTP sampler used in the CFTP discussion and retained for comparison.

- **`cftp_monotonicity_check.cpp`** — Enumerates monotonicity violations under internal rejection for CFTP (Table 3 in the paper).

- **`cftp_bias_test.cpp`** — Measures bias in naive CFTP output distribution (Table 5 in the paper).

- **`cftp_coupling_diagnostic.cpp`** — CFTP diagnostic: runs backward coupling and checks for false coalescence (Table 4 in the paper).

- **`cftp_universality_check.cpp`** — Checks whether extremal-chain coalescence implies universal coalescence.

- **`benchmark_suite.py`** — Python harness for comparing performance of descent/cotransition/transition with double/exact arithmetic, using the bundled layered-permutation data in `mpp_layered_permutations.csv`.

## Artifacts and validation data

The repository ships the raw evidence backing every numerical claim in the paper.

### `artifacts/counterexamples_n17_20/`

Certification trail for the disproof of the Merzon--Smirnov conjecture at $n = 17$ and the analogous counterexamples at $n = 18, 19, 20$:

- `logs/` — raw exact-evaluation logs for the certified $n=17,18,19,20$ counterexamples and layered comparators, plus a small amount of historical audit material kept for provenance.
- `certificate.csv` — long-form table with one row per certified (permutation, implementation), including each certified log's SHA-256 hash.
- `certificate.tex` — auto-generated `tabular` snippet input by SI Sec.~F.
- `build_certificate_csv.py` — verifier that re-parses every log, asserts digit-for-digit agreement between cotransition and transition, aborts on mismatch, and re-emits the CSV/TeX.
- `README.md` — full submission recipe, including the Rivanna SLURM job for the $n = 19, 20$ runs.

### `artifacts/n100_production/`

Raw outputs of the $n = 100$ MCMC production run that produced the permuton, liquid-region, and height-fluctuation figures in SI Sec.~4. Run on Rivanna 2026-03-08 with seed 9200, geometric droops, burn-in $10^{10}$, thinning $5 \times 10^8$, $B = 10\,000$ samples across 40 parallel chains. Includes the rendered PNGs reproduced in the SI, all of `perms`, `perm_matrix`, `height_avg`, `height_dxdy`, `height_sum`, the burn-in trace, and the droop histogram.

### `mcmc_validation/`

Quantitative validation of the MCMC sampler (SI Sec.~4D):

- `data/` — small-$n$ uniformity (chi-squared, TV distance), within-fiber test at $n=4$, acceptance rates broken down by move type and starting state, integrated autocorrelation times, ESS, and a multi-start coupling trace at $n=60$. CSVs and the matching `.tex` snippets used in SI Sec.~4D.
- `img/multistart_n60.pdf` — rendered figure.
- `scripts/` — analysis pipeline: `analyze_mcmc_validation.py`, `compute_null_envelope.py`, `measure_acceptance_rates.py`, `plot_multistart_trace.py`, `within_fiber_test.py`, `run_validation_suite.sh`, `rivanna_validation_n100.slurm`.

## Compilation

All C++ programs require a C++17 compiler (`clang++` or `g++`). Each program is a single self-contained `.cpp` file. External dependencies are noted below.

### Python scripts

Python 3.7+ is required. Optional: `scipy` (for exact chi-squared p-values in `uniformity_test.py`). No other Python packages are needed.

### schubert.cpp

Requires [GMP](https://gmplib.org/) (GNU Multiple Precision Arithmetic Library) and [Boost.Multiprecision](https://www.boost.org/doc/libs/release/libs/multiprecision/) (header-only).

```bash
# macOS (Homebrew):
clang++ -O3 -std=c++17 -pthread -flto \
  -I/opt/homebrew/include -L/opt/homebrew/lib \
  schubert.cpp -o schubert -lgmp -lgmpxx

# Linux:
g++ -O3 -std=c++17 -march=native -pthread -flto \
  schubert.cpp -o schubert -lgmp -lgmpxx
```

Install dependencies: `brew install gmp boost` (macOS) or `apt install libgmp-dev libboost-dev` (Debian/Ubuntu).

### bpd_mcmc.cpp

Requires [libpng](http://www.libpng.org/). Optional: OpenMP for parallel batch mode.

```bash
# Single-threaded (macOS):
clang++ -O3 -std=c++17 -flto \
  -I/opt/homebrew/opt/libpng/include -L/opt/homebrew/opt/libpng/lib \
  bpd_mcmc.cpp -o bpd_mcmc -lpng

# Multi-threaded (macOS with OpenMP via Homebrew):
clang++ -O3 -std=c++17 -Xclang -fopenmp \
  -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
  -I/opt/homebrew/opt/libpng/include -L/opt/homebrew/opt/libpng/lib \
  bpd_mcmc.cpp -o bpd_mcmc -lpng

# Multi-threaded (Linux):
g++ -O3 -std=c++17 -march=native -flto -fopenmp \
  bpd_mcmc.cpp -o bpd_mcmc -lpng
```

Install dependencies: `brew install libpng libomp` (macOS) or `apt install libpng-dev` (Debian/Ubuntu).

### bpd_cftp_sampler.cpp

Requires libpng (same as `bpd_mcmc.cpp`). Optional: OpenMP for batch mode.

```bash
# Single-threaded (macOS):
clang++ -O3 -std=c++17 \
  -I/opt/homebrew/opt/libpng/include -L/opt/homebrew/opt/libpng/lib \
  bpd_cftp_sampler.cpp -o bpd_cftp_sampler -lpng

# With OpenMP (macOS):
clang++ -O3 -std=c++17 -Xclang -fopenmp \
  -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
  -I/opt/homebrew/opt/libpng/include -L/opt/homebrew/opt/libpng/lib \
  bpd_cftp_sampler.cpp -o bpd_cftp_sampler -lpng

# Linux:
g++ -O3 -std=c++17 -march=native -flto -fopenmp \
  bpd_cftp_sampler.cpp -o bpd_cftp_sampler -lpng
```

### Verification and diagnostic tools

These are self-contained (no external libraries beyond the C++17 standard library):

```bash
# macOS:
for f in cftp_monotonicity_check bpd_connectivity_check cftp_coupling_diagnostic \
         cftp_bias_test cftp_universality_check explore_allowed_moves \
         connectivity_proof_check; do
  clang++ -O3 -std=c++17 ${f}.cpp -o ${f}
done

# Linux:
for f in cftp_monotonicity_check bpd_connectivity_check cftp_coupling_diagnostic \
         cftp_bias_test cftp_universality_check explore_allowed_moves \
         connectivity_proof_check; do
  g++ -O3 -std=c++17 -march=native ${f}.cpp -o ${f}
done
```

## Usage examples

### Computing Schubert specializations

```bash
# Single permutation (best algorithm auto-selected):
./schubert 3,2,1
./schubert 1,3,8,7,6,5,4,2 --exact

# Specific algorithm:
./schubert 1,3,8,7,6,5,4,2 --cotrans --exact

# Full search for maximum over S_n:
./schubert max:12

# Heuristic beam search:
./schubert heuristic:16:10000

# Layered permutation test:
./schubert layered_test:15
```

### MCMC sampling of reduced BPDs

Note: `bpd_mcmc` appends a log of each run (command line and output files) to `bpd_mcmc_runs.log` in the working directory.

```bash
# Single sample with PNG output:
./bpd_mcmc 36

# Batch sampling (1000 samples, geometric droops, start from w0):
./bpd_mcmc batch:60:1000 --burnin 10G --thin 5M \
  --droop-dist geometric --start w0

# Production run with hybrid protocol:
./bpd_mcmc batch:100:10000 --burnin 10G --thin 500M \
  --droop-dist revlog --collect-droop-dist geometric \
  --start w0 --threads 40
```

### Experimental CFTP sampler

```bash
# Single sample
./bpd_cftp_sampler 10

# Batch export for diagnostics
./bpd_cftp_sampler batch:10:1000 --export
```

### Uniformity test

```bash
python3 uniformity_test.py
```

### CFTP diagnostics

```bash
# Monotonicity violations
./cftp_monotonicity_check cftp 6 200

# False-coalescence and ordering diagnostics
./cftp_coupling_diagnostic ordering 4 10000
./cftp_coupling_diagnostic cftp_ext 4 100000
./cftp_coupling_diagnostic cftp_int 4 100000

# Bias / universality checks
./cftp_bias_test 4 100000
./cftp_universality_check 4 100000
```

### Connectivity checks

```bash
./bpd_connectivity_check 8
./bpd_connectivity_check --fast --oeis 9
./connectivity_proof_check --all 7
```

## License

MIT. See [LICENSE](LICENSE) for the full text.
