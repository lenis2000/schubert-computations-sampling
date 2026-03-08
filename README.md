# Computational Tools for Schubert Specializations and RBPD Sampling

Code accompanying the paper:

> D. Anderson, G. Panova, L. Petrov.
> *Computation and sampling for Schubert specializations.*

## Overview

This repository contains the Schubert evaluator, the main MCMC sampler for reduced bumpless pipe dreams, and the auxiliary code used for connectivity checks, CFTP diagnostics, and validation experiments.

### Core programs

- **`schubert.cpp`** — Computation of principal specializations $\mathfrak{S}_w(1^n)$ via three recurrence formulas (descent, cotransition, transition), with full and heuristic search for the maximum over $S_n$.

- **`bpd_mcmc.cpp`** — MCMC sampler for uniformly random reduced bumpless pipe dreams (RBPDs). Uses local 2x2 flips and rectangular droop/undroop moves. Outputs permuton data, height functions, and individual permutations.

- **`bpd_cftp_sampler.cpp`** — Experimental backward-CFTP sampler retained for the paper's negative-result and comparison experiments. This is diagnostic code, not the recommended production sampler; for actual sampling runs use `bpd_mcmc.cpp`.

### Validation and diagnostics

- **`uniformity_test.py`** — Small-$n$ validation of the MCMC stationary distribution using chi-squared and total variation tests.

- **`bpd_connectivity_check.cpp`** — Verifies connectivity of the RBPD graph under reducedness-preserving flips (Conjecture 4.1 in the paper, verified for $n \le 8$).

- **`explore_allowed_moves.cpp`** — Enumerates allowed flip/droop moves in the RBPD state space and supports the connectivity proof experiments.

- **`verify_connectivity_proof.cpp`** — Verifies components of the connectivity proof computationally.

### CFTP failure experiments

- **`bpd_cftp_sampler.cpp`** — Experimental backward-CFTP sampler used in the CFTP discussion and retained for comparison.

- **`monotonicity_check.cpp`** — Enumerates monotonicity violations under internal rejection for CFTP (Table 3 in the paper).

- **`cftp_bias_test.cpp`** — Measures bias in naive CFTP output distribution (Table 5 in the paper).

- **`cftp_diagnostic.cpp`** — CFTP diagnostic: runs backward coupling and checks for false coalescence (Table 4 in the paper).

- **`cftp_universality_check.cpp`** — Checks whether extremal-chain coalescence implies universal coalescence.

- **`benchmark_suite.py`** — Python harness for comparing performance of descent/cotransition/transition with double/exact arithmetic.

## Compilation

All C++ programs require C++17. External dependencies are noted below.

### schubert.cpp

Requires GMP (GNU Multiple Precision Arithmetic Library).

```bash
# macOS (Apple Silicon):
clang++ -O3 -mcpu=apple-m2 -flto -std=c++17 -pthread \
  -I/opt/homebrew/include -L/opt/homebrew/lib \
  schubert.cpp -o schubert -lgmp -lgmpxx

# Linux:
g++ -O3 -march=native -flto -std=c++17 -pthread \
  schubert.cpp -o schubert -lgmp -lgmpxx
```

### bpd_mcmc.cpp

Requires libpng. Optional: OpenMP for parallel batch mode.

```bash
# Single-threaded (macOS):
clang++ -O3 -std=c++17 -mcpu=apple-m2 -flto \
  -I/opt/homebrew/opt/libpng/include/libpng16 -L/opt/homebrew/opt/libpng/lib \
  bpd_mcmc.cpp -o bpd_mcmc -lpng16

# Multi-threaded (macOS with OpenMP):
clang++ -O3 -std=c++17 -mcpu=apple-m2 -Xclang -fopenmp \
  -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp \
  -I/opt/homebrew/opt/libpng/include/libpng16 -L/opt/homebrew/opt/libpng/lib \
  bpd_mcmc.cpp -o bpd_mcmc -lpng16

# Multi-threaded (Linux):
g++ -O3 -std=c++17 -march=native -flto -fopenmp \
  bpd_mcmc.cpp -o bpd_mcmc -lpng
```

### bpd_cftp_sampler.cpp

Requires libpng. Optional: OpenMP for batch mode.

```bash
# Single-threaded:
clang++ -O3 -std=c++17 bpd_cftp_sampler.cpp -o bpd_cftp_sampler -lpng

# With OpenMP:
clang++ -O3 -std=c++17 -Xclang -fopenmp \
  -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp \
  bpd_cftp_sampler.cpp -o bpd_cftp_sampler -lpng
```

### Verification tools

These are self-contained (no external libraries):

```bash
# Example:
clang++ -O3 -std=c++17 monotonicity_check.cpp -o monotonicity_check
clang++ -O3 -std=c++17 bpd_connectivity_check.cpp -o bpd_connectivity_check
clang++ -O3 -std=c++17 cftp_diagnostic.cpp -o cftp_diagnostic
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
./monotonicity_check cftp 6 200

# False-coalescence and ordering diagnostics
./cftp_diagnostic ordering 4 10000
./cftp_diagnostic cftp_ext 4 100000
./cftp_diagnostic cftp_int 4 100000

# Bias / universality checks
./cftp_bias_test 4 100000
./cftp_universality_check 4 100000
```

### Connectivity checks

```bash
./bpd_connectivity_check 8
./bpd_connectivity_check --fast --oeis 9
./verify_connectivity_proof --all 7
```

## Suggested Paper Paragraph

If the public repository is meant to match the current file layout, a more accurate code-availability paragraph is:

```tex
\subsection{Code availability}
\label{sec:code}

\textcolor{blue}{The public code repository accompanying this paper is available at
\textit{[repository URL]}. It includes \texttt{schubert.cpp} (three formulas for
computing principal specializations, together with exact and heuristic max search),
\texttt{bpd\_mcmc.cpp} (the MCMC sampler for reduced bumpless pipe dreams),
\texttt{bpd\_cftp\_sampler.cpp} (experimental backward-CFTP code retained for the
paper's negative-result and comparison experiments), validation and diagnostic tools
for the CFTP failure results (Tables~3--5), and computational tools for RBPD
connectivity.}
```

## License

MIT
