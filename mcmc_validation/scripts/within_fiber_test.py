#!/usr/bin/env python3
"""
Within-fiber uniformity test at small n.

The small-n chi-squared test in uniformity_test.py only checks the
projection to the boundary permutation (did w appear with frequency
proportional to Upsilon_w). A within-fiber bias -- preferring one RBPD
over another while still hitting the correct marginal for w -- would
slip through that test. This script verifies the FULL-STATE-SPACE
uniformity directly: at small n (n=4 has 42 RBPDs total), we hash each
sampled grid, count frequencies, and compare to uniform (1/N_total) via
chi-squared and total-variation distance.

Produce samples first with:

  ./bpd_mcmc batch:4:<B> --thin <T> --seed <S> --export-grids \
      --no-png --no-tikz --no-height

Then run this script on the resulting grids file. At n=4 the state space
has 42 RBPDs (= sum_w Upsilon_w = number of 4x4 alternating sign
matrices), and B = 1,000,000 with thin = 1,000 is enough to detect
sub-percent bias with power > 0.99 at alpha = 0.01.

Usage:
    python3 mcmc_validation/scripts/within_fiber_test.py \
        --grids 20260422_111211_grids_n4_B1000000_geometric_se_identity_s42_t1.00K.txt \
        --n 4 \
        [--csv-out mcmc_validation/data/within_fiber_n4.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter


# Expected total RBPD count at n, from sum_{w in S_n} Upsilon_w.
# Computed by uniformity_test.py _compute_all_sw_python.
# (Note: this is NOT the ASM count; ASMs at n=4 number 42, but one of
# those is NOT a reduced BPD for any S_4 permutation.)
RBPD_COUNT = {3: 7, 4: 41, 5: 393, 6: 6080, 7: 150371, 8: 5903710}


def read_grids(path: str):
    """Yield tuples (line_number, canonical grid string) for each sample.

    Transparently handles .gz files.
    """
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for ln, line in enumerate(f):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield ln, s


def chi_squared(observed: dict, expected: float):
    """Pearson chi^2 statistic plus degrees of freedom (number_cells - 1)."""
    chi2 = 0.0
    for obs in observed.values():
        chi2 += (obs - expected) ** 2 / expected
    df = len(observed) - 1
    return chi2, df


def tv_distance(observed: dict, B: int, num_cells: int):
    """Total variation to uniform 1/num_cells."""
    p_uniform = 1.0 / num_cells
    tv = 0.0
    for obs in observed.values():
        tv += abs(obs / B - p_uniform)
    # Cells with zero count: contribute (0 - p_uniform) = p_uniform per missed cell.
    tv += (num_cells - len(observed)) * p_uniform
    return 0.5 * tv


def chi2_p_value(chi2: float, df: int) -> float:
    """Survival probability; tries scipy if available, else Wilson-Hilferty."""
    try:
        from scipy.stats import chi2 as scipy_chi2
        return float(scipy_chi2.sf(chi2, df))
    except Exception:
        # Wilson-Hilferty approximation: ((chi2/df)^(1/3) - (1 - 2/(9df))) /
        # sqrt(2/(9df)) is ~Normal(0,1).
        import math
        a = 2.0 / (9 * df)
        z = ((chi2 / df) ** (1.0 / 3) - (1 - a)) / math.sqrt(a)
        return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True, help="grids file from bpd_mcmc --export-grids")
    ap.add_argument("--n", type=int, required=True, help="n; must match the grids file")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Read grids and tally
    counts = Counter()
    total = 0
    for _, g in read_grids(args.grids):
        if len(g) != args.n * args.n:
            print(f"error: grid line length {len(g)} != n*n = {args.n*args.n}", file=sys.stderr)
            return 1
        counts[g] += 1
        total += 1

    expected_total_cells = RBPD_COUNT.get(args.n)
    if expected_total_cells is None:
        print(f"error: unknown RBPD count for n={args.n}", file=sys.stderr)
        return 1

    distinct = len(counts)
    missing = expected_total_cells - distinct

    expected = total / expected_total_cells  # per-cell expected count under uniform
    # Fill in zero counts for unobserved cells so chi-squared and TV include them.
    all_counts = dict(counts)
    if missing > 0:
        # Use synthetic keys; all that matters for chi^2 is how many zero cells exist.
        for i in range(missing):
            all_counts[f"_missing_{i}"] = 0

    chi2, df = chi_squared(all_counts, expected)
    p = chi2_p_value(chi2, df)
    tv = tv_distance(counts, total, expected_total_cells)

    # Worst relative cell deviation over observed cells with expected >= 1.
    worst_rel = 0.0
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    if expected >= 1:
        for obs in counts.values():
            rd = abs(obs - expected) / expected
            if rd > worst_rel:
                worst_rel = rd

    print(f"n                      = {args.n}")
    print(f"samples (B)            = {total}")
    print(f"expected cells (RBPDs) = {expected_total_cells}")
    print(f"observed distinct      = {distinct}")
    print(f"missing cells          = {missing}")
    print(f"expected count / cell  = {expected:.2f}")
    print(f"min / max observed     = {min_count} / {max_count}")
    print(f"chi^2 ({df} dof)         = {chi2:.3f}")
    print(f"p-value                = {p:.4f}")
    print(f"TV distance            = {tv:.6f}")
    print(f"worst relative dev     = {worst_rel:.4f}")
    passed = p > 0.01 and missing == 0
    print(f"PASS                   = {passed}")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "n", "total_samples", "expected_cells", "observed_distinct",
                "missing_cells", "expected_per_cell", "min_count", "max_count",
                "chi2", "df", "p_value", "tv_distance", "worst_rel_dev", "passed",
            ])
            w.writerow([
                args.n, total, expected_total_cells, distinct, missing,
                f"{expected:.4f}", min_count, max_count,
                f"{chi2:.4f}", df, f"{p:.6f}", f"{tv:.6f}",
                f"{worst_rel:.6f}", "PASS" if passed else "FAIL",
            ])
        print(f"wrote {args.csv_out}")

    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
