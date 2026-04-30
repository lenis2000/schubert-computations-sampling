#!/usr/bin/env python3
"""
Compute a null envelope for Table S6 diagnostics.

For each (n, B) pair in small_n_uniformity.csv, draw K multinomial samples
of size B from the exact Upsilon_w weighted target distribution, compute
the total variation distance and the worst relative cell deviation, and
report the mean and 95th percentile over K replicates.

This answers the reviewer objection that raw TV ~0.25 or worst_cell ~4 at
n=8 look large: they are expected under multinomial sampling noise at the
given B and should be compared against this envelope rather than against 0.

Usage:
    python3 compute_null_envelope.py \
        --csv-in  PNAS/mcmc_validation/data/small_n_uniformity.csv \
        --csv-out PNAS/mcmc_validation/data/small_n_null_envelope.csv \
        --replicates 2000 \
        --seed 0

Writes one row per (n, B) combination: exact weights, envelope mean/95th
percentile for both statistics.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from itertools import permutations

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code")
sys.path.insert(0, CODE_DIR)

from uniformity_test import _compute_all_sw_python  # noqa: E402


def exact_probabilities(n: int):
    """Return (perms, probs) with probs proportional to Upsilon_w."""
    sw = _compute_all_sw_python(n)
    perms = list(sw.keys())
    weights = np.array([sw[p] for p in perms], dtype=np.float64)
    total = weights.sum()
    if total == 0:
        raise RuntimeError(f"All Upsilon_w zero at n={n}, unexpected")
    return perms, weights / total


def null_envelope(n: int, B: int, K: int, rng: np.random.Generator):
    """Draw K multinomial samples of size B, return TV and worst-rel arrays.

    worst-rel matches uniformity_test.py: max over cells with expected
    count >= 1 of |observed - expected| / expected.
    """
    _, p = exact_probabilities(n)
    counts = rng.multinomial(B, p, size=K)
    emp = counts.astype(np.float64) / B
    diff = emp - p[np.newaxis, :]
    tv = 0.5 * np.abs(diff).sum(axis=1)
    mask = (p * B) >= 1.0  # cells with expected count >= 1
    rel = np.zeros_like(diff)
    rel[:, mask] = np.abs(diff[:, mask]) / p[np.newaxis, mask]
    worst = rel.max(axis=1)
    return tv, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-in", required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Read the (n, num_samples) pairs from the existing table.
    pairs = []
    seen = set()
    with open(args.csv_in) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            B = int(row["num_samples"])
            key = (n, B)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    pairs.sort()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    fieldnames = [
        "n",
        "num_samples",
        "num_perms",
        "replicates",
        "tv_mean",
        "tv_p95",
        "worst_rel_mean",
        "worst_rel_p95",
    ]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for n, B in pairs:
            tv, worst = null_envelope(n, B, args.replicates, rng)
            num_perms = len(list(permutations(range(1, n + 1))))
            row = {
                "n": n,
                "num_samples": B,
                "num_perms": num_perms,
                "replicates": args.replicates,
                "tv_mean": f"{tv.mean():.6g}",
                "tv_p95": f"{np.percentile(tv, 95):.6g}",
                "worst_rel_mean": f"{worst.mean():.6g}",
                "worst_rel_p95": f"{np.percentile(worst, 95):.6g}",
            }
            w.writerow(row)
            print(
                f"n={n} B={B}  TV: mean={row['tv_mean']} p95={row['tv_p95']}  "
                f"worst_rel: mean={row['worst_rel_mean']} p95={row['worst_rel_p95']}"
            )


if __name__ == "__main__":
    main()
