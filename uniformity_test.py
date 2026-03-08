#!/usr/bin/env python3
"""
Uniformity verification test for the BPD MCMC sampler.

For n=5,6,7,8, verifies that the MCMC sampler produces reduced BPDs
with the correct stationary distribution:
    P(w) = S_w(1^n) / sum_v S_v(1^n)

Usage:
    python3 uniformity_test.py          # Run all tests (n=5,6,7,8)
    python3 uniformity_test.py 5        # Run only n=5
    python3 uniformity_test.py 5 6      # Run n=5 and n=6
"""

import subprocess
import sys
import os
import re
import math
import tempfile
from itertools import permutations
from collections import Counter

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHUBERT_BIN = os.path.join(SCRIPT_DIR, "schubert")
MCMC_BIN = os.path.join(SCRIPT_DIR, "bpd_mcmc")


def compute_sw(perm):
    """Compute S_w(1^n) by calling the schubert binary."""
    perm_str = ",".join(str(x) for x in perm)
    result = subprocess.run(
        [SCHUBERT_BIN, perm_str, "--double", "--cotrans"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"schubert failed for {perm_str} (exit {result.returncode}):\n{result.stderr}")
    for line in result.stdout.split("\n"):
        m = re.search(r"Result:\s*S_w\(1\^\d+\)\s*=\s*(\S+)", line)
        if m:
            return int(m.group(1).split('.')[0])
    raise ValueError(f"Could not parse S_w for {perm_str}:\n{result.stdout}")


def parse_mathematica_perms(filename):
    """Parse Mathematica-format permutation list: {{p1,...},{q1,...},...}"""
    with open(filename, 'r') as f:
        text = f.read().strip()

    # Remove outer braces
    if text.startswith('{') and text.endswith('}'):
        text = text[1:-1]

    perms = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            j = text.index('}', i)
            inner = text[i+1:j]
            perm = tuple(int(x) for x in inner.split(','))
            perms.append(perm)
            i = j + 1
        else:
            i += 1

    return perms


def run_mcmc(n, B, work_dir):
    """Run bpd_mcmc batch mode and return list of sampled permutations."""
    burnin = max(100 * n**3, 50000)
    thin = max(10 * n**2, 500)

    cmd = [MCMC_BIN, f"batch:{n}:{B}",
           "--burnin", str(burnin),
           "--thin", str(thin),
           "--threads", "1"]

    print(f"  Running: {' '.join(cmd)}")
    print(f"  (burnin={burnin}, thin={thin})")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=600, cwd=work_dir)

    if result.returncode != 0:
        print(f"  MCMC failed: {result.stderr}")
        return None

    for line in result.stdout.split('\n'):
        if 'Completed' in line or 'samples/sec' in line:
            print(f"  {line.strip()}")

    # Output file may have a timestamp prefix (e.g., 20260304_144533_perms_...)
    import glob as globmod
    pattern = os.path.join(work_dir, f"*perms_n{n}_B{B}.txt")
    matches = globmod.glob(pattern)
    if not matches:
        print(f"  Output file not found: {pattern}")
        return None
    perms_file = max(matches, key=os.path.getmtime)

    perms = parse_mathematica_perms(perms_file)
    return perms


def chi_squared_test(observed, expected, min_expected=5.0):
    """
    Chi-squared goodness-of-fit test with bin merging for low-count cells.
    Returns (chi2_stat, p_value, df, n_merged).
    """
    items = sorted(zip(expected, observed), key=lambda x: x[0])

    merged_exp = []
    merged_obs = []
    cur_exp = 0.0
    cur_obs = 0

    for e, o in items:
        cur_exp += e
        cur_obs += o
        if cur_exp >= min_expected:
            merged_exp.append(cur_exp)
            merged_obs.append(cur_obs)
            cur_exp = 0.0
            cur_obs = 0

    if cur_exp > 0:
        if merged_exp:
            merged_exp[-1] += cur_exp
            merged_obs[-1] += cur_obs
        else:
            merged_exp.append(cur_exp)
            merged_obs.append(cur_obs)

    n_merged = len(items) - len(merged_exp)

    chi2 = sum((o - e)**2 / e for o, e in zip(merged_obs, merged_exp))
    df = len(merged_exp) - 1

    if HAS_SCIPY:
        p_value = 1.0 - scipy_stats.chi2.cdf(chi2, df)
    else:
        p_value = _chi2_p_value_approx(chi2, df)

    return chi2, p_value, df, n_merged


def _chi2_p_value_approx(chi2, df):
    """Approximate p-value using Wilson-Hilferty normal approximation."""
    if df <= 0:
        return 1.0
    z = ((chi2 / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return p


def total_variation_distance(emp_probs, theo_probs):
    """TV distance between two distributions (dicts: key -> probability)."""
    all_keys = set(emp_probs) | set(theo_probs)
    return 0.5 * sum(abs(emp_probs.get(k, 0) - theo_probs.get(k, 0))
                      for k in all_keys)


def test_uniformity(n, B):
    """Run uniformity test for S_n with B samples. Returns True if passed."""
    print(f"\n{'='*60}")
    print(f"Uniformity test: n={n}, B={B}")
    print(f"{'='*60}")

    # Step 1: Compute S_w for all permutations
    all_perms = list(permutations(range(1, n+1)))
    n_perms = len(all_perms)
    print(f"\nStep 1: Computing S_w(1^{n}) for all {n_perms} permutations...")

    sw_values = {}
    for i, perm in enumerate(all_perms):
        if (i+1) % 100 == 0 or i == 0 or i == n_perms - 1:
            print(f"\r  Computing: {i+1}/{n_perms}", end="", flush=True)
        sw_values[perm] = compute_sw(perm)
    print()

    total_sw = sum(sw_values.values())
    max_w = max(sw_values, key=sw_values.get)
    min_w = min(sw_values, key=sw_values.get)
    print(f"  Total sum S_w: {total_sw}")
    print(f"  Max S_w = {sw_values[max_w]} at {''.join(str(x) for x in max_w)}")
    print(f"  Min S_w = {sw_values[min_w]} at {''.join(str(x) for x in min_w)}")

    theo_probs = {w: sw / total_sw for w, sw in sw_values.items()}

    # Step 2: Run MCMC sampler
    print(f"\nStep 2: Running MCMC sampler (B={B})...")
    with tempfile.TemporaryDirectory() as work_dir:
        samples = run_mcmc(n, B, work_dir)

    if samples is None:
        print("FAIL: MCMC sampler failed")
        return False

    print(f"  Got {len(samples)} samples")

    # Step 3: Empirical frequencies
    print(f"\nStep 3: Computing empirical frequencies...")
    counts = Counter(samples)
    emp_probs = {w: c / len(samples) for w, c in counts.items()}

    n_observed = len(counts)
    print(f"  Observed {n_observed} distinct permutations out of {n_perms}")

    # Check for unexpected permutations
    unexpected = set(counts.keys()) - set(all_perms)
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected permutations in samples!")
        for w in list(unexpected)[:5]:
            print(f"    {w}")

    # Step 4: Statistical tests
    print(f"\nStep 4: Statistical tests...")

    observed = [counts.get(w, 0) for w in all_perms]
    expected = [theo_probs[w] * len(samples) for w in all_perms]

    # Chi-squared test
    chi2, p_value, df, n_merged = chi_squared_test(observed, expected)
    print(f"\n  Chi-squared test:")
    print(f"    chi2 = {chi2:.2f}, df = {df}, p-value = {p_value:.6f}")
    print(f"    Bins merged (expected < 5): {n_merged}")
    chi2_pass = p_value > 0.001
    print(f"    Result: {'PASS' if chi2_pass else 'FAIL'} (threshold: p > 0.001)")

    # Total variation distance
    tv = total_variation_distance(emp_probs, theo_probs)
    expected_tv = math.sqrt(n_perms / (2 * len(samples)))
    print(f"\n  Total variation distance:")
    print(f"    TV = {tv:.6f}")
    print(f"    Expected TV ~ {expected_tv:.6f} (sqrt(n!/(2B)))")
    tv_pass = tv < 3 * expected_tv
    print(f"    Result: {'PASS' if tv_pass else 'FAIL'} (threshold: TV < 3x expected)")

    # Top 10 deviations
    deviations = []
    for w in all_perms:
        obs_p = emp_probs.get(w, 0)
        exp_p = theo_probs[w]
        if exp_p > 0:
            rel_dev = abs(obs_p - exp_p) / exp_p
        else:
            rel_dev = float('inf') if obs_p > 0 else 0
        deviations.append((w, obs_p, exp_p, rel_dev,
                          counts.get(w, 0), exp_p * len(samples)))

    deviations.sort(key=lambda x: -x[3])
    print(f"\n  Top 10 permutations by relative deviation:")
    fmt = "    {:<20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(fmt.format("Perm", "Obs", "Exp", "Obs%", "Exp%", "RelDev"))
    for w, obs_p, exp_p, rel_dev, obs_c, exp_c in deviations[:10]:
        w_str = "(" + ",".join(str(x) for x in w) + ")"
        print(fmt.format(w_str, str(obs_c), f"{exp_c:.1f}",
                        f"{obs_p*100:.3f}%", f"{exp_p*100:.3f}%",
                        f"{rel_dev:.3f}"))

    # Max permutation focus
    max_obs = counts.get(max_w, 0)
    max_exp = theo_probs[max_w] * len(samples)
    max_dev = abs(max_obs - max_exp) / max_exp if max_exp > 0 else 0
    w_str = "(" + ",".join(str(x) for x in max_w) + ")"
    print(f"\n  Max permutation check:")
    print(f"    Permutation: {w_str}")
    print(f"    S_w = {sw_values[max_w]}, P(w) = {theo_probs[max_w]:.6f}")
    print(f"    Observed: {max_obs}, Expected: {max_exp:.1f}, "
          f"Rel.dev: {max_dev:.4f}")

    # Verdict
    overall_pass = chi2_pass and tv_pass
    print(f"\n  {'='*40}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  {'='*40}")

    return overall_pass


def main():
    for binary, name in [(SCHUBERT_BIN, "schubert"), (MCMC_BIN, "bpd_mcmc")]:
        if not os.path.exists(binary):
            print(f"Error: {name} binary not found at {binary}")
            print(f"Compile it first (see CLAUDE.md)")
            sys.exit(1)

    configs = [
        (5, 1_000_000),
        (6, 500_000),
        (7, 200_000),
        (8, 100_000),
    ]

    if len(sys.argv) > 1:
        try:
            requested_n = [int(x) for x in sys.argv[1:]]
        except ValueError:
            print("Error: arguments must be integers. Usage: uniformity_test.py [5] [6] [7] [8]")
            sys.exit(1)
        configs = [(n, B) for n, B in configs if n in requested_n]

    if not configs:
        print("No valid configs. Usage: uniformity_test.py [5] [6] [7] [8]")
        sys.exit(1)

    results = {}
    for n, B in configs:
        results[n] = test_uniformity(n, B)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for n, passed in results.items():
        print(f"  n={n}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
