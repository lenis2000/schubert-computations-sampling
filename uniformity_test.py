#!/usr/bin/env python3
"""
Uniformity verification test for the BPD MCMC sampler.

For n=5,6,7,8, verifies that the MCMC sampler produces reduced BPDs
with the correct stationary distribution:
    P(w) = S_w(1^n) / sum_v S_v(1^n)

Power note for the small_n_uniformity.csv sample budget:
For Cohen's w=0.05 (a 5% cell-wise bias), chi^2 GoF with
df in [100, 40000] reaches power >= 0.95 at alpha=0.01 with
N on the order of 50k (small df) to 300k (large df after
automatic bin merging). Defaults below meet or exceed that.
Sample sizes were chosen as:
    n=5: 200K, n=6: 200K, n=7: 100K, n=8: 50K.
These are conservative and produce p-values that are stable
across seeds.

Usage:
    python3 uniformity_test.py                    # Run all tests (n=5,6,7,8), start=w0
    python3 uniformity_test.py 5                  # Run only n=5, start=w0
    python3 uniformity_test.py 5 6                # Run n=5 and n=6, start=w0
    python3 uniformity_test.py --start-mode=w0    # Explicitly pick one start mode
    python3 uniformity_test.py --all-starts       # Loop over w0, id, rothe, random
    python3 uniformity_test.py --csv=<path>       # Emit CSV row per (n, start) to <path>
    python3 uniformity_test.py --seed=42          # Fixed RNG seed (reproducible)
"""

import subprocess
import sys
import os
import re
import math
import tempfile
import csv
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


# ---------------------------------------------------------------------------
# S_w(1^n) oracle
# ---------------------------------------------------------------------------
# Primary path: pure-Python cotransition-UP recurrence, which is fast for
# n <= 8 and has no external dependency.  The schubert binary is used only
# as a fallback / cross-check if present and executable.

_SW_CACHE = {}


def _compute_all_sw_python(n):
    """Pure-Python cotransition-UP recurrence.

    Matches the algorithm in the repo's schubert.cpp (find_cotrans_index +
    is_bruhat_cover_up).  S_w0(1^n) = 1; for pi != w0 the first staircase
    position a (smallest i with (i+1) + pi[i] <= n in 1-indexed form) is
    used as a pivot row, and S_pi(1^n) = sum over b > a of S_{pi t_{a,b}}(1^n)
    for Bruhat covers going UP.
    """
    if n in _SW_CACHE:
        return _SW_CACHE[n]

    def length(w):
        return sum(1 for i in range(len(w)) for j in range(i + 1, len(w)) if w[i] > w[j])

    def find_cotrans_index(pi):
        for i in range(n):
            if (i + 1) + pi[i] <= n:
                return i
        return -1

    def is_bruhat_cover_up(pi, a, b):
        pa = pi[a]
        pb = pi[b]
        if pa >= pb:
            return False
        for m in range(a + 1, b):
            pm = pi[m]
            if pa < pm < pb:
                return False
        return True

    all_perms = sorted(permutations(range(1, n + 1)), key=lambda w: -length(w))
    sw = {tuple(range(n, 0, -1)): 1}
    for pi in all_perms:
        if pi in sw:
            continue
        a = find_cotrans_index(pi)
        if a < 0:
            sw[pi] = 1
            continue
        total = 0
        for b in range(a + 1, n):
            if is_bruhat_cover_up(pi, a, b):
                sigma = list(pi)
                sigma[a], sigma[b] = sigma[b], sigma[a]
                total += sw[tuple(sigma)]
        sw[pi] = total

    _SW_CACHE[n] = sw
    return sw


def compute_sw(perm):
    """Compute S_w(1^n). Uses the pure-Python oracle (memoised per n)."""
    n = len(perm)
    sw = _compute_all_sw_python(n)
    return sw[tuple(perm)]


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


def _rothe_layered_at(n):
    """Deterministic layered-ish permutation for 'rothe' start at small n.

    Uses w(1,3,...) style layered block pattern. Falls back to w0 for n<=3.
    """
    if n <= 3:
        return list(range(n, 0, -1))
    # Simple layered partition: block sizes ~ roughly balanced, avoids identity
    # and w0 as special cases.  Concretely: split into blocks of size 2, 3, then
    # pad.  For n=5: (2,3) -> (2,1,5,4,3).  For n=8: (2,3,3) -> (2,1,5,4,3,8,7,6).
    if n == 4:
        blocks = [2, 2]
    elif n == 5:
        blocks = [2, 3]
    elif n == 6:
        blocks = [3, 3]
    elif n == 7:
        blocks = [3, 4]
    elif n == 8:
        blocks = [2, 3, 3]
    else:
        blocks = []
        remaining = n
        size = 2
        while remaining > 0:
            b = min(size, remaining)
            blocks.append(b)
            remaining -= b
            size = 3 if size == 2 else 2
    w = []
    pos = 1
    for b in blocks:
        for j in range(b - 1, -1, -1):
            w.append(pos + j)
        pos += b
    return w


def run_mcmc(n, B, work_dir, start_mode="w0", seed=None):
    """Run bpd_mcmc batch mode and return list of sampled permutations.

    start_mode in {'w0', 'id', 'identity', 'rothe', 'random'}.
    """
    burnin = max(100 * n**3, 50000)
    thin = max(10 * n**2, 500)

    cmd = [MCMC_BIN, f"batch:{n}:{B}",
           "--burnin", str(burnin),
           "--thin", str(thin),
           "--threads", "1"]

    # Map start mode to bpd_mcmc's --start syntax
    if start_mode == "w0":
        cmd += ["--start", "w0"]
    elif start_mode in ("id", "identity"):
        cmd += ["--start", "identity"]
    elif start_mode == "random":
        cmd += ["--start", "random"]
    elif start_mode == "rothe":
        perm = _rothe_layered_at(n)
        cmd += ["--start", "rothe:" + ",".join(str(x) for x in perm)]
    else:
        raise ValueError(f"Unknown start_mode {start_mode!r}")

    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    print(f"  Running: {' '.join(cmd)}")
    print(f"  (burnin={burnin}, thin={thin}, start={start_mode})")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=1200, cwd=work_dir)

    if result.returncode != 0:
        print(f"  MCMC failed: {result.stderr}")
        return None

    for line in result.stdout.split('\n'):
        if 'Completed' in line or 'samples/sec' in line:
            print(f"  {line.strip()}")

    # Output file has a timestamp prefix and a parameter suffix, e.g.
    # "20260304_144533_perms_n5_B10000_revlog_se_w0_s42_t500.txt".
    import glob as globmod
    pattern = os.path.join(work_dir, f"*perms_n{n}_B{B}*.txt")
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


def test_uniformity(n, B, start_mode="w0", seed=None):
    """Run uniformity test for S_n with B samples.

    Returns a dict with fields: {'n', 'start', 'num_samples', 'chi2',
    'p_value', 'df', 'n_merged', 'tv_distance', 'worst_cell_deviation',
    'passed'}, or None on sampler failure.
    """
    print(f"\n{'='*60}")
    print(f"Uniformity test: n={n}, B={B}, start={start_mode}")
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
        samples = run_mcmc(n, B, work_dir, start_mode=start_mode, seed=seed)

    if samples is None:
        print("FAIL: MCMC sampler failed")
        return None

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
    chi2_pass = p_value > 0.01
    print(f"    Result: {'PASS' if chi2_pass else 'FAIL'} (threshold: p > 0.01)")

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

    # Worst-cell relative deviation (across permutations with expected count >= 1)
    worst_cell_dev = 0.0
    for w in all_perms:
        exp_c = theo_probs[w] * len(samples)
        obs_c = counts.get(w, 0)
        if exp_c >= 1.0:
            rd = abs(obs_c - exp_c) / exp_c
            if rd > worst_cell_dev:
                worst_cell_dev = rd

    # Verdict
    overall_pass = chi2_pass and tv_pass
    print(f"\n  {'='*40}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  {'='*40}")

    return {
        "n": n,
        "start": start_mode,
        "num_samples": len(samples),
        "chi2": chi2,
        "df": df,
        "n_merged": n_merged,
        "p_value": p_value,
        "tv_distance": tv,
        "worst_cell_deviation": worst_cell_dev,
        "passed": overall_pass,
    }


CSV_FIELDS = [
    "n", "start", "num_samples", "chi2", "df", "n_merged",
    "p_value", "tv_distance", "worst_cell_deviation", "passed",
]


def write_csv(path, rows):
    """Write/overwrite the small_n_uniformity CSV deterministically."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    # Sort for byte-identical output across reruns
    sort_key = lambda r: (r["n"], r["start"])
    rows_sorted = sorted(rows, key=sort_key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows_sorted:
            row = {k: r[k] for k in CSV_FIELDS}
            # Normalise numeric formatting for reproducibility
            for k in ("chi2", "p_value", "tv_distance", "worst_cell_deviation"):
                row[k] = f"{r[k]:.10g}"
            row["passed"] = "PASS" if r["passed"] else "FAIL"
            writer.writerow(row)


def parse_cli(argv):
    """Parse CLI arguments.  Returns (ns, starts, seed, csv_path)."""
    ns = []
    starts = ["w0"]  # default
    seed = None
    csv_path = None
    all_starts = False

    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--start-mode="):
            val = a.split("=", 1)[1]
            if val not in ("w0", "id", "identity", "rothe", "random"):
                print(f"Error: invalid --start-mode={val}. Use w0|id|rothe|random")
                sys.exit(1)
            # Canonicalise id -> identity for output
            starts = [val if val != "id" else "identity"]
        elif a == "--all-starts":
            all_starts = True
        elif a.startswith("--seed="):
            seed = int(a.split("=", 1)[1])
        elif a.startswith("--csv="):
            csv_path = a.split("=", 1)[1]
        else:
            try:
                ns.append(int(a))
            except ValueError:
                print(f"Error: unknown argument {a!r}.  See module docstring.")
                sys.exit(1)
        i += 1

    if all_starts:
        starts = ["w0", "identity", "rothe", "random"]

    return ns, starts, seed, csv_path


def main():
    if not os.path.exists(MCMC_BIN):
        print(f"Error: bpd_mcmc binary not found at {MCMC_BIN}")
        print("Compile it first (see the top-level README.md)")
        sys.exit(1)

    # Default per-n sample budget (see header comment for power analysis).
    configs = [
        (5, 200_000),
        (6, 200_000),
        (7, 100_000),
        (8,  50_000),
    ]

    requested_ns, starts, seed, csv_path = parse_cli(sys.argv[1:])
    if requested_ns:
        configs = [(n, B) for n, B in configs if n in requested_ns]

    if not configs:
        print("No valid configs.  See module docstring for usage.")
        sys.exit(1)

    # Deterministic integer offset per start mode (avoid hash() which is
    # PYTHONHASHSEED-randomized and would break CSV reproducibility).
    START_OFFSET = {"w0": 0, "identity": 1, "id": 1, "rothe": 2, "random": 3}

    rows = []
    all_pass = True
    for start in starts:
        for n, B in configs:
            # Seed per (n, start) so each run is reproducible but distinct
            run_seed = None if seed is None else (seed + 1000 * n + 37 * START_OFFSET[start])
            rec = test_uniformity(n, B, start_mode=start, seed=run_seed)
            if rec is None:
                print(f"  FAIL: sampler failed at n={n}, start={start}")
                all_pass = False
                continue
            rows.append(rec)
            if not rec["passed"]:
                all_pass = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in rows:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  n={r['n']} start={r['start']:<9s}  p={r['p_value']:.4f}  TV={r['tv_distance']:.4f}  {tag}")
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    if csv_path is not None and rows:
        write_csv(csv_path, rows)
        print(f"  CSV written to {csv_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
