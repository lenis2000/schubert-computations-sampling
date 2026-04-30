#!/usr/bin/env python3
"""
Compute Schubert specializations for all permutations in a Cayley ball
of given radius around the max layered permutation for a given n.

Cayley metric: d(u,v) = ℓ(u⁻¹v) where ℓ = Coxeter length (min adjacent transpositions).
Ball of radius r: all permutations reachable by at most r adjacent transpositions.

Usage:
    python3 cayley_ball.py [--n N] [--radius R] [--method METHOD]

    N defaults to 18, R defaults to 1, METHOD defaults to "best"
"""

import subprocess
import sys
import argparse
import signal
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

_children = []  # track spawned subprocesses

def cleanup_children(signum=None, frame=None):
    """Kill all child processes on exit."""
    for p in _children:
        try:
            p.kill()
        except Exception:
            pass
    if signum is not None:
        sys.exit(1)

signal.signal(signal.SIGINT, cleanup_children)
signal.signal(signal.SIGTERM, cleanup_children)
import atexit
atexit.register(cleanup_children)

# MPP3 optimal layered compositions (from schubert.cpp MPPY_DATA, n=1..25)
OPTIMAL_LAYERED = {
    1: [1], 2: [1, 1], 3: [1, 2], 4: [1, 3], 5: [1, 1, 3],
    6: [1, 1, 4], 7: [1, 2, 4], 8: [1, 2, 5], 9: [1, 2, 6],
    10: [1, 3, 6], 11: [1, 3, 7], 12: [1, 3, 8],
    13: [1, 1, 3, 8], 14: [1, 1, 4, 8], 15: [1, 1, 4, 9],
    16: [1, 1, 4, 10], 17: [1, 2, 4, 10], 18: [1, 2, 4, 11],
    19: [1, 2, 5, 11], 20: [1, 2, 5, 12], 21: [1, 2, 5, 13],
    22: [1, 2, 6, 13], 23: [1, 2, 6, 14], 24: [1, 3, 6, 14],
    25: [1, 3, 6, 15],
}

def build_layered(composition):
    """Build layered permutation from composition (1-indexed)."""
    perm = []
    pos = 0
    for block_size in composition:
        # Each block is a decreasing sequence
        block = list(range(pos + block_size, pos, -1))
        perm.extend(block)
        pos += block_size
    return tuple(perm)

def perm_to_str(perm):
    return ','.join(str(x) for x in perm)

def apply_swap(perm, i):
    """Apply adjacent transposition s_i (swap positions i and i+1). 0-indexed."""
    lst = list(perm)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return tuple(lst)

def cayley_ball(center, radius):
    """
    BFS to find all permutations within Cayley distance `radius` of `center`.
    Returns dict: perm -> distance.
    """
    n = len(center)
    visited = {center: 0}
    queue = deque([center])

    while queue:
        w = queue.popleft()
        d = visited[w]
        if d >= radius:
            continue
        for i in range(n - 1):
            neighbor = apply_swap(w, i)
            if neighbor not in visited:
                visited[neighbor] = d + 1
                queue.append(neighbor)

    return visited

def compute_schubert(perm, binary="./schubert", exact=False, threads=None):
    """Compute S_w(1^n) using the schubert binary (cotrans)."""
    prec = "--exact" if exact else "--double"
    cmd = [binary, perm_to_str(perm), prec, "--cotrans"]
    if threads is not None:
        cmd.append(f"--threads={threads}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _children.append(proc)
    try:
        stdout, stderr = proc.communicate(timeout=1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        print(f"  WARNING: Timeout for {perm_to_str(perm)}")
        return None
    finally:
        if proc in _children:
            _children.remove(proc)

    if proc.returncode != 0:
        print(f"  WARNING: Process exited with code {proc.returncode} for {perm_to_str(perm)}")
        return None

    # Parse the result line (split by both \n and \r to handle progress output)
    output = stdout + stderr
    for line in output.replace('\r', '\n').split('\n'):
        if line.startswith('Result:'):
            parts = line.split('=')
            if len(parts) >= 2:
                val_str = parts[-1].strip()
                return float(val_str)

    print(f"  WARNING: Could not parse output for {perm_to_str(perm)}")
    print(f"  Output (last 300 chars): ...{stdout[-300:]}")
    return None

def inversions(perm):
    """Count inversions (= Coxeter length)."""
    n = len(perm)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description='Cayley ball Schubert specializations')
    parser.add_argument('--n', type=int, default=18, help='Size of symmetric group (default: 18)')
    parser.add_argument('--radius', type=int, default=1, help='Cayley ball radius (default: 1)')
    parser.add_argument('--exact', action='store_true', help='Use exact arithmetic (default: double)')
    parser.add_argument('--workers', type=int, default=12, help='Parallel workers (default: 12)')
    parser.add_argument('--binary', type=str, default='./schubert', help='Path to schubert binary')
    args = parser.parse_args()

    n = args.n
    radius = args.radius

    if n not in OPTIMAL_LAYERED:
        print(f"No optimal layered composition known for n={n}")
        sys.exit(1)

    comp = OPTIMAL_LAYERED[n]
    center = build_layered(comp)
    center_ell = inversions(center)

    prec_label = "exact" if args.exact else "double"
    print(f"n = {n}, radius = {radius}, precision = {prec_label}")
    print(f"Optimal layered composition: {comp}")
    print(f"Center permutation: [{perm_to_str(center)}]")
    print(f"Center length: ℓ = {center_ell}")
    print()

    # First compute center value (with product formula for speed)
    print("Computing center value (product formula)...")
    prec = "--exact" if args.exact else "--double"
    cmd = [args.binary, perm_to_str(center), prec, "--cotrans"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    center_val = None
    for line in result.stdout.split('\n'):
        if line.startswith('Result:'):
            val_str = line.split('=')[-1].strip()
            center_val = float(val_str)
            break

    if center_val is None:
        print("ERROR: Could not compute center value")
        print(result.stdout)
        sys.exit(1)

    import math
    print(f"S_center(1^{n}) = {center_val:.6e}")
    print(f"log2 / n² = {math.log2(center_val) / (n*n):.8f}")
    print()

    # BFS to find ball
    print(f"Computing Cayley ball of radius {radius}...")
    ball = cayley_ball(center, radius)
    print(f"Ball size: {len(ball)} permutations")

    # Count by distance
    by_dist = {}
    for perm, d in ball.items():
        by_dist.setdefault(d, []).append(perm)
    for d in sorted(by_dist):
        print(f"  Distance {d}: {len(by_dist[d])} permutations")
    print()

    # For large n, limit parallelism to avoid OOM (BFS128 uses huge frontiers)
    if n >= 17:
        WORKERS = min(args.workers, 1)
        THREADS = 6  # single worker can use all cores
    elif n >= 14:
        WORKERS = min(args.workers, 2)
        THREADS = 3
    else:
        WORKERS = args.workers
        THREADS = None  # default

    results = [(center, 0, center_val, center_ell)]
    work = [(perm, d) for perm, d in ball.items() if perm != center]
    total = len(ball)
    done_count = [1]  # use list for mutability in closure
    lock = threading.Lock()
    t0 = time.time()

    def do_one(perm, d):
        ell = inversions(perm)
        val = compute_schubert(perm, binary=args.binary, exact=args.exact, threads=THREADS)
        with lock:
            done_count[0] += 1
            sys.stdout.write(f"\r  [{done_count[0]}/{total}] d={d}, ℓ={ell}  ")
            sys.stdout.flush()
        return (perm, d, val, ell)

    print(f"Computing {len(work)} permutations ({WORKERS} workers, {THREADS or 'default'} threads/each)...")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(do_one, perm, d) for perm, d in work]
        for f in as_completed(futures):
            results.append(f.result())

    elapsed = time.time() - t0
    print(f"\r  Done in {elapsed:.1f}s" + " " * 40)
    print()

    # Sort by value descending
    results.sort(key=lambda x: -(x[2] or 0))

    # Display results
    print(f"{'Rank':>4} {'Dist':>4} {'ℓ':>5} {'Δℓ':>4} {'S_w':>20} {'ratio':>10} {'log2/n²':>10}  Permutation")
    print("-" * 120)

    for rank, (perm, d, val, ell) in enumerate(results, 1):
        if val is None:
            ratio_str = "ERROR"
            log_str = "ERROR"
        else:
            ratio = val / center_val
            log_val = math.log2(val) / (n*n) if val > 0 else float('-inf')
            ratio_str = f"{ratio:.6f}"
            log_str = f"{log_val:.8f}"

        delta_ell = ell - center_ell

        # Mark if layered
        marker = " *" if perm == center else ""
        perm_str = f"[{perm_to_str(perm)}]"
        if len(perm_str) > 60:
            perm_str = perm_str[:57] + "...]"

        print(f"{rank:>4} {d:>4} {ell:>5} {delta_ell:>+4} {val:>20.6e} {ratio_str:>10} {log_str:>10}  {perm_str}{marker}")

        if rank >= 50 and len(results) > 55:
            print(f"  ... ({len(results) - 50} more permutations omitted)")
            break

    print()
    print(f"* = center (max layered)")

    # Summary
    better = sum(1 for _, _, v, _ in results if v is not None and v > center_val)
    equal = sum(1 for _, _, v, _ in results if v is not None and v == center_val)
    worse = sum(1 for _, _, v, _ in results if v is not None and v < center_val)

    print(f"\nSummary: {better} better, {equal} equal, {worse} worse than center")

    # Print equal-or-better permutations with full details
    print(f"\nEqual or better than center (ratio >= 1.0):")
    for perm, d, val, ell in results:
        if val is not None and val >= center_val:
            ratio = val / center_val
            if perm == center:
                label = "CENTER"
            elif val > center_val:
                label = "BETTER"
            else:
                label = "equal"
            print(f"  [{label}] d={d}, ℓ={ell}, ratio={ratio:.8f}: [{perm_to_str(perm)}]")

if __name__ == '__main__':
    main()
