#!/usr/bin/env python3
"""Autocorrelation / ESS analysis for an MCMC validation run.

Reads the exported outputs produced by ``code/bpd_mcmc`` in batch mode
(``perms_n<N>_B<B>*.txt``, optionally ``height_all_n<N>_B<B>*.txt`` and
``tile_counts_n<N>_B<B>*.txt``), computes a fixed list of scalar observables
per sample, estimates lag-1 .. lag-``max-lag`` autocorrelation, applies the
Sokal automatic-windowing rule to estimate the integrated autocorrelation
time tau_int, and writes two CSVs:

  --out-autocorr  observable, lag, acf, n_samples
  --out-ess       observable, n_samples, mean, var, tau_int, window, ess

Observables:

  - ell               : number of inversions of w, which equals the cross
                        count of any reduced BPD for w (matches output of
                        bpd_mcmc's on-screen "ell")
  - se_boundary       : row index i (1-based) such that w(i) = n; a scalar
                        proxy for the southeast boundary of the permuton
  - M_<r>_<c>         : indicator 1{w(r) = c}, for (r, c) in {(10,10),
                        (30,30), (50,50)} (1-based, only emitted if r <= n
                        and c <= n)
  - density_<tag>     : fraction of tiles of type ``tag`` in the BPD, for
                        tag in {cross, straight, elbow} (from the
                        tile_counts file; skipped if unavailable). The
                        ``empty`` density is omitted because reduced BPDs
                        of S_n permutations satisfy #blank = #cross
                        (hence density_empty coincides with density_cross
                        sample-by-sample).
  - height_<r>_<c>    : value of the height function at probe (r, c),
                        for (r, c) in {(n//4, n//4), (n//2, n//2),
                        (3n//4, 3n//4), (n//2, n//4)} (skipped if the
                        height_all file is unavailable)

Synthetic unit test:

  ./analyze_mcmc_validation.py --synthetic-gaussian 10000 --seed 42 \
      --out-ess /tmp/gauss.csv

  emits a single CSV row for observable ``gaussian``. For i.i.d. N(0,1),
  Sokal tau_int should be close to 0.5 and ESS close to B.

This script is deterministic: the same inputs produce byte-identical
CSVs. Floating-point values are rendered with a fixed format.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


INT_RE = re.compile(r"-?\d+")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def load_perms(path: Path, n: int) -> np.ndarray:
    """Parse the Mathematica-format ``{{...},{...},...}`` perms file.

    Returns an (B, n) int array with values in 1..n.
    """
    text = path.read_text()
    ints = np.array([int(m.group()) for m in INT_RE.finditer(text)], dtype=np.int64)
    if ints.size % n != 0:
        raise ValueError(
            f"perms file {path} has {ints.size} integers, not a multiple of n={n}"
        )
    perms = ints.reshape(-1, n)
    if (perms.min() < 1) or (perms.max() > n):
        raise ValueError(
            f"perms file {path} contains values outside 1..{n} (min={perms.min()}, "
            f"max={perms.max()}); wrong n?"
        )
    return perms


def load_tile_counts(path: Path) -> np.ndarray:
    """Parse the tile_counts_n*_B*.txt whitespace-separated file.

    Returns a (B, 6) int array. Tile type columns are:
    0=blank, 1=cross, 2=r_elbow, 3=j_elbow, 4=vert, 5=horiz.
    """
    data = np.loadtxt(path, dtype=np.int64, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 6:
        raise ValueError(f"tile_counts file {path} must have 6 columns, got {data.shape[1]}")
    return data


def load_heights(path: Path, n: int) -> np.ndarray:
    """Parse the height_all_n*_B*.txt block file.

    Returns an (B, n+1, n+1) int array. Blocks are delimited by ``# sample``
    comment lines; each block has n+1 rows of n+1 whitespace-separated ints.
    """
    m = n + 1
    samples: list[list[list[int]]] = []
    current: list[list[int]] = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if current:
                    samples.append(current)
                    current = []
                continue
            row = [int(x) for x in s.split()]
            if len(row) != m:
                raise ValueError(
                    f"height file {path} row has {len(row)} cols, expected n+1={m}"
                )
            current.append(row)
    if current:
        samples.append(current)
    arr = np.array(samples, dtype=np.int64)
    if arr.ndim != 3 or arr.shape[1] != m or arr.shape[2] != m:
        raise ValueError(
            f"height file {path} parsed shape {arr.shape}, expected (B, {m}, {m})"
        )
    return arr


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


def inversions(perm_row: np.ndarray) -> int:
    """Number of inversions of a 1-based permutation.

    Implemented with an O(n^2) double loop which is fine for n <= 200.
    """
    n = perm_row.size
    count = 0
    for i in range(n):
        a = perm_row[i]
        # count j > i with perm[j] < a
        count += int(np.sum(perm_row[i + 1:] < a))
    return count


def build_observables(
    perms: np.ndarray,
    tile_counts: np.ndarray | None,
    heights: np.ndarray | None,
    n: int,
) -> dict[str, np.ndarray]:
    B = perms.shape[0]
    obs: dict[str, np.ndarray] = {}

    ell = np.fromiter((inversions(perms[b]) for b in range(B)), dtype=np.float64, count=B)
    obs["ell"] = ell

    # SE-boundary proxy: row index (1-based) i such that w(i) = n
    se_boundary = np.zeros(B, dtype=np.float64)
    for b in range(B):
        idx = np.where(perms[b] == n)[0]
        if idx.size != 1:
            raise ValueError(f"row {b} in perms is not a valid permutation")
        se_boundary[b] = int(idx[0]) + 1  # 1-based
    obs["se_boundary"] = se_boundary

    # Permutation-matrix cell indicators M[r][c] (1-based); only emit if r,c <= n
    for (r, c) in [(10, 10), (30, 30), (50, 50)]:
        if r <= n and c <= n:
            indicator = (perms[:, r - 1] == c).astype(np.float64)
            obs[f"M_{r}_{c}"] = indicator

    if tile_counts is not None:
        if tile_counts.shape[0] != B:
            raise ValueError(
                f"tile_counts has {tile_counts.shape[0]} rows, expected {B}"
            )
        total = float(n * n)
        cross = tile_counts[:, 1].astype(np.float64) / total
        straight = (tile_counts[:, 4] + tile_counts[:, 5]).astype(np.float64) / total
        elbow = (tile_counts[:, 2] + tile_counts[:, 3]).astype(np.float64) / total
        obs["density_cross"] = cross
        obs["density_straight"] = straight
        obs["density_elbow"] = elbow

    if heights is not None:
        if heights.shape[0] != B:
            raise ValueError(f"heights has {heights.shape[0]} blocks, expected {B}")
        probes = [(n // 4, n // 4), (n // 2, n // 2), (3 * n // 4, 3 * n // 4), (n // 2, n // 4)]
        # height grid is (n+1) x (n+1); 1-indexed probe (r, c) lives at index [r][c]
        m = n + 1
        for (r, c) in probes:
            if 0 <= r < m and 0 <= c < m:
                obs[f"height_{r}_{c}"] = heights[:, r, c].astype(np.float64)

    return obs


# ---------------------------------------------------------------------------
# Autocorrelation / Sokal window
# ---------------------------------------------------------------------------


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation rho(t) for t = 0..max_lag.

    Uses FFT-based computation. Returns a length-(max_lag+1) array with
    rho(0) = 1 (by construction, when variance > 0) and rho(t) the lag-t
    autocorrelation normalised by the lag-0 autocovariance.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.array([1.0] + [0.0] * max_lag)
    d = x - x.mean()
    var0 = float(np.dot(d, d))
    if var0 <= 0.0:
        out = np.zeros(max_lag + 1)
        out[0] = 1.0
        return out
    # FFT-based autocovariance: pad to next power of two of 2n
    size = 1
    while size < 2 * n:
        size <<= 1
    f = np.fft.rfft(d, n=size)
    ac = np.fft.irfft(f * np.conjugate(f), n=size)[: max_lag + 1]
    # Biased estimator: divide by n (not n - lag)
    rho = ac / var0
    return rho


def sokal_window(rho: np.ndarray, c: float = 6.0) -> tuple[float, int, bool]:
    """Sokal automatic-windowing integrated autocorrelation time.

    tau_int(M) = 0.5 + sum_{t=1}^{M} rho(t). Window M is the smallest M
    with M >= c * tau_int(M). Returns (tau_int, M, converged). If no
    window is reached before rho runs out, returns the full-range sum,
    the last lag, and converged=False so callers can flag the result as
    unreliable (the truncation is driven by max_lag, not by the Sokal
    rule, and tau_int may be biased).
    """
    tau = 0.5
    for M in range(1, rho.size):
        tau += float(rho[M])
        if M >= c * tau:
            return tau, M, True
    return tau, rho.size - 1, False


@dataclass
class ObsResult:
    name: str
    n_samples: int
    mean: float
    var: float
    tau_int: float
    window: int
    ess: float
    acf: np.ndarray  # length max_lag + 1


def analyze_observables(
    obs: dict[str, np.ndarray],
    max_lag: int,
    sokal_c: float,
) -> list[ObsResult]:
    results: list[ObsResult] = []
    for name in obs:
        x = obs[name]
        n = x.size
        rho = autocorrelation(x, max_lag)
        tau, window, converged = sokal_window(rho, c=sokal_c)
        if not converged:
            # Window truncation is driven by max_lag, not by the Sokal rule:
            # tau_int/ESS may be biased. Fail loudly so the caller notices
            # rather than shipping an unreliable ESS row into the paper.
            print(
                f"warning: Sokal window for observable {name!r} did not converge "
                f"within max_lag={max_lag} (window hit ceiling at lag={window}, "
                f"tau_int={tau:.6g}). Increase --max-lag or collect more samples; "
                f"the reported tau_int/ESS are likely biased.",
                file=sys.stderr,
            )
        ess = n / (2.0 * tau) if tau > 0 else float("inf")
        results.append(
            ObsResult(
                name=name,
                n_samples=n,
                mean=float(x.mean()),
                var=float(x.var(ddof=1)) if n > 1 else 0.0,
                tau_int=tau,
                window=window,
                ess=ess,
                acf=rho,
            )
        )
    return results


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def fmt_float(x: float) -> str:
    """Stable fixed-precision float formatter for byte-identical output."""
    if not np.isfinite(x):
        return "nan" if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return f"{x:.10g}"


def write_ess_csv(path: Path, results: Sequence[ObsResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["observable", "n_samples", "mean", "var", "tau_int", "window", "ess"])
        for r in results:
            w.writerow([
                r.name,
                r.n_samples,
                fmt_float(r.mean),
                fmt_float(r.var),
                fmt_float(r.tau_int),
                r.window,
                fmt_float(r.ess),
            ])


def write_autocorr_csv(path: Path, results: Sequence[ObsResult], max_lag: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["observable", "lag", "acf", "n_samples"])
        for r in results:
            upto = min(max_lag, r.acf.size - 1)
            for t in range(1, upto + 1):
                w.writerow([r.name, t, fmt_float(float(r.acf[t])), r.n_samples])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--perms", type=Path, help="path to perms_n<N>_B<B>*.txt (Mathematica-format)")
    p.add_argument("--heights", type=Path, default=None, help="optional per-sample height_all file")
    p.add_argument("--tile-counts", type=Path, default=None, help="optional tile_counts file")
    p.add_argument("--n", type=int, help="grid size n (required unless parsed from filename)")
    p.add_argument("--max-lag", type=int, default=100, help="max lag reported in autocorr CSV")
    p.add_argument("--sokal-c", type=float, default=6.0, help="Sokal automatic-windowing constant c")
    p.add_argument("--out-autocorr", type=Path, default=None, help="output CSV of per-lag autocorr")
    p.add_argument("--out-ess", type=Path, default=None, help="output CSV of per-observable ESS")
    p.add_argument(
        "--synthetic-gaussian",
        type=int,
        default=None,
        metavar="B",
        help="run the i.i.d. Gaussian unit test with B samples; ignores perms/--n",
    )
    p.add_argument("--seed", type=int, default=1, help="seed for the synthetic test")
    args = p.parse_args(argv)

    if args.synthetic_gaussian is not None:
        rng = np.random.default_rng(args.seed)
        x = rng.standard_normal(args.synthetic_gaussian)
        obs = {"gaussian": x}
        results = analyze_observables(obs, max_lag=args.max_lag, sokal_c=args.sokal_c)
        if args.out_ess:
            write_ess_csv(args.out_ess, results)
        if args.out_autocorr:
            write_autocorr_csv(args.out_autocorr, results, max_lag=args.max_lag)
        r = results[0]
        print(
            f"gaussian B={r.n_samples} mean={fmt_float(r.mean)} var={fmt_float(r.var)} "
            f"tau_int={fmt_float(r.tau_int)} window={r.window} ess={fmt_float(r.ess)}"
        )
        return 0

    if args.perms is None:
        p.error("--perms is required (or use --synthetic-gaussian B)")
    if args.n is None:
        m = re.search(r"perms_n(\d+)_B", str(args.perms))
        if not m:
            p.error("--n is required when filename does not contain perms_n<N>_B")
        n = int(m.group(1))
    else:
        n = args.n

    perms = load_perms(args.perms, n)
    # Reject B below a floor where autocorrelation / ESS are meaningless:
    # autocorrelation() early-returns [1.0, 0.0, ...] for n < 2, and with
    # only a handful of samples the Sokal window collapses to tau_int=0.5
    # / ESS=B/2, which looks identical to the i.i.d. case and has silently
    # landed a garbage row in the SI before. The multistart driver in
    # run_validation_suite.sh uses B=1 (trace-only); feeding those outputs
    # here must fail loudly rather than produce a misleading CSV.
    min_B = 20
    if perms.shape[0] < min_B:
        raise SystemExit(
            f"only {perms.shape[0]} samples in {args.perms}; need at least "
            f"{min_B} for autocorrelation / ESS to be meaningful. Use a "
            f"collection-mode run (large B) for this analysis; the "
            f"multistart/validation driver with B=1 only emits the burn-in "
            f"trace and is not suitable input for this script."
        )
    tile_counts = load_tile_counts(args.tile_counts) if args.tile_counts else None
    heights = load_heights(args.heights, n) if args.heights else None

    obs = build_observables(perms, tile_counts, heights, n)
    results = analyze_observables(obs, max_lag=args.max_lag, sokal_c=args.sokal_c)

    if args.out_autocorr:
        write_autocorr_csv(args.out_autocorr, results, max_lag=args.max_lag)
    if args.out_ess:
        write_ess_csv(args.out_ess, results)

    for r in results:
        print(
            f"{r.name:24s} B={r.n_samples} mean={fmt_float(r.mean)} "
            f"var={fmt_float(r.var)} tau_int={fmt_float(r.tau_int)} "
            f"window={r.window} ess={fmt_float(r.ess)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
