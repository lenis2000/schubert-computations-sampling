#!/usr/bin/env python3
"""Run bpd_mcmc and summarize burn-in acceptance rates by droop distribution.

Default use-case:
  n = 60, burnin = 5G, all four droop distributions, multi-start sweep
  over {w0, identity, rothe:<layered>}, one short post-burn-in sample per run.

The script runs each (distribution x start) configuration in a dedicated
output directory, captures stdout/stderr, parses the acceptance.jsonl written
by bpd_mcmc, and writes:

  - per_run.csv           one row per MCMC run
  - summary.csv           per-distribution aggregate
  - acceptance_rates.csv  matrix (dist x start) x (overall/local/droop/down/up)
  - acceptance_rates.tex  auto-generated LaTeX tabular snippet

There is also a `regenerate-tex` subcommand that rebuilds the .tex snippet
from an existing acceptance_rates.csv so that 'diff is empty on rerun' can
be verified deterministically without re-running MCMC.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import statistics as stats
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


DISTS = ("geometric", "loguniform", "uniform", "revlog")

# MPP3 layered-permutation block sizes from data_MPP3.tex, keyed by n.
# Covers the validation-relevant sizes (n=20 for the checked-in acceptance
# table, n=60 for the production sweep, n=100 for Rivanna) plus the full
# range in between so --n values used during debugging still get the
# documented MPP3 optimum rather than a silent degenerate fallback.
LAYERED_OPTIMUM_BY_N: dict[int, tuple[int, ...]] = {
    20: (1, 2, 5, 12),
    25: (1, 3, 6, 15),
    30: (1, 3, 8, 18),
    40: (1, 2, 4, 10, 23),
    50: (1, 2, 5, 13, 29),
    60: (1, 3, 6, 15, 35),
    80: (1, 1, 4, 8, 20, 46),
    100: (1, 2, 4, 11, 25, 57),
}


def build_layered_permutation(layers: tuple[int, ...]) -> list[int]:
    """Layered permutation with the given block sizes (reversed inside each block)."""
    w: list[int] = []
    pos = 1
    for b in layers:
        for j in range(b - 1, -1, -1):
            w.append(pos + j)
        pos += b
    return w


def default_rothe_start_for_n(n: int) -> str:
    """Return a `rothe:<csv>` start string for n using the MPP3 layered optimum.

    Raises SystemExit when n is not in LAYERED_OPTIMUM_BY_N: falling back
    to an ad-hoc split silently mis-labels the resulting row as a Rothe
    start for the MPP3 optimum, which is how the checked-in tables are
    documented.
    """
    if n not in LAYERED_OPTIMUM_BY_N:
        known = ", ".join(str(k) for k in sorted(LAYERED_OPTIMUM_BY_N))
        raise SystemExit(
            f"no MPP3 layered optimum tabulated for n={n}; "
            f"known n are {known}. Pass --starts explicitly (e.g. "
            f"'rothe:<csv>') or extend LAYERED_OPTIMUM_BY_N."
        )
    w = build_layered_permutation(LAYERED_OPTIMUM_BY_N[n])
    return "rothe:" + ",".join(str(v) for v in w)


def parse_rothe_values(spec: str) -> list[int]:
    """Parse the CSV body of a rothe:<csv> spec into an int list.

    Raises argparse.ArgumentTypeError if the body is not a well-formed
    permutation of 1..k.
    """
    assert spec.startswith("rothe:")
    body = spec[len("rothe:"):]
    parts = [s.strip() for s in body.split(",")]
    if not parts or any(not p for p in parts):
        raise argparse.ArgumentTypeError(
            f"invalid --start spec {spec!r}: rothe: requires a comma-separated list"
        )
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --start spec {spec!r}: rothe: values must be integers"
        )
    k = len(values)
    if sorted(values) != list(range(1, k + 1)):
        raise argparse.ArgumentTypeError(
            f"invalid --start spec {spec!r}: "
            f"rothe: values must be a permutation of 1..{k}"
        )
    return values


def resolve_start_labels(starts: list[str]) -> dict[str, str]:
    """Return {spec: short-label} giving a unique label per distinct start spec.

    Non-rothe specs use themselves as labels. For rothe specs, a single
    distinct rothe gets label "rothe"; multiple distinct rothes get
    "rothe1", "rothe2", ... in the order of first appearance. Identical
    rothe specs share a label (they are the same run configuration).
    """
    unique_rothes: list[str] = []
    for spec in starts:
        if spec.startswith("rothe:") and spec not in unique_rothes:
            unique_rothes.append(spec)
    labels: dict[str, str] = {}
    multi = len(unique_rothes) > 1
    for spec in starts:
        if spec in labels:
            continue
        if spec.startswith("rothe:"):
            if multi:
                labels[spec] = f"rothe{unique_rothes.index(spec) + 1}"
            else:
                labels[spec] = "rothe"
        else:
            labels[spec] = spec
    return labels


def validate_start_spec(spec: str) -> str:
    allowed = {"w0", "identity", "random"}
    if spec in allowed:
        return spec
    if spec.startswith("rothe:"):
        parse_rothe_values(spec)  # raises ArgumentTypeError on malformed input
        return spec
    raise argparse.ArgumentTypeError(
        f"invalid --start spec {spec!r}; "
        f"expected one of {sorted(allowed)} or 'rothe:<csv>'"
    )


@dataclass
class RunResult:
    dist: str
    start: str
    start_detail: str
    repeat: int
    seed: int
    burnin_elapsed_s: float
    burnin_accept_pct: float
    burnin_droop_accept_pct: float | None
    burnin_local_accept_pct: float | None
    burnin_droop_down_pct: float | None
    burnin_droop_up_pct: float | None
    burnin_ell: int | None
    proposals: int | None
    accepts: int | None
    local_proposals: int | None
    local_accepts: int | None
    droop_proposals: int | None
    droop_accepts: int | None
    droop_down_accepts: int | None
    droop_up_accepts: int | None
    sampling_accept_pct: float | None
    sampling_droop_accept_pct: float | None
    raw_log: str
    jsonl_line: dict[str, object] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    regen = subparsers.add_parser(
        "regenerate-tex",
        help="Rebuild acceptance_rates.tex from an existing acceptance_rates.csv",
    )
    regen.add_argument("--csv", required=True, type=Path,
                       help="Path to acceptance_rates.csv")
    regen.add_argument("--tex", required=True, type=Path,
                       help="Path to write acceptance_rates.tex")

    parser.add_argument("--binary", default="./bpd_mcmc",
                        help="Path to the bpd_mcmc binary (default: ./bpd_mcmc)")
    parser.add_argument("--n", type=int, default=60,
                        help="Problem size n (default: 60)")
    parser.add_argument("--burnin", default="5G",
                        help="Burn-in length passed to bpd_mcmc (default: 5G)")
    parser.add_argument("--starts", nargs="+", type=validate_start_spec,
                        default=None,
                        help="Starting states to sweep over "
                             "(e.g. 'w0' 'identity' 'rothe:1,4,3,2,...'). "
                             "Default: w0 identity rothe:<MPP3 layered for --n>")
    parser.add_argument("--anchor", default="se", choices=("se", "nw"),
                        help="Droop anchor (default: se)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of post-burn-in samples to collect (default: 1)")
    parser.add_argument("--thin", default="1",
                        help="Thinning interval for the short collection phase (default: 1)")
    parser.add_argument("--geom-p", type=float, default=None,
                        help="Geometric side-length parameter passed to bpd_mcmc")
    parser.add_argument("--geom-mean", type=float, default=None,
                        help="Geometric mean side length passed to bpd_mcmc")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of chains/threads (default: 1)")
    parser.add_argument("--seed", type=int, default=5000,
                        help="Base seed (default: 5000)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats per (distribution, start) pair (default: 1)")
    parser.add_argument("--dists", nargs="+", default=list(DISTS), choices=DISTS,
                        help="Distributions to test (default: all four)")
    parser.add_argument("--outdir", default=None,
                        help="Directory for raw logs and summaries (default: timestamped)")

    args = parser.parse_args(argv)
    return args


def ensure_binary(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"binary not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"binary path is not a file: {path}")
    return path


def default_outdir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"acceptance_runs_{stamp}"


def build_command(args: argparse.Namespace, binary: Path, dist: str,
                  start: str, seed: int) -> list[str]:
    cmd = [
        str(binary),
        f"batch:{args.n}:{args.samples}",
        "--droop-dist", dist,
        "--start", start,
        "--burnin", args.burnin,
        "--thin", args.thin,
        "--anchor", args.anchor,
        "--threads", str(args.threads),
        "--seed", str(seed),
        "--no-png",
        "--no-tikz",
        "--no-height",
    ]
    if args.geom_p is not None:
        cmd.extend(["--geom-p", str(args.geom_p)])
    if args.geom_mean is not None:
        cmd.extend(["--geom-mean", str(args.geom_mean)])
    return cmd


_JSON_RE = re.compile(r"^\{.*\}$")


def load_jsonl(path: Path) -> list[dict]:
    """Read JSON-per-line entries from path. Raises if file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"acceptance.jsonl not found at {path}: bpd_mcmc is expected to "
            f"write this file in batch mode. Rebuild the binary and re-run."
        )
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not _JSON_RE.match(line):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def find_burnin_jsonl(entries: list[dict], n: int, seed: int, dist: str,
                      start_detail: str) -> dict | None:
    # Caller must restrict `entries` to rows appended by the just-finished
    # subprocess (e.g., a suffix slice of the JSONL taken after recording the
    # pre-run row count). Matching on n/seed/dist/start_detail then picks the
    # burn-in row for this invocation from among the new rows (which may also
    # include a collection-phase row).
    matches = [
        e for e in entries
        if e.get("phase") == "burnin"
        and e.get("n") == n
        and e.get("seed") == seed
        and e.get("dist") == dist
        and e.get("start_detail") == start_detail
    ]
    return matches[-1] if matches else None


def parse_burnin_line(stdout: str) -> tuple[float, int | None]:
    # Only parse fields that are *not* present in acceptance.jsonl:
    # elapsed wall time, and the thread-0 ell position. Acceptance
    # percentages come from the JSONL counters instead so that per_run.csv
    # / summary.csv match acceptance_rates.csv (both aggregate across all
    # chains, whereas stdout only shows "Thread 0 acceptance").
    burnin_match = re.search(
        r"Burn-in done \(\d+ chains, ([0-9.]+)s\)\.",
        stdout,
    )
    if not burnin_match:
        raise ValueError("could not find burn-in elapsed-time summary in stdout")

    burnin_elapsed_s = float(burnin_match.group(1))
    ell_match = re.search(r", ell=([0-9]+)", stdout)
    burnin_ell = int(ell_match.group(1)) if ell_match else None

    return burnin_elapsed_s, burnin_ell


def parse_sampling_line(stdout: str) -> tuple[float | None, float | None]:
    sampling_match = re.search(r"Sampling acceptance: ([0-9.]+)% overall", stdout)
    if not sampling_match:
        return None, None

    sampling_accept_pct = float(sampling_match.group(1))
    droop_match = re.search(
        r"Sampling acceptance: [0-9.]+% overall, droop ([0-9.]+)% \(down [^,]+, up [^)]+\)",
        stdout,
    )
    sampling_droop_accept_pct = float(droop_match.group(1)) if droop_match else None
    return sampling_accept_pct, sampling_droop_accept_pct


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    # Use a stable, union-of-keys column order so dataclass-asdict and synthetic
    # rows both serialize cleanly.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[RunResult]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    keys = sorted(
        {(r.dist, r.start) for r in results},
        key=lambda ds: (DISTS.index(ds[0]), ds[1]),
    )
    for dist, start in keys:
        subset = [r for r in results if r.dist == dist and r.start == start]
        droop_vals = [r.burnin_droop_accept_pct for r in subset
                      if r.burnin_droop_accept_pct is not None]
        local_vals = [r.burnin_local_accept_pct for r in subset
                      if r.burnin_local_accept_pct is not None]
        ell_vals = [r.burnin_ell for r in subset if r.burnin_ell is not None]
        row: dict[str, object] = {
            "dist": dist,
            "start": start,
            "runs": len(subset),
            "burnin_accept_pct_mean": round(
                stats.mean(r.burnin_accept_pct for r in subset), 3
            ),
            "burnin_accept_pct_min": round(
                min(r.burnin_accept_pct for r in subset), 3
            ),
            "burnin_accept_pct_max": round(
                max(r.burnin_accept_pct for r in subset), 3
            ),
            "burnin_droop_accept_pct_mean": (
                round(stats.mean(droop_vals), 3) if droop_vals else None
            ),
            "burnin_local_accept_pct_mean": (
                round(stats.mean(local_vals), 3) if local_vals else None
            ),
            "burnin_ell_mean": (
                round(stats.mean(ell_vals), 1) if ell_vals else None
            ),
        }
        if len(subset) > 1:
            row["burnin_accept_pct_sd"] = round(
                stats.stdev(r.burnin_accept_pct for r in subset), 3
            )
            row["burnin_droop_accept_pct_sd"] = (
                round(stats.stdev(droop_vals), 3) if len(droop_vals) > 1 else 0.0
            )
            row["burnin_local_accept_pct_sd"] = (
                round(stats.stdev(local_vals), 3) if len(local_vals) > 1 else 0.0
            )
        summary.append(row)
    return summary


def _pct(numer: float | int | None, denom: float | int | None) -> float | None:
    if numer is None or denom is None or denom == 0:
        return None
    return 100.0 * float(numer) / float(denom)


def build_matrix_rows(results: list[RunResult]) -> list[dict[str, object]]:
    """Aggregate per-run results into (dist, start) matrix rows.

    Columns: n, dist, start, runs, overall_pct, local_pct, droop_pct,
             droop_down_pct, droop_up_pct, proposals, local_proposals,
             droop_proposals, accepts, local_accepts, droop_accepts,
             droop_down_accepts, droop_up_accepts
    """
    rows: list[dict[str, object]] = []
    keys = sorted(
        {(r.dist, r.start) for r in results},
        key=lambda ds: (DISTS.index(ds[0]), ds[1]),
    )
    for dist, start in keys:
        subset = [r for r in results if r.dist == dist and r.start == start]
        # Sum raw counters across repeats so percentages are weighted correctly.
        totals = {
            key: sum(getattr(r, key) or 0 for r in subset)
            for key in (
                "proposals",
                "accepts",
                "local_proposals",
                "local_accepts",
                "droop_proposals",
                "droop_accepts",
                "droop_down_accepts",
                "droop_up_accepts",
            )
        }
        n_vals = {r.jsonl_line["n"] for r in subset if r.jsonl_line is not None}
        n_val = next(iter(n_vals)) if len(n_vals) == 1 else None
        row: dict[str, object] = {
            "n": n_val,
            "dist": dist,
            "start": start,
            "runs": len(subset),
            "overall_pct": _pct(totals["accepts"], totals["proposals"]),
            "local_pct": _pct(totals["local_accepts"], totals["local_proposals"]),
            "droop_pct": _pct(totals["droop_accepts"], totals["droop_proposals"]),
            "droop_down_pct": _pct(totals["droop_down_accepts"], totals["droop_proposals"]),
            "droop_up_pct": _pct(totals["droop_up_accepts"], totals["droop_proposals"]),
            **totals,
        }
        # Round percentages for CSV/LaTeX display; keep None passthrough.
        for key in ("overall_pct", "local_pct", "droop_pct",
                    "droop_down_pct", "droop_up_pct"):
            val = row[key]
            if val is not None:
                row[key] = round(val, 4)
        rows.append(row)
    return rows


def render_tex_table(matrix_rows: list[dict[str, object]]) -> str:
    """Render a booktabs-free LaTeX tabular snippet from matrix rows.

    Layout:
      dist  start  overall%  local%  droop%  droop_down%  droop_up%
    """
    lines: list[str] = []
    lines.append("% Auto-generated by measure_acceptance_rates.py. Do not edit by hand.")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\hline")
    lines.append(" & & \\multicolumn{5}{c}{acceptance rate (\\%)} \\\\")
    lines.append("dist & start & overall & local & droop & droop down & droop up \\\\")
    lines.append("\\hline")

    def fmt(val: object) -> str:
        if val is None or val == "":
            return "--"
        if isinstance(val, (int,)):
            return f"{val}"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)

    for row in matrix_rows:
        lines.append(
            f"{row['dist']} & {row['start']} & "
            f"{fmt(row.get('overall_pct'))} & "
            f"{fmt(row.get('local_pct'))} & "
            f"{fmt(row.get('droop_pct'))} & "
            f"{fmt(row.get('droop_down_pct'))} & "
            f"{fmt(row.get('droop_up_pct'))} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def read_matrix_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            cleaned: dict[str, object] = {}
            for key, val in raw.items():
                if val == "" or val is None:
                    cleaned[key] = None
                    continue
                # Try int, then float, then string.
                try:
                    cleaned[key] = int(val)
                    continue
                except ValueError:
                    pass
                try:
                    cleaned[key] = float(val)
                    continue
                except ValueError:
                    pass
                cleaned[key] = val
            rows.append(cleaned)
    return rows


def regenerate_tex(csv_path: Path, tex_path: Path) -> int:
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    matrix = read_matrix_csv(csv_path)
    tex = render_tex_table(matrix)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="ascii")
    print(f"wrote {tex_path} ({len(matrix)} rows)")
    return 0


def print_table(results: list[RunResult], summary_rows: list[dict[str, object]],
                matrix_rows: list[dict[str, object]]) -> None:
    print("\nPer-run burn-in acceptance")
    print("dist         start       repeat  seed   overall%  droop%  local%  ell")
    for result in results:
        droop = (f"{result.burnin_droop_accept_pct:.3f}"
                 if result.burnin_droop_accept_pct is not None else "-")
        local = (f"{result.burnin_local_accept_pct:.3f}"
                 if result.burnin_local_accept_pct is not None else "-")
        ell = str(result.burnin_ell) if result.burnin_ell is not None else "-"
        print(f"{result.dist:11s} {result.start:10s} {result.repeat:6d} "
              f"{result.seed:5d} {result.burnin_accept_pct:8.3f} "
              f"{droop:7s} {local:7s} {ell}")

    print("\nAggregated burn-in acceptance")
    header = "dist         start       runs  overall_mean  droop_mean  local_mean  ell_mean"
    print(header)
    for row in summary_rows:
        def f(val: object, w: int, prec: int) -> str:
            return f"{val:{w}.{prec}f}" if isinstance(val, (int, float)) else "-".rjust(w)
        print(f"{row['dist']:11s} {row['start']:10s} {row['runs']:4d} "
              f"{f(row['burnin_accept_pct_mean'], 12, 3)} "
              f"{f(row['burnin_droop_accept_pct_mean'], 11, 3)} "
              f"{f(row['burnin_local_accept_pct_mean'], 11, 3)} "
              f"{f(row['burnin_ell_mean'], 8, 1)}")

    print("\nMatrix: (dist x start) x acceptance columns")
    print("dist         start       overall%  local%    droop%    down%     up%")
    for row in matrix_rows:
        def f(val: object) -> str:
            return f"{val:9.4f}" if isinstance(val, float) else "     -   "
        print(f"{row['dist']:11s} {row['start']:10s} "
              f"{f(row.get('overall_pct'))} {f(row.get('local_pct'))} "
              f"{f(row.get('droop_pct'))} {f(row.get('droop_down_pct'))} "
              f"{f(row.get('droop_up_pct'))}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if getattr(args, "command", None) == "regenerate-tex":
        return regenerate_tex(args.csv, args.tex)

    if args.geom_p is not None and args.geom_mean is not None:
        raise SystemExit("use at most one of --geom-p and --geom-mean")

    if args.starts is None:
        args.starts = ["w0", "identity", default_rothe_start_for_n(args.n)]

    # Fail fast if any rothe:<csv> start's length doesn't match --n, so that a
    # later-in-the-sweep typo can't waste hours of earlier runs.
    for spec in args.starts:
        if spec.startswith("rothe:"):
            values = parse_rothe_values(spec)
            if len(values) != args.n:
                raise SystemExit(
                    f"--start {spec!r} has length {len(values)} but --n is {args.n}"
                )

    binary = ensure_binary(args.binary)

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir else default_outdir().resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "acceptance.jsonl"

    meta_path = outdir / "README.txt"
    meta_path.write_text(
        "Acceptance-rate sweep for bpd_mcmc\n"
        f"binary: {binary}\n"
        f"n: {args.n}\n"
        f"burnin: {args.burnin}\n"
        f"starts: {' '.join(args.starts)}\n"
        f"anchor: {args.anchor}\n"
        f"samples: {args.samples}\n"
        f"thin: {args.thin}\n"
        f"geom_p: {args.geom_p}\n"
        f"geom_mean: {args.geom_mean}\n"
        f"threads: {args.threads}\n"
        f"seed base: {args.seed}\n"
        f"repeats: {args.repeats}\n"
        f"dists: {' '.join(args.dists)}\n",
        encoding="ascii",
    )

    results: list[RunResult] = []
    labels_by_spec = resolve_start_labels(args.starts)
    print(f"Output directory: {outdir}")
    for start_idx, start_spec in enumerate(args.starts):
        slabel = labels_by_spec[start_spec]
        for dist_idx, dist in enumerate(args.dists):
            for repeat in range(args.repeats):
                # Seed derivation: base + 100000*start_idx + 1000*repeat + dist_idx.
                # The 100000 multiplier leaves room for up to 100 repeats before
                # the repeat bucket of one start collides with the next start.
                seed = (
                    args.seed + 100000 * start_idx
                    + 1000 * repeat + dist_idx
                )
                cmd = build_command(args, binary, dist, start_spec, seed)
                run_name = f"{slabel}_{dist}_r{repeat + 1}"
                log_path = outdir / f"{run_name}.log"

                # Snapshot the pre-run row count so we can isolate rows that
                # *this* subprocess appends. Without this, a binary that runs
                # to returncode=0 but fails to append (e.g., old build without
                # JSONL support) would silently inherit an old row matching
                # the same n/seed/dist/start_detail.
                jsonl_rows_before = (
                    len(load_jsonl(jsonl_path)) if jsonl_path.exists() else 0
                )

                print(f"\n[{run_name}] {shlex.join(cmd)}")
                completed = subprocess.run(
                    cmd,
                    cwd=outdir,
                    capture_output=True,
                    text=True,
                )
                log_path.write_text(
                    f"$ {shlex.join(cmd)}\n\nSTDOUT\n{completed.stdout}"
                    f"\n\nSTDERR\n{completed.stderr}",
                    encoding="utf-8",
                )
                if completed.returncode != 0:
                    print(f"command failed; see {log_path}", file=sys.stderr)
                    return completed.returncode

                burnin_elapsed_s, burnin_ell = parse_burnin_line(completed.stdout)
                sampling_accept_pct, sampling_droop_accept_pct = parse_sampling_line(
                    completed.stdout
                )

                # Pull fine-grained counters from acceptance.jsonl (burn-in phase).
                # Only consider rows appended during this subprocess: if the
                # binary returned 0 but did not append anything (e.g., old
                # build), `new_entries` is empty and we fail hard instead of
                # silently reusing a stale historical row with the same
                # n/seed/dist/start_detail.
                entries = load_jsonl(jsonl_path)
                new_entries = entries[jsonl_rows_before:]
                if not new_entries:
                    raise SystemExit(
                        f"bpd_mcmc appended no new rows to {jsonl_path} for "
                        f"n={args.n}, seed={seed}, dist={dist!r}, "
                        f"start_detail={start_spec!r}. "
                        f"The binary likely lacks JSONL support; "
                        f"rebuild from current sources and re-run."
                    )
                jline = find_burnin_jsonl(new_entries, args.n, seed, dist, start_spec)
                if jline is None:
                    raise SystemExit(
                        f"no burn-in JSONL entry found in {jsonl_path} for "
                        f"n={args.n}, seed={seed}, dist={dist!r}, "
                        f"start_detail={start_spec!r}. "
                        f"The bpd_mcmc binary did not tag its output correctly; "
                        f"rebuild from current sources and re-run."
                    )
                proposals = jline["proposals"]
                accepts = jline["accepts"]
                local_proposals = jline["local_proposals"]
                local_accepts = jline["local_accepts"]
                droop_proposals = jline["droop_proposals"]
                droop_accepts = jline["droop_accepts"]
                droop_down_accepts = jline["droop_down_accepts"]
                droop_up_accepts = jline["droop_up_accepts"]
                # Compute all burn-in acceptance percentages from the JSONL
                # counters (aggregated across all T chains). This keeps
                # per_run.csv / summary.csv consistent with acceptance_rates.csv,
                # which is built from the same counters; the stdout line only
                # reports "Thread 0 acceptance", which would diverge when
                # --threads > 1.
                burnin_accept_pct = _pct(accepts, proposals) or 0.0
                burnin_local_accept_pct = _pct(local_accepts, local_proposals)
                burnin_droop_accept_pct = _pct(droop_accepts, droop_proposals)
                droop_down_pct = _pct(droop_down_accepts, droop_proposals)
                droop_up_pct = _pct(droop_up_accepts, droop_proposals)

                result = RunResult(
                    dist=dist,
                    start=slabel,
                    start_detail=start_spec,
                    repeat=repeat + 1,
                    seed=seed,
                    burnin_elapsed_s=burnin_elapsed_s,
                    burnin_accept_pct=burnin_accept_pct,
                    burnin_droop_accept_pct=burnin_droop_accept_pct,
                    burnin_local_accept_pct=burnin_local_accept_pct,
                    burnin_droop_down_pct=(round(droop_down_pct, 4)
                                           if droop_down_pct is not None else None),
                    burnin_droop_up_pct=(round(droop_up_pct, 4)
                                         if droop_up_pct is not None else None),
                    burnin_ell=burnin_ell,
                    proposals=proposals,
                    accepts=accepts,
                    local_proposals=local_proposals,
                    local_accepts=local_accepts,
                    droop_proposals=droop_proposals,
                    droop_accepts=droop_accepts,
                    droop_down_accepts=droop_down_accepts,
                    droop_up_accepts=droop_up_accepts,
                    sampling_accept_pct=sampling_accept_pct,
                    sampling_droop_accept_pct=sampling_droop_accept_pct,
                    # Record just the filename (the log lives in outdir next
                    # to the CSV). An absolute path is a diagnostic leak: it
                    # bakes the author's machine layout (e.g., /tmp/acc_n20)
                    # into committed CSVs, where it points at files that no
                    # longer exist once the CSV is copied into data/.
                    raw_log=log_path.name,
                    jsonl_line=jline,
                )
                results.append(result)

    per_run_rows: list[dict[str, object]] = []
    for r in results:
        row = asdict(r)
        # jsonl_line is a diagnostic; drop it from the CSV to keep columns small.
        row.pop("jsonl_line", None)
        per_run_rows.append(row)
    summary_rows = summarize(results)
    matrix_rows = build_matrix_rows(results)
    write_csv(outdir / "per_run.csv", per_run_rows)
    write_csv(outdir / "summary.csv", summary_rows)
    write_csv(outdir / "acceptance_rates.csv", matrix_rows)
    tex = render_tex_table(matrix_rows)
    (outdir / "acceptance_rates.tex").write_text(tex, encoding="ascii")
    print_table(results, summary_rows, matrix_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
