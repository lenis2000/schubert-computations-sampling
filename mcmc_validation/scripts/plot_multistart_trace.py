#!/usr/bin/env python3
"""Aggregate burn-in traces from multi-start bpd_mcmc runs into a CSV and a
two-panel trace figure.

Consumes a directory tree produced by ``run_validation_suite.sh``:

    <runs-dir>/<start>_s<seed>/*_burnin_trace_n<N>_*.txt

Each burn-in trace file is a whitespace table with columns ``step ell corner``
(plus comment header lines starting with ``#``) written natively by
``code/bpd_mcmc``. The ``corner`` statistic
    #{i : i + w(i) > 2n - n/4}
is the southeast-corner proxy used as the scalar "SE-boundary proxy" in the
PNAS SI validation figure.

Outputs:
  --out-csv  long-format CSV with columns start,seed,step,ell,corner
             (one row per trace point, deterministic order)
  --out-pdf  two-panel figure (ell and corner vs. MCMC step), one colour
             per start, mean line + min/max shaded band across seeds

Design notes:
- Trace rows are written start-major, then seed-major, then step-major, so
  rerunning on the same input produces a byte-identical CSV.
- Floats use ``%.10g`` to match the rest of the validation suite.
- The plotter is defensive: if a start has only one seed, the band collapses
  to a single line and matplotlib draws no fill.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STARTS_ORDER = ["w0", "identity", "rothe", "random"]
START_LABELS = {
    "w0": r"$\mathsf{b}_{w_0}$",
    "identity": r"$\mathsf{b}_{\mathrm{id}}$",
    "rothe": r"Rothe (MPP3 layered)",
    "random": r"random RBPD",
}
START_COLORS = {
    "w0": "#232D4B",       # UVA navy
    "identity": "#E57200", # UVA orange
    "rothe": "#2C7A2C",    # green
    "random": "#8B1A1A",   # dark red
}

TRACE_FILENAME_RE = re.compile(r"^.*burnin_trace_n(\d+)_.*\.txt$")
RUN_DIR_RE = re.compile(r"^(?P<start>[A-Za-z0-9]+)_s(?P<seed>\d+)$")


def fmt_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan" if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return f"{x:.10g}"


def find_trace_file(run_dir: Path, n: int) -> Path | None:
    candidates = []
    for p in sorted(run_dir.glob("*burnin_trace_n*.txt")):
        m = TRACE_FILENAME_RE.match(p.name)
        if m and int(m.group(1)) == n:
            candidates.append(p)
    if not candidates:
        return None
    # Most-recent timestamp first; filenames are timestamped YYYYMMDD_HHMMSS_...
    candidates.sort()
    return candidates[-1]


def load_trace(path: Path) -> np.ndarray:
    """Return an (T, 3) array of columns [step, ell, corner]."""
    data = np.loadtxt(path, comments="#", dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(
            f"burn-in trace {path} has {data.shape[1]} columns, expected 3 "
            f"(step ell corner)"
        )
    data = data[:, :3]
    # Older bpd_mcmc builds emitted a duplicate trailing row at the final step
    # (fixed in the current writer). Collapse repeated step indices so
    # rebuilds are deterministic regardless of which writer produced the file.
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    return data[np.sort(unique_idx)]


def collect_runs(runs_dir: Path, n: int) -> dict[str, dict[int, np.ndarray]]:
    """Return {start_label: {seed: trace_array}} from the runs directory."""
    out: dict[str, dict[int, np.ndarray]] = {}
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = RUN_DIR_RE.match(sub.name)
        if not m:
            continue
        start = m.group("start")
        seed = int(m.group("seed"))
        tf = find_trace_file(sub, n)
        if tf is None:
            print(f"warn: no burn-in trace for n={n} in {sub}", file=sys.stderr)
            continue
        trace = load_trace(tf)
        out.setdefault(start, {})[seed] = trace
    return out


def collect_from_csv(csv_path: Path) -> dict[str, dict[int, np.ndarray]]:
    """Reconstruct {start: {seed: trace}} from a committed trace CSV.

    This lets the figure rebuild from just the CSV without re-running the
    MCMC chains, which is the mode used when only the committed data (not
    the raw run dirs) is available.
    """
    out: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
    with csv_path.open() as f:
        r = csv.DictReader(f)
        required = {"start", "seed", "step", "ell", "corner"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in r:
            start = row["start"]
            seed = int(row["seed"])
            out.setdefault(start, {}).setdefault(seed, []).append(
                (int(row["step"]), int(row["ell"]), int(row["corner"]))
            )
    return {
        start: {seed: np.array(rows, dtype=np.int64) for seed, rows in seed_map.items()}
        for start, seed_map in out.items()
    }


def write_csv(path: Path, traces: dict[str, dict[int, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["start", "seed", "step", "ell", "corner"])
        for start in STARTS_ORDER:
            if start not in traces:
                continue
            for seed in sorted(traces[start]):
                trace = traces[start][seed]
                for step, ell, corner in trace:
                    w.writerow([start, seed, int(step), int(ell), int(corner)])


def align_to_common_steps(seed_traces: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect traces from multiple seeds of one start onto shared step indices.

    Returns (steps, ell_matrix, corner_matrix) where each matrix has shape
    (len(shared_steps), n_seeds). Seeds are ordered by increasing seed value.
    """
    seeds = sorted(seed_traces)
    assert seeds, "empty seed_traces"
    step_sets = [set(map(int, seed_traces[s][:, 0])) for s in seeds]
    shared = sorted(set.intersection(*step_sets))
    steps = np.array(shared, dtype=np.int64)
    ell_mat = np.empty((len(steps), len(seeds)), dtype=np.float64)
    corner_mat = np.empty((len(steps), len(seeds)), dtype=np.float64)
    for j, s in enumerate(seeds):
        tr = seed_traces[s]
        idx = {int(tr[k, 0]): k for k in range(tr.shape[0])}
        for i, st in enumerate(shared):
            k = idx[st]
            ell_mat[i, j] = float(tr[k, 1])
            corner_mat[i, j] = float(tr[k, 2])
    return steps, ell_mat, corner_mat


def plot_traces(
    path: Path,
    traces: dict[str, dict[int, np.ndarray]],
    n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150)
    ax_ell, ax_corner = axes

    n_seeds_seen = set()
    for start in STARTS_ORDER:
        if start not in traces:
            continue
        seed_traces = traces[start]
        n_seeds_seen.add(len(seed_traces))
        steps, ell_mat, corner_mat = align_to_common_steps(seed_traces)
        if steps.size == 0:
            print(
                f"warn: no shared step grid for start={start} across seeds "
                f"{sorted(seed_traces)}; skipping",
                file=sys.stderr,
            )
            continue
        color = START_COLORS[start]
        label = START_LABELS[start]

        ell_mean = ell_mat.mean(axis=1)
        corner_mean = corner_mat.mean(axis=1)
        ell_lo, ell_hi = ell_mat.min(axis=1), ell_mat.max(axis=1)
        corner_lo, corner_hi = corner_mat.min(axis=1), corner_mat.max(axis=1)

        ax_ell.plot(steps, ell_mean, color=color, label=label, lw=1.6)
        ax_corner.plot(steps, corner_mean, color=color, label=label, lw=1.6)
        if ell_mat.shape[1] > 1:
            ax_ell.fill_between(steps, ell_lo, ell_hi, color=color, alpha=0.18, lw=0)
            ax_corner.fill_between(steps, corner_lo, corner_hi, color=color, alpha=0.18, lw=0)

    # Layered-optimum theoretical ell reference for b_id chain's approximate
    # target -- purely visual hint, no semantic weight.
    max_ell = n * (n - 1) // 2
    ax_ell.axhline(max_ell, color="#888888", lw=0.8, ls=":", label=r"$\binom{n}{2}$")

    ax_ell.set_xlabel("MCMC step")
    ax_ell.set_ylabel(r"$\ell(w) = \#\text{crosses}$")
    ax_ell.set_title("Length $\\ell(w)$ along the chain")

    ax_corner.set_xlabel("MCMC step")
    ax_corner.set_ylabel(r"$\#\{i : i + w(i) > 2n - n/4\}$")
    ax_corner.set_title("Southeast-corner proxy")

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

    # Single legend across both panels.
    handles, labels = ax_ell.get_legend_handles_labels()
    # Move the dotted binom line to the end so the start legend reads top-down.
    ordered: list = []
    for h, lb in zip(handles, labels):
        if lb.startswith(r"$\binom"):
            continue
        ordered.append((h, lb))
    for h, lb in zip(handles, labels):
        if lb.startswith(r"$\binom"):
            ordered.append((h, lb))
            break
    fig.legend(
        [h for h, _ in ordered],
        [lb for _, lb in ordered],
        loc="lower center",
        ncol=min(len(ordered), 5),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Multi-start BPD MCMC validation, $n={n}$ "
        f"(bands: min/max over {max(n_seeds_seen) if n_seeds_seen else 0} seeds)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    # Suppress Matplotlib's default /CreationDate so rebuilds from the same
    # committed CSV are byte-identical. Without this the PDF differs only in
    # its embedded timestamp on every invocation.
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs-dir", type=Path, default=None, help="directory of <start>_s<seed>/ subdirs (primary input)")
    p.add_argument("--csv-in", type=Path, default=None, help="rebuild figure from an existing trace CSV (no --runs-dir)")
    p.add_argument("--n", type=int, required=True, help="grid size n (must match bpd_mcmc runs)")
    p.add_argument("--out-csv", type=Path, default=None, help="output multistart_n<N>_trace.csv (only with --runs-dir)")
    p.add_argument("--out-pdf", type=Path, required=True, help="output multistart_n<N>.pdf")
    args = p.parse_args(argv)

    if (args.runs_dir is None) == (args.csv_in is None):
        p.error("exactly one of --runs-dir or --csv-in must be provided")

    if args.runs_dir is not None:
        traces = collect_runs(args.runs_dir, args.n)
        if not traces:
            print(f"error: no usable burn-in traces under {args.runs_dir}", file=sys.stderr)
            return 1
        if args.out_csv is not None:
            write_csv(args.out_csv, traces)
    else:
        traces = collect_from_csv(args.csv_in)
        if not traces:
            print(f"error: no rows in {args.csv_in}", file=sys.stderr)
            return 1
        if args.out_csv is not None:
            print("warn: --out-csv ignored in --csv-in rebuild mode", file=sys.stderr)

    plot_traces(args.out_pdf, traces, args.n)

    # Summary print for the driver's log
    for start in STARTS_ORDER:
        if start not in traces:
            continue
        seeds = sorted(traces[start])
        sizes = {s: traces[start][s].shape[0] for s in seeds}
        print(f"{start:10s} seeds={seeds} trace_points={sizes}")
    if args.out_csv is not None and args.runs_dir is not None:
        print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
