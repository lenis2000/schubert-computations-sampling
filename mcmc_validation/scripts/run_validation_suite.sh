#!/usr/bin/env bash
# Multi-start diagnostic run for the MCMC validation dossier (Task 5 of the
# PNAS SI validation plan). Drives code/bpd_mcmc from four starting states
# (b_w0, b_id, Rothe of the MPP3 layered optimum, random RBPD) with identical
# burn-in and collection settings, across a handful of independent seeds per
# start. Aggregates each chain's (step, ell, corner) burn-in trace and emits
# multistart_n<N>_trace.csv plus the two-panel trace figure
# multistart_n<N>.pdf (ell and southeast-corner-count vs. MCMC steps).
#
# The corner statistic #{i : i + w(i) > 2n - n/4} is the southeast-corner
# proxy produced natively by bpd_mcmc and serves as the "SE-boundary proxy"
# called for by the plan. It is a coarse but fast scalar that concentrates
# around 0 for well-mixed RBPDs.
#
# Defaults target n=60; all parameters are overridable via environment
# variables so the same script can be used for the SI figure and for Leo's
# longer production sweeps. HARD UPPER BOUND: n = 60. If you need n = 100,
# use rivanna_validation_n100.slurm (not this script).
#
# Usage:
#   bash run_validation_suite.sh
#   N=60 BURNIN=1G SEEDS="5000 5001 5002" bash run_validation_suite.sh
#
# Environment variables:
#   N              grid size (default 60; must be <= 60)
#   BURNIN         burn-in step count, K/M/G suffixes (default 10M for the
#                  smoke-test scale; use 1G or 10G for the production figure)
#   THIN           thinning interval for the throwaway collect phase (1M)
#   B              number of samples to collect per chain (1 -- we only want
#                  the burn-in trace; collection is just to trigger cleanup)
#   SEEDS          space-separated seed list (default "5000 5001 5002")
#   DROOP_DIST     droop rectangle size distribution (default geometric)
#   ANCHOR         droop anchor corner (default se)
#   OUTDIR         root output directory (default
#                  PNAS/mcmc_validation/data/multistart_n<N>_runs)
#   DATA_DIR       where the aggregated CSV lands (default
#                  PNAS/mcmc_validation/data)
#   IMG_DIR        where the aggregated PDF lands (default
#                  PNAS/mcmc_validation/img)
#   ROTHE_CSV      comma-separated Rothe start permutation (default: MPP3
#                  layered optimum for n=60, which is w(1,3,6,15,35))
#   MCMC_BIN       path to the bpd_mcmc binary (default code/bpd_mcmc)
#   PLOT_SCRIPT    path to the aggregation/plot script (default next to
#                  this script)
#
# Artifacts written:
#   $OUTDIR/<start>_s<seed>/ (one per chain; keeps bpd_mcmc's raw outputs)
#   $DATA_DIR/multistart_n<N>_trace.csv
#   $IMG_DIR/multistart_n<N>.pdf

set -euo pipefail

# --- Resolve paths ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

N="${N:-60}"
if (( N > 60 )); then
    echo "Error: N=$N > 60. This script is capped at n=60 by the validation plan." >&2
    echo "       For n=100 use PNAS/mcmc_validation/scripts/rivanna_validation_n100.slurm." >&2
    exit 1
fi

BURNIN="${BURNIN:-10M}"
THIN="${THIN:-1M}"
B="${B:-1}"
SEEDS="${SEEDS:-5000 5001 5002}"
DROOP_DIST="${DROOP_DIST:-geometric}"
ANCHOR="${ANCHOR:-se}"

# THREADS is intentionally fixed at 1. The aggregation pipeline assumes one
# chain per (start, seed) rundir: bpd_mcmc's batch mode only exports thread 0's
# burn-in trace, so any THREADS>1 value would silently burn extra chains and
# discard their traces -- the resulting CSV/PDF would no longer match the
# documented experiment. Reject caller overrides instead of silently accepting.
if [[ -n "${THREADS:-}" && "${THREADS}" != "1" ]]; then
    echo "Error: THREADS=$THREADS is not supported by this script." >&2
    echo "       bpd_mcmc batch mode only exports thread 0's burn-in trace, so" >&2
    echo "       the multistart aggregation requires one chain per (start, seed)." >&2
    echo "       Run separate seeds instead of parallel chains inside one invocation." >&2
    exit 1
fi
THREADS=1

OUTDIR="${OUTDIR:-$REPO_ROOT/PNAS/mcmc_validation/data/multistart_n${N}_runs}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/PNAS/mcmc_validation/data}"
IMG_DIR="${IMG_DIR:-$REPO_ROOT/PNAS/mcmc_validation/img}"
MCMC_BIN="${MCMC_BIN:-$REPO_ROOT/code/bpd_mcmc}"
PLOT_SCRIPT="${PLOT_SCRIPT:-$SCRIPT_DIR/plot_multistart_trace.py}"

# When ROTHE_CSV isn't supplied, delegate to measure_acceptance_rates.py so
# the MPP3 layered optima live in exactly one place. That module raises if
# the requested n has no tabulated optimum, which we surface here instead
# of silently falling back to a mis-labeled ad-hoc split.
if [[ -z "${ROTHE_CSV:-}" ]]; then
    if ! ROTHE_CSV="$(python3 - "$REPO_ROOT/code" "$N" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from measure_acceptance_rates import (
    LAYERED_OPTIMUM_BY_N, build_layered_permutation,
)
n = int(sys.argv[2])
if n not in LAYERED_OPTIMUM_BY_N:
    known = ", ".join(str(k) for k in sorted(LAYERED_OPTIMUM_BY_N))
    print(f"no MPP3 layered optimum tabulated for n={n}; "
          f"known n are {known}", file=sys.stderr)
    sys.exit(2)
w = build_layered_permutation(LAYERED_OPTIMUM_BY_N[n])
print(",".join(str(v) for v in w))
PYEOF
)"; then
        echo "Error: could not derive a default Rothe start for N=$N." >&2
        echo "       Pass ROTHE_CSV=<csv> explicitly." >&2
        exit 1
    fi
fi

# Validate ROTHE_CSV upfront (whether caller-supplied or auto-derived) so an
# invalid permutation fails fast, before any chains launch. Without this guard
# a bad csv surfaces only on the third leg (rothe) after w0 and identity have
# already consumed hours of wall-clock.
if ! python3 - "$N" "$ROTHE_CSV" <<'PYEOF' >/dev/null
import sys
n = int(sys.argv[1])
csv = sys.argv[2].strip()
try:
    perm = [int(x) for x in csv.split(",") if x != ""]
except ValueError as exc:
    print(f"ROTHE_CSV must be a comma-separated list of integers: {exc}",
          file=sys.stderr)
    sys.exit(2)
if len(perm) != n:
    print(f"ROTHE_CSV has {len(perm)} entries but n={n}", file=sys.stderr)
    sys.exit(2)
if sorted(perm) != list(range(1, n + 1)):
    print(f"ROTHE_CSV must be a permutation of 1..{n}; got {perm}",
          file=sys.stderr)
    sys.exit(2)
PYEOF
then
    echo "Error: ROTHE_CSV=\"$ROTHE_CSV\" is not a valid permutation of 1..$N." >&2
    exit 1
fi

mkdir -p "$OUTDIR" "$DATA_DIR" "$IMG_DIR"

if [[ ! -x "$MCMC_BIN" ]]; then
    echo "Error: bpd_mcmc binary not found or not executable at $MCMC_BIN" >&2
    echo "       Build it first (see CLAUDE.md) or set MCMC_BIN=/path/to/bpd_mcmc." >&2
    exit 1
fi

# --- Run matrix ------------------------------------------------------------
STARTS=("w0" "identity" "rothe" "random")

# --- Manifest guard --------------------------------------------------------
# The aggregator (plot_multistart_trace.py) picks the newest burn-in trace
# file in each rundir. Without a guard, a parameter change (BURNIN, THIN,
# DROOP_DIST, ANCHOR, seed set) or a partial-retry leftover could silently
# mix old and new traces into one CSV/PDF. Pin the parameter set for OUTDIR
# in a manifest and fail loudly if an existing OUTDIR was produced with a
# different configuration.
MANIFEST="$OUTDIR/run_manifest.txt"
CURRENT_MANIFEST="n=$N
burnin=$BURNIN
thin=$THIN
droop_dist=$DROOP_DIST
anchor=$ANCHOR
starts=${STARTS[*]}
seeds=$SEEDS"

if [[ -f "$MANIFEST" ]]; then
    if [[ "$(cat "$MANIFEST")" != "$CURRENT_MANIFEST" ]]; then
        echo "Error: OUTDIR $OUTDIR was previously populated with different parameters." >&2
        echo "       Existing manifest:" >&2
        sed 's/^/         /' "$MANIFEST" >&2
        echo "       Current parameters:" >&2
        printf '%s\n' "$CURRENT_MANIFEST" | sed 's/^/         /' >&2
        echo "       Delete $OUTDIR or pass OUTDIR=<fresh path> to isolate the new sweep." >&2
        exit 1
    fi
fi

# Drop stale rundirs not in the current start x seed matrix so the plotter
# can't pick up orphaned traces from a shrunk seed list.
for existing_sub in "$OUTDIR"/*/; do
    [[ -d "$existing_sub" ]] || continue
    subname="$(basename "$existing_sub")"
    keep=0
    for start in "${STARTS[@]}"; do
        for seed in $SEEDS; do
            if [[ "$subname" == "${start}_s${seed}" ]]; then
                keep=1
                break 2
            fi
        done
    done
    if (( !keep )); then
        echo "    removing stale rundir outside current matrix: $existing_sub" >&2
        rm -rf "$existing_sub"
    fi
done

echo "=== Multi-start validation suite (n=$N) ==="
echo "    burnin=$BURNIN  thin=$THIN  B=$B  droop_dist=$DROOP_DIST  anchor=$ANCHOR"
echo "    seeds: $SEEDS"
echo "    outdir: $OUTDIR"
run_one() {
    local start_label="$1"
    local seed="$2"
    local start_spec
    case "$start_label" in
        w0)       start_spec="w0" ;;
        identity) start_spec="identity" ;;
        rothe)    start_spec="rothe:$ROTHE_CSV" ;;
        random)   start_spec="random" ;;
        *) echo "Error: unknown start $start_label" >&2; return 1 ;;
    esac

    local rundir="$OUTDIR/${start_label}_s${seed}"
    # Wipe and recreate so a retry after a failed run or a rerun with new
    # parameters can't leave a stale *_burnin_trace_* file that the plotter
    # would pick up (it selects the newest matching file in each rundir).
    rm -rf "$rundir"
    mkdir -p "$rundir"
    echo "--> ${start_label} seed=${seed}  ($rundir)"

    # Short label in run manifest to let the plotter tie files back to starts.
    (cd "$rundir" && "$MCMC_BIN" "batch:${N}:${B}" \
        --droop-dist "$DROOP_DIST" \
        --anchor "$ANCHOR" \
        --start "$start_spec" \
        --burnin "$BURNIN" \
        --thin "$THIN" \
        --threads "$THREADS" \
        --seed "$seed" \
        --no-png --no-tikz --no-height \
        > run.log 2>&1) || {
        echo "Error: bpd_mcmc failed for start=$start_label seed=$seed" >&2
        tail -20 "$rundir/run.log" >&2
        return 1
    }
}

for start_label in "${STARTS[@]}"; do
    for seed in $SEEDS; do
        run_one "$start_label" "$seed"
    done
done

# Record the exact parameter set so future reruns can detect mismatches.
printf '%s\n' "$CURRENT_MANIFEST" > "$MANIFEST"

# --- Aggregate & plot ------------------------------------------------------
echo
echo "=== Aggregating traces -> CSV + PDF ==="
python3 "$PLOT_SCRIPT" \
    --runs-dir "$OUTDIR" \
    --n "$N" \
    --out-csv "$DATA_DIR/multistart_n${N}_trace.csv" \
    --out-pdf "$IMG_DIR/multistart_n${N}.pdf"

echo
echo "All done. Artifacts:"
echo "  $DATA_DIR/multistart_n${N}_trace.csv"
echo "  $IMG_DIR/multistart_n${N}.pdf"
