#!/bin/bash
set -euo pipefail
#SBATCH --job-name=bpd_validate_n100
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=bpd_validate_n100_%j.out
#SBATCH --account=intprob
#
# Multi-start MCMC validation run at n=100 (Rivanna-only).
# This file is the n=100 analogue of PNAS/mcmc_validation/scripts/run_validation_suite.sh.
# It is intentionally NOT submitted by the validation-plan loop -- Leo sbatches it by hand.
#
# Matrix: 4 starts (w0, identity, rothe, random) x 3 seeds = 12 independent
# chains, each run single-threaded so one invocation produces one burn-in
# trace. All 12 chains are launched inside a single Slurm job via xargs so
# they share the 40 allocated cores.
#
# After the job finishes, aggregate the outputs on the cluster or rsync the
# run dirs back and run:
#   python3 PNAS/mcmc_validation/scripts/plot_multistart_trace.py \
#       --runs-dir PNAS/mcmc_validation/data/multistart_n100_runs \
#       --n 100 \
#       --out-csv PNAS/mcmc_validation/data/multistart_n100_trace.csv \
#       --out-pdf PNAS/mcmc_validation/img/multistart_n100.pdf

module load gcc/14.2.0 libpng

cd "$SLURM_SUBMIT_DIR"

N=100
BURNIN="${BURNIN:-10G}"
THIN="${THIN:-500M}"
B="${B:-1}"
DROOP_DIST="${DROOP_DIST:-geometric}"
ANCHOR="${ANCHOR:-se}"
SEEDS="${SEEDS:-9100 9101 9102}"

# MPP3 layered optimum at n=100 is w(1, 2, 4, 11, 25, 57) (data_MPP3.tex:1251).
# Concatenate reversed blocks exactly like
# code/measure_acceptance_rates.py::build_layered_permutation.
ROTHE_CSV_N100="1,\
3,2,\
7,6,5,4,\
18,17,16,15,14,13,12,11,10,9,8,\
43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,\
100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,\
75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,\
50,49,48,47,46,45,44"
ROTHE_CSV_N100="$(echo "$ROTHE_CSV_N100" | tr -d ' ' | tr -d '\n')"

RUN_ROOT="PNAS/mcmc_validation/data/multistart_n${N}_runs"
mkdir -p "$RUN_ROOT"

STARTS=(w0 identity rothe random)

# Manifest guard + stale-dir cleanup: mirror run_validation_suite.sh so that a
# rerun with a different seed set (or left-behind rundirs from an earlier
# sweep) can't quietly contaminate multistart_n${N}_trace.csv / the PDF. The
# aggregator (plot_multistart_trace.py) takes the newest burn-in trace in each
# <start>_s<seed> subdir under RUN_ROOT, so any stale subdir outside the
# current matrix is a silent mixing hazard.
MANIFEST="$RUN_ROOT/run_manifest.txt"
CURRENT_MANIFEST="n=$N
burnin=$BURNIN
thin=$THIN
droop_dist=$DROOP_DIST
anchor=$ANCHOR
starts=${STARTS[*]}
seeds=$SEEDS"

if [[ -f "$MANIFEST" ]]; then
    if [[ "$(cat "$MANIFEST")" != "$CURRENT_MANIFEST" ]]; then
        echo "Error: RUN_ROOT $RUN_ROOT was previously populated with different parameters." >&2
        echo "       Existing manifest:" >&2
        sed 's/^/         /' "$MANIFEST" >&2
        echo "       Current parameters:" >&2
        printf '%s\n' "$CURRENT_MANIFEST" | sed 's/^/         /' >&2
        echo "       Delete $RUN_ROOT or point RUN_ROOT at a fresh path to isolate the new sweep." >&2
        exit 1
    fi
fi

# Drop stale rundirs outside the current start x seed matrix so the plotter
# can't pick up orphaned traces.
for existing_sub in "$RUN_ROOT"/*/; do
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

# Compile inside the job so the binary is tuned to the allocated node.
echo "Compiling bpd_mcmc for Rivanna..."
g++ -O3 -std=c++17 -march=native -flto -fopenmp \
    code/bpd_mcmc.cpp -o code/bpd_mcmc -lpng || {
    echo "Compile failed" >&2
    exit 1
}

run_one() {
    local start_label="$1"
    local seed="$2"
    local start_spec
    case "$start_label" in
        w0)       start_spec="w0" ;;
        identity) start_spec="identity" ;;
        rothe)    start_spec="rothe:$ROTHE_CSV_N100" ;;
        random)   start_spec="random" ;;
    esac
    local rundir="$RUN_ROOT/${start_label}_s${seed}"
    # Wipe and recreate so a retry after a failed run or a rerun with new
    # parameters can't leave a stale *_burnin_trace_* file that the plotter
    # would pick up (it selects the newest matching file in each rundir).
    rm -rf "$rundir"
    mkdir -p "$rundir"
    echo "=== ${start_label} seed=${seed} ==="
    (cd "$rundir" && ../../../../../code/bpd_mcmc "batch:${N}:${B}" \
        --droop-dist "$DROOP_DIST" \
        --anchor "$ANCHOR" \
        --start "$start_spec" \
        --burnin "$BURNIN" \
        --thin "$THIN" \
        --threads 1 \
        --seed "$seed" \
        --no-png --no-tikz --no-height) 2>&1 | sed "s|^|  [${start_label} s${seed}] |"
}

export -f run_one
export N BURNIN THIN B DROOP_DIST ANCHOR RUN_ROOT ROTHE_CSV_N100

# Launch all chains concurrently; 40 cores -> easily fits single-threaded chains.
{
    for start in "${STARTS[@]}"; do
        for seed in $SEEDS; do
            echo "$start $seed"
        done
    done
} | xargs -n 2 -P 12 bash -c 'run_one "$0" "$1"'

# Record the exact parameter set so future reruns can detect mismatches.
printf '%s\n' "$CURRENT_MANIFEST" > "$MANIFEST"

echo
echo "All chains done. Aggregate on your workstation with:"
echo "  python3 PNAS/mcmc_validation/scripts/plot_multistart_trace.py \\"
echo "      --runs-dir $RUN_ROOT \\"
echo "      --n $N \\"
echo "      --out-csv PNAS/mcmc_validation/data/multistart_n${N}_trace.csv \\"
echo "      --out-pdf PNAS/mcmc_validation/img/multistart_n${N}.pdf"
