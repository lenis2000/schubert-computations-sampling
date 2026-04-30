#!/bin/bash
# Certify a single (permutation, method) with adaptive memo cap / threads.
#
# Slurm scripts receive positional args from sbatch as $1, $2, ...
# so we avoid --export (which uses commas as list separators and breaks
# on permutation strings like "1,3,2,...").
#
# Usage:
#   sbatch [--cpus-per-task=N --mem=MG] rivanna_certify_single.slurm \
#       <one-line-permutation>  <cotrans|transition>  <outname>
#
# Example:
#   sbatch --cpus-per-task=16 --mem=96G rivanna_certify_single.slurm \
#       '1,3,2,8,7,6,5,20,4,19,18,17,16,15,14,13,12,11,10,9' \
#       cotrans  n20_counterex_cotrans
#
#SBATCH --job-name=certify
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=certify_%x_%j.out
#SBATCH --account=intprob

set -euo pipefail

# -------- Validate positional inputs --------
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: sbatch $0 <PERM> <METHOD> <OUTNAME> [CAP_LOG2_OVERRIDE]" >&2
    echo "  got $# args: $*" >&2
    exit 2
fi
PERM="$1"
METHOD="$2"
OUTNAME="$3"
CAP_OVERRIDE="${4:-}"
case "$METHOD" in
    cotrans|transition|descent) ;;
    *)
        echo "error: METHOD must be cotrans, transition, or descent (got: $METHOD)" >&2
        exit 2
        ;;
esac

module load gcc/14.2.0 gmp

cd "$SLURM_SUBMIT_DIR"

# -------- Node profile --------
echo "=== Node profile ==="
echo "  hostname:        $(hostname)"
echo "  partition:       ${SLURM_JOB_PARTITION:-n/a}"
echo "  cpus-per-task:   ${SLURM_CPUS_PER_TASK:-1}"
echo "  mem-per-node:    ${SLURM_MEM_PER_NODE:-?} MB"
echo "  /proc/cpuinfo:   $(grep -c ^processor /proc/cpuinfo) procs, $(lscpu | awk -F: '/Model name/ {gsub(/^ +/, "", $2); print $2; exit}')"
echo "  /proc/meminfo:   $(awk '/MemTotal/ {printf "%.1f GB total\n", $2/1024/1024}' /proc/meminfo)"

# -------- Adaptive MEMO_HARD_CAP --------
# Each exact memo entry is ~120 B (key + mpz_class value + hash-table overhead).
# Use largest power of 2 that fits in ~(MEM - 8G) at 120 B/entry.
MEM_MB="${SLURM_MEM_PER_NODE:-49152}"
USABLE_MB=$(( MEM_MB - 8192 ))
if [[ $USABLE_MB -lt 4096 ]]; then USABLE_MB=4096; fi
# MAX_ENTRIES = USABLE_MB * 1024 * 1024 / 120
LOG2=$(awk -v mb="$USABLE_MB" 'BEGIN{ x=mb*1024.0*1024.0/120.0; print int(log(x)/log(2)) }')
# Clamp LOG2 to [27, 32]: never below shipped 2^27, never above 2^32 (unreasonable).
if [[ $LOG2 -lt 27 ]]; then LOG2=27; fi
if [[ $LOG2 -gt 32 ]]; then LOG2=32; fi
# Optional hand-override (4th positional arg): trades memo size for frontier headroom.
if [[ -n "$CAP_OVERRIDE" ]]; then
    echo "  override:        LOG2 $LOG2 -> $CAP_OVERRIDE (CAP_LOG2_OVERRIDE set)"
    LOG2=$CAP_OVERRIDE
fi
echo "  adaptive cap:    2^${LOG2} entries (~$(( (1 << LOG2) * 120 / 1024 / 1024 / 1024 )) GB at 120 B/entry)"

# -------- Per-job scratch dir so parallel jobs don't race on filenames --------
JOB_SCRATCH="${SLURM_SUBMIT_DIR}/job_${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_SCRATCH"
cp "${SLURM_SUBMIT_DIR}/schubert.cpp" "$JOB_SCRATCH/schubert_adapt.cpp"
cd "$JOB_SCRATCH"

# -------- Patch the shipped source for this node --------
sed -i "s/static const size_t MEMO_HARD_CAP = 1ULL << 27;/static const size_t MEMO_HARD_CAP = 1ULL << ${LOG2};/" schubert_adapt.cpp
sed -i 's|#include <boost/multiprecision/cpp_int.hpp>|// boost replaced by mpz_class alias for Rivanna|' schubert_adapt.cpp
sed -i 's|using boost::multiprecision::cpp_int;|using cpp_int = mpz_class;|' schubert_adapt.cpp
sed -i 's|\.convert_to<std::string>()|.get_str()|g' schubert_adapt.cpp
sed -i 's|\.convert_to<double>()|.get_d()|g' schubert_adapt.cpp
sed -i 's|lr\.exact_value\.str()|lr.exact_value.get_str()|g' schubert_adapt.cpp
grep -E 'MEMO_HARD_CAP = ' schubert_adapt.cpp | head -1

echo "=== Compiling in $JOB_SCRATCH ==="
g++ -O3 -march=native -mtune=native -flto -ffast-math \
    -funroll-loops -fomit-frame-pointer -DNDEBUG \
    -std=c++17 -pthread -fopenmp \
    schubert_adapt.cpp -o schubert_adapt -lgmp -lgmpxx || exit 1

# -------- Run --------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "=== Certification: ${OUTNAME} (method=${METHOD}, OMP_NUM_THREADS=$OMP_NUM_THREADS) ==="
echo "    PERM=${PERM}"
/usr/bin/time -v ./schubert_adapt --exact --${METHOD} "$PERM" > "${OUTNAME}.log" 2>&1
RC=$?
echo "=== ${OUTNAME} done: exit=${RC} ==="
grep -E 'Result:|S_w\s*=' "${OUTNAME}.log" || echo "(no Result line)"

exit $RC
