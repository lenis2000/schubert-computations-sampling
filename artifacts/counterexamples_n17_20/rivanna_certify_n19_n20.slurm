#!/bin/bash
#SBATCH --job-name=certify_n19_n20
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=certify_n19_n20_%j.out
#SBATCH --account=intprob

# Certifies the counterexamples at n=19 (transition only) and n=20
# (both cotransition and transition) in exact GMP arithmetic.
# n=17 and n=18 are already certified locally (see artifacts/
# counterexamples_n17_20/logs/) and are NOT re-run here.
#
# After this job completes, scp the six .log files back into
# artifacts/counterexamples_n17_20/logs/ and rerun
# build_certificate_csv.py (when it exists).

set -euo pipefail

module load gcc/14.2.0 gmp

cd "$SLURM_SUBMIT_DIR"

# -------- Patch the shipped source for Rivanna --------
# (1) Bump the memoization cap to 256M entries (the shipped 128M was
#     tuned for a MacBook with 16GB RAM; on this 48GB node we can
#     afford 1<<28 ~= 25GB exact-arith RAM).
# (2) Alias boost::multiprecision::cpp_int to GMP's mpz_class.
#     Rivanna does not ship boost as a module, but cpp_int is only used
#     for the layered-product code path (dead code for the
#     counterexample certification runs here); the APIs are
#     compatible for the basic arithmetic the layered path needs.
cp schubert.cpp schubert_rivanna.cpp
sed -i 's/static const size_t MEMO_HARD_CAP = 1ULL << 27;/static const size_t MEMO_HARD_CAP = 1ULL << 28;/' schubert_rivanna.cpp
sed -i 's|#include <boost/multiprecision/cpp_int.hpp>|// boost replaced by mpz_class alias for Rivanna (no boost module)|' schubert_rivanna.cpp
sed -i 's|using boost::multiprecision::cpp_int;|using cpp_int = mpz_class;|' schubert_rivanna.cpp
# mpz_class spells .convert_to<T>() as .get_str() / .get_d() etc.
sed -i 's|\.convert_to<std::string>()|.get_str()|g' schubert_rivanna.cpp
sed -i 's|\.convert_to<double>()|.get_d()|g' schubert_rivanna.cpp
# cpp_int also spells .str() as .get_str() on mpz_class
sed -i 's|lr\.exact_value\.str()|lr.exact_value.get_str()|g' schubert_rivanna.cpp
grep 'MEMO_HARD_CAP = ' schubert_rivanna.cpp | head -1
grep -nE 'cpp_int|boost|convert_to' schubert_rivanna.cpp | head -5

# -------- Compile for Rivanna (AMD EPYC) --------
echo "Compiling schubert_rivanna..."
g++ -O3 -march=native -mtune=native -flto -ffast-math \
    -funroll-loops -fomit-frame-pointer -DNDEBUG \
    -std=c++17 -pthread -fopenmp \
    schubert_rivanna.cpp -o schubert_rivanna -lgmp -lgmpxx || exit 1

echo "=== Certification: n=19 transition ==="
# n=19 counterexample s_8 * w(1,2,5,11)
PERM19='1,3,2,8,7,6,5,19,4,18,17,16,15,14,13,12,11,10,9'
./schubert_rivanna --exact --transition "$PERM19" > n19_counterex_transition.log 2>&1
echo "n=19 transition: exit $?"
grep -E 'Result:|S_w\s*=' n19_counterex_transition.log || true

echo "=== Certification: n=20 cotransition ==="
# n=20 counterexample s_8 * w(1,2,5,12)
PERM20='1,3,2,8,7,6,5,20,4,19,18,17,16,15,14,13,12,11,10,9'
./schubert_rivanna --exact --cotrans "$PERM20" > n20_counterex_cotrans.log 2>&1
echo "n=20 cotrans: exit $?"
grep -E 'Result:|S_w\s*=' n20_counterex_cotrans.log || true

echo "=== Certification: n=20 transition ==="
./schubert_rivanna --exact --transition "$PERM20" > n20_counterex_transition.log 2>&1
echo "n=20 transition: exit $?"
grep -E 'Result:|S_w\s*=' n20_counterex_transition.log || true

echo "=== Summary ==="
for f in n19_counterex_transition.log n20_counterex_cotrans.log n20_counterex_transition.log; do
    printf '%-40s ' "$f"
    grep -E 'Result:|S_w\s*=' "$f" | head -1 || echo "(no result)"
done

echo "Done."
