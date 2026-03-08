// CFTP Diagnostic Tool - checks whether bounding chain ordering is maintained
// and whether CFTP output distribution matches uniform over RBPDs.
//
// Compile: clang++ -O3 -std=c++17 cftp_diagnostic.cpp -o cftp_diagnostic
//
// Usage:
//   ./cftp_diagnostic ordering <n> <num_samples>    Check h_b0 <= h_b1 at every step
//   ./cftp_diagnostic freq <n> <num_samples>        Frequency test vs uniform

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <numeric>

// Tile definitions: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz

// =============================================================================
// Height function computation
// =============================================================================

// Compute height function on dual-lattice vertices (i,j), 0 <= i,j <= n.
// Convention: six-vertex lines are contour lines of h.
// Moving right across an edge:
//   line present -> +1, line absent -> +0.
// Boundary normalization: h = 0 on top and left boundaries.
void compute_height(const std::vector<uint8_t>& grid, int n, std::vector<int>& h) {
    h.assign((n + 1) * (n + 1), 0);
    auto idx = [n](int i, int j) { return i * (n + 1) + j; };

    for (int i = 0; i <= n; i++) {
        int crossings = 0;
        for (int j = 0; j < n; j++) {
            h[idx(i, j)] = crossings;
            bool has_vertical_line = false;
            if (i == 0) {
                has_vertical_line = false;
            } else if (i == n) {
                has_vertical_line = true;
            } else {
                uint8_t t = grid[(i - 1) * n + j];
                has_vertical_line = (t == 1 || t == 2 || t == 4);
            }
            if (has_vertical_line) crossings++;
        }
        h[idx(i, n)] = crossings;
    }
}

// Order in this normalization: state0 <= state1 iff h0[v] >= h1[v] pointwise.
// Returns true if ordering holds, false if violated.
// If violated, sets (vi, vj) to the first violating vertex.
bool check_ordering(const std::vector<int>& h0, const std::vector<int>& h1,
                    int n, int& vi, int& vj) {
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            if (h0[i*(n+1)+j] < h1[i*(n+1)+j]) {
                vi = i; vj = j;
                return false;
            }
        }
    }
    return true;
}

void print_height(const std::vector<int>& h, int n) {
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            printf("%3d ", h[i*(n+1)+j]);
        }
        printf("\n");
    }
}

// =============================================================================
// BPD Engine (core logic from bpd_sampler.cpp, with height check instrumentation)
// =============================================================================

class BPDEngine {
    int n;
    std::vector<uint8_t> b0;  // Chain from identity (min)
    std::vector<uint8_t> b1;  // Chain from w0 (max)
    std::mt19937 rng;

    std::vector<uint8_t> temp0, temp1;

    // Height function buffers
    std::vector<int> h0, h1;

public:
    // Diagnostic counters
    int64_t total_steps = 0;
    int64_t ordering_violations = 0;
    int64_t moves_applied = 0;        // steps where at least one chain moved
    int64_t cross_moves = 0;          // moves involving crosses
    int64_t cross_rejections_b0 = 0;  // cross moves rejected on b0 (reducedness)
    int64_t cross_rejections_b1 = 0;  // cross moves rejected on b1
    int64_t asymmetric_rejections = 0; // move accepted on one chain, rejected on other
    int64_t asymmetric_acceptance = 0; // move geometrically possible on one, not other

    BPDEngine(int size) : n(size) {
        b0.resize(n * n);
        b1.resize(n * n);
        temp0.resize(n * n);
        temp1.resize(n * n);
    }

    void seed(uint32_t s) { rng.seed(s); }
    int get_n() const { return n; }

    inline uint8_t get(const std::vector<uint8_t>& b, int r, int c) const {
        return b[r * n + c];
    }
    inline void set(std::vector<uint8_t>& b, int r, int c, uint8_t val) {
        b[r * n + c] = val;
    }

    void generateRotheBPD(std::vector<uint8_t>& result, const std::vector<int>& w) {
        int minW = *std::min_element(w.begin(), w.end());
        for (int j = 1; j <= n; j++) {
            if (j + minW - 1 < w[0]) set(result, 0, j - 1, 0);
            else if (j + minW - 1 == w[0]) set(result, 0, j - 1, 2);
            else set(result, 0, j - 1, 5);
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (j + minW - 1 < w[i - 1]) {
                    uint8_t up = get(result, i - 2, j - 1);
                    set(result, i - 1, j - 1, (up == 2 || up == 4 || up == 1) ? 4 : 0);
                } else if (j + minW - 1 == w[i - 1]) {
                    set(result, i - 1, j - 1, 2);
                } else {
                    uint8_t up = get(result, i - 2, j - 1);
                    set(result, i - 1, j - 1, (up == 4 || up == 2 || up == 1) ? 1 : 5);
                }
            }
        }
    }

    void reset() {
        std::vector<int> identity(n), w0(n);
        for (int i = 0; i < n; i++) { identity[i] = i + 1; w0[i] = n - i; }
        generateRotheBPD(b0, identity);
        generateRotheBPD(b1, w0);
    }

    int get_sk_from_cross(const std::vector<uint8_t>& grid, int i, int j) const {
        if (get(grid, i, j) != 1) return -1;
        int k = 1;
        int maxS = std::min(i, j);
        for (int s = 1; s <= maxS; s++) {
            uint8_t t = get(grid, i - s, j - s);
            if (t == 2 || t == 3 || t == 4 || t == 5) k += 1;
            else if (t == 1) k += 2;
        }
        return k;
    }

    bool bpd2perm_check(const std::vector<uint8_t>& grid) const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;
        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1, j = n + s - t - 1;
                int k = get_sk_from_cross(grid, i, j);
                if (k != -1) { if (ww[k-1] > ww[k]) return false; std::swap(ww[k-1], ww[k]); }
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1, j = n - t - 1;
                int k = get_sk_from_cross(grid, i, j);
                if (k != -1) { if (ww[k-1] > ww[k]) return false; std::swap(ww[k-1], ww[k]); }
            }
        }
        return true;
    }

    std::vector<int> compute_perm(const std::vector<uint8_t>& grid) const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;
        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1, j = n + s - t - 1;
                int k = get_sk_from_cross(grid, i, j);
                if (k != -1 && k < n) std::swap(ww[k-1], ww[k]);
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1, j = n - t - 1;
                int k = get_sk_from_cross(grid, i, j);
                if (k != -1 && k < n) std::swap(ww[k-1], ww[k]);
            }
        }
        return ww;
    }

    // --- Local moves (same as bpd_sampler.cpp) ---

    inline bool try_drip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i-1, j-1), ne = get(grid, i-1, j);
        uint8_t sw = get(grid, i, j-1), se = get(grid, i, j);
        if (nw == 2 && se == 0) {
            set(grid, i-1, j-1, 0); set(grid, i, j, 3);
            set(grid, i-1, j, (ne == 5) ? 2 : 4);
            set(grid, i, j-1, (sw == 4) ? 2 : 5);
            return true;
        }
        return false;
    }

    inline bool try_undrip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i-1, j-1), ne = get(grid, i-1, j);
        uint8_t sw = get(grid, i, j-1), se = get(grid, i, j);
        if (nw == 0 && se == 3 && (ne == 2 || ne == 4) && (sw == 2 || sw == 5)) {
            set(grid, i-1, j-1, 2); set(grid, i, j, 0);
            set(grid, i-1, j, (ne == 2) ? 5 : 3);
            set(grid, i, j-1, (sw == 2) ? 4 : 3);
            return true;
        }
        return false;
    }

    inline bool try_cross_drip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i-1, j-1), ne = get(grid, i-1, j);
        uint8_t sw = get(grid, i, j-1), se = get(grid, i, j);
        if (nw == 2 && se == 2) {
            set(grid, i-1, j-1, 0); set(grid, i, j, 1);
            set(grid, i-1, j, (ne == 3) ? 4 : 2);
            set(grid, i, j-1, (sw == 3) ? 5 : 2);
            return true;
        }
        if (nw == 1 && se == 0) {
            set(grid, i-1, j-1, 3); set(grid, i, j, 3);
            set(grid, i-1, j, (ne == 3) ? 4 : 2);
            set(grid, i, j-1, (sw == 3) ? 5 : 2);
            return true;
        }
        if (nw == 1 && se == 2) {
            set(grid, i-1, j-1, 3); set(grid, i, j, 1);
            set(grid, i-1, j, (ne == 3) ? 4 : 2);
            set(grid, i, j-1, (sw == 3) ? 5 : 2);
            return true;
        }
        return false;
    }

    inline bool try_cross_undrip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i-1, j-1), ne = get(grid, i-1, j);
        uint8_t sw = get(grid, i, j-1), se = get(grid, i, j);
        if (nw == 0 && se == 1) {
            set(grid, i-1, j-1, 2); set(grid, i, j, 2);
            set(grid, i-1, j, (ne == 4) ? 3 : 5);
            set(grid, i, j-1, (sw == 5) ? 3 : 4);
            return true;
        }
        if (nw == 3 && se == 3) {
            set(grid, i-1, j-1, 1); set(grid, i, j, 0);
            set(grid, i-1, j, (ne == 4) ? 3 : 5);
            set(grid, i, j-1, (sw == 5) ? 3 : 4);
            return true;
        }
        if (nw == 3 && se == 1) {
            set(grid, i-1, j-1, 1); set(grid, i, j, 2);
            set(grid, i-1, j, (ne == 4) ? 3 : 5);
            set(grid, i, j-1, (sw == 5) ? 3 : 4);
            return true;
        }
        return false;
    }

    inline bool try_upmove(std::vector<uint8_t>& grid, int i, int j) {
        if (try_drip(grid, i, j)) return true;
        return try_cross_drip(grid, i, j);
    }
    inline bool try_downmove(std::vector<uint8_t>& grid, int i, int j) {
        if (try_undrip(grid, i, j)) return true;
        return try_cross_undrip(grid, i, j);
    }

    bool involves_cross(const std::vector<uint8_t>& old_b0, const std::vector<uint8_t>& new_b0,
                        const std::vector<uint8_t>& old_b1, const std::vector<uint8_t>& new_b1,
                        int i, int j) const {
        int r0 = i-1, c0 = j-1, r1 = i, c1 = j;
        return get(old_b0,r0,c0)==1 || get(new_b0,r0,c0)==1 ||
               get(old_b1,r0,c0)==1 || get(new_b1,r0,c0)==1 ||
               get(old_b0,r1,c1)==1 || get(new_b0,r1,c1)==1 ||
               get(old_b1,r1,c1)==1 || get(new_b1,r1,c1)==1;
    }

    // Perform one coupled step WITH ordering check.
    // Returns true if ordering violation detected.
    bool step_coupled_checked() {
        if (n < 2) return false;
        total_steps++;

        std::uniform_int_distribution<int> pos_dist(1, n-1);
        std::uniform_int_distribution<int> dir_dist(0, 1);

        int maxTries = 10 * std::max(1, (n-1)*(n-1));

        for (int t = 0; t < maxTries; t++) {
            bool isUp = dir_dist(rng) == 0;
            int i = pos_dist(rng);
            int j = pos_dist(rng);

            std::copy(b0.begin(), b0.end(), temp0.begin());
            std::copy(b1.begin(), b1.end(), temp1.begin());

            bool moved0, moved1;
            if (isUp) {
                moved0 = try_upmove(temp0, i, j);
                moved1 = try_upmove(temp1, i, j);
            } else {
                moved0 = try_downmove(temp0, i, j);
                moved1 = try_downmove(temp1, i, j);
            }

            if (!moved0 && !moved1) continue;

            moves_applied++;

            bool is_cross = involves_cross(b0, temp0, b1, temp1, i, j);
            if (is_cross) cross_moves++;

            // Track asymmetric geometric availability
            if (moved0 != moved1) asymmetric_acceptance++;

            if (is_cross) {
                bool ok0 = !moved0 || bpd2perm_check(temp0);
                bool ok1 = !moved1 || bpd2perm_check(temp1);

                if (!ok0) cross_rejections_b0++;
                if (!ok1) cross_rejections_b1++;

                // Track asymmetric reducedness rejection
                if (ok0 != ok1) asymmetric_rejections++;

                if (ok0 && ok1) {
                    // Accept the move on both chains
                    if (moved0) std::swap(b0, temp0);
                    if (moved1) std::swap(b1, temp1);

                    // CHECK ORDERING
                    compute_height(b0, n, h0);
                    compute_height(b1, n, h1);
                    int vi, vj;
                    if (!check_ordering(h0, h1, n, vi, vj)) {
                        ordering_violations++;
                        printf("\n*** ORDERING VIOLATION at step %lld ***\n",
                               (long long)total_steps);
                        printf("Move: %s at (%d,%d), cross_move=%s\n",
                               isUp ? "UP" : "DOWN", i, j, is_cross ? "yes" : "no");
                        printf("moved0=%d, moved1=%d\n", moved0, moved1);
                        printf("Violation at face vertex (%d,%d): h0=%d < h1=%d\n",
                               vi, vj, h0[vi*(n+1)+vj], h1[vi*(n+1)+vj]);
                        printf("h_b0:\n"); print_height(h0, n);
                        printf("h_b1:\n"); print_height(h1, n);
                        printf("b0 perm: ");
                        auto p0 = compute_perm(b0);
                        for (int k = 0; k < n; k++) printf("%d ", p0[k]);
                        printf("\nb1 perm: ");
                        auto p1 = compute_perm(b1);
                        for (int k = 0; k < n; k++) printf("%d ", p1[k]);
                        printf("\n\n");
                        return true;
                    }
                    return false;
                }
                // One or both rejected for reducedness - don't apply
                continue;
            } else {
                // No cross involved - always safe (drip/undrip preserve reducedness)
                if (moved0) std::swap(b0, temp0);
                if (moved1) std::swap(b1, temp1);

                // CHECK ORDERING
                compute_height(b0, n, h0);
                compute_height(b1, n, h1);
                int vi, vj;
                if (!check_ordering(h0, h1, n, vi, vj)) {
                    ordering_violations++;
                    printf("\n*** ORDERING VIOLATION at step %lld ***\n",
                           (long long)total_steps);
                    printf("Move: %s at (%d,%d), cross_move=no\n",
                           isUp ? "UP" : "DOWN", i, j);
                    printf("moved0=%d, moved1=%d\n", moved0, moved1);
                    printf("Violation at face vertex (%d,%d): h0=%d < h1=%d\n",
                           vi, vj, h0[vi*(n+1)+vj], h1[vi*(n+1)+vj]);
                    printf("h_b0:\n"); print_height(h0, n);
                    printf("h_b1:\n"); print_height(h1, n);
                    return true;
                }
                return false;
            }
        }

        // Exhaustive fallback (same as sampler)
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                for (int dir = 0; dir < 2; dir++) {
                    std::copy(b0.begin(), b0.end(), temp0.begin());
                    std::copy(b1.begin(), b1.end(), temp1.begin());
                    bool moved0, moved1;
                    if (dir == 0) {
                        moved0 = try_upmove(temp0, i, j);
                        moved1 = try_upmove(temp1, i, j);
                    } else {
                        moved0 = try_downmove(temp0, i, j);
                        moved1 = try_downmove(temp1, i, j);
                    }
                    if (!moved0 && !moved1) continue;
                    moves_applied++;
                    bool is_cross = involves_cross(b0, temp0, b1, temp1, i, j);
                    if (is_cross) {
                        cross_moves++;
                        bool ok0 = !moved0 || bpd2perm_check(temp0);
                        bool ok1 = !moved1 || bpd2perm_check(temp1);
                        if (!ok0) cross_rejections_b0++;
                        if (!ok1) cross_rejections_b1++;
                        if (ok0 != ok1) asymmetric_rejections++;
                        if (ok0 && ok1) {
                            if (moved0) std::swap(b0, temp0);
                            if (moved1) std::swap(b1, temp1);
                            compute_height(b0, n, h0);
                            compute_height(b1, n, h1);
                            int vi, vj;
                            if (!check_ordering(h0, h1, n, vi, vj)) {
                                ordering_violations++;
                                printf("\n*** ORDERING VIOLATION (exhaustive) step %lld ***\n",
                                       (long long)total_steps);
                                return true;
                            }
                            return false;
                        }
                    } else {
                        if (moved0) std::swap(b0, temp0);
                        if (moved1) std::swap(b1, temp1);
                        compute_height(b0, n, h0);
                        compute_height(b1, n, h1);
                        int vi, vj;
                        if (!check_ordering(h0, h1, n, vi, vj)) {
                            ordering_violations++;
                            printf("\n*** ORDERING VIOLATION (exhaustive) step %lld ***\n",
                                   (long long)total_steps);
                            return true;
                        }
                        return false;
                    }
                }
            }
        }
        return false;
    }

    bool is_met() const { return b0 == b1; }
    const std::vector<uint8_t>& get_b0() const { return b0; }
    const std::vector<uint8_t>& get_b1() const { return b1; }
    const std::vector<uint8_t>& get_result() const { return b0; }

    void reset_counters() {
        total_steps = 0;
        ordering_violations = 0;
        moves_applied = 0;
        cross_moves = 0;
        cross_rejections_b0 = 0;
        cross_rejections_b1 = 0;
        asymmetric_rejections = 0;
        asymmetric_acceptance = 0;
    }
};

// =============================================================================
// BPD state key for frequency counting
// =============================================================================

struct BPDKey {
    std::vector<uint8_t> grid;
    bool operator<(const BPDKey& o) const { return grid < o.grid; }
    bool operator==(const BPDKey& o) const { return grid == o.grid; }
};

// =============================================================================
// Ordering test
// =============================================================================

void run_ordering_test(int n, int num_samples) {
    printf("=== ORDERING TEST: n=%d, %d samples ===\n\n", n, num_samples);

    BPDEngine engine(n);
    std::mt19937 seed_rng(std::random_device{}());

    int64_t total_violations = 0;
    int64_t total_steps_all = 0;
    int64_t total_moves_all = 0;
    int64_t total_cross_all = 0;
    int64_t total_asym_rej = 0;
    int64_t total_asym_acc = 0;
    int completed = 0;
    int64_t max_coupling_steps = (int64_t)100 * n * n * n;

    for (int s = 0; s < num_samples; s++) {
        engine.reset();
        engine.reset_counters();
        uint32_t seed = seed_rng();
        engine.seed(seed);

        int64_t steps = 0;
        bool violation_found = false;

        while (steps < max_coupling_steps) {
            bool violation = engine.step_coupled_checked();
            steps++;
            if (violation) {
                violation_found = true;
                total_violations++;
                printf("Sample %d (seed=%u): VIOLATION at step %lld\n\n",
                       s, seed, (long long)steps);
                break;
            }
            if (engine.is_met()) break;
        }

        if (!violation_found && !engine.is_met()) {
            printf("Sample %d: timeout after %lld steps (no violation, no coalescence)\n",
                   s, (long long)steps);
        }

        total_steps_all += engine.total_steps;
        total_moves_all += engine.moves_applied;
        total_cross_all += engine.cross_moves;
        total_asym_rej += engine.asymmetric_rejections;
        total_asym_acc += engine.asymmetric_acceptance;

        if (engine.is_met()) completed++;

        if ((s+1) % std::max(1, num_samples/20) == 0 || s == num_samples - 1) {
            printf("\r[%d/%d] violations=%lld, completed=%d, steps=%lld, "
                   "cross_moves=%lld, asym_reject=%lld, asym_geom=%lld   ",
                   s+1, num_samples, (long long)total_violations, completed,
                   (long long)total_steps_all, (long long)total_cross_all,
                   (long long)total_asym_rej, (long long)total_asym_acc);
            fflush(stdout);
        }
    }

    printf("\n\n=== ORDERING TEST RESULTS ===\n");
    printf("n = %d\n", n);
    printf("Samples attempted: %d\n", num_samples);
    printf("Samples coalesced: %d\n", completed);
    printf("Total steps: %lld\n", (long long)total_steps_all);
    printf("Total moves applied: %lld\n", (long long)total_moves_all);
    printf("Total cross moves: %lld\n", (long long)total_cross_all);
    printf("Asymmetric geometry (move possible on one chain only): %lld\n",
           (long long)total_asym_acc);
    printf("Asymmetric reducedness rejection: %lld\n", (long long)total_asym_rej);
    printf("\n*** ORDERING VIOLATIONS: %lld ***\n", (long long)total_violations);
    if (total_violations == 0) {
        printf(">>> The bounding chain ordering h_b0 <= h_b1 was NEVER violated. <<<\n");
    } else {
        printf(">>> VIOLATIONS DETECTED! CFTP monotonicity is broken. <<<\n");
    }
    printf("\n");
}

// =============================================================================
// Frequency test (exact uniformity check at small n)
// =============================================================================

void run_freq_test(int n, int num_samples) {
    printf("=== FREQUENCY TEST: n=%d, %d samples ===\n\n", n, num_samples);

    if (n > 6) {
        printf("Warning: n=%d may have too many RBPD states for meaningful frequency test.\n", n);
        printf("Proceeding anyway (will test permutation frequencies instead).\n\n");
    }

    BPDEngine engine(n);
    std::mt19937 seed_rng(std::random_device{}());

    // Count BPD state frequencies
    std::map<BPDKey, int> bpd_counts;
    // Count permutation frequencies
    std::map<std::vector<int>, int> perm_counts;

    int completed = 0;
    int timeouts = 0;
    int64_t max_coupling_steps = (int64_t)100 * n * n * n;

    for (int s = 0; s < num_samples; s++) {
        engine.reset();
        engine.reset_counters();
        uint32_t seed = seed_rng();
        engine.seed(seed);

        int64_t steps = 0;
        while (steps < max_coupling_steps) {
            engine.step_coupled_checked();
            steps++;
            if (engine.is_met()) break;
        }

        if (engine.is_met()) {
            completed++;
            BPDKey key{std::vector<uint8_t>(engine.get_result().begin(),
                                            engine.get_result().end())};
            bpd_counts[key]++;

            auto perm = engine.compute_perm(engine.get_result());
            perm_counts[perm]++;
        } else {
            timeouts++;
        }

        if ((s+1) % std::max(1, num_samples/20) == 0 || s == num_samples - 1) {
            printf("\r[%d/%d] completed=%d, timeouts=%d, distinct_BPDs=%d, distinct_perms=%d   ",
                   s+1, num_samples, completed, timeouts,
                   (int)bpd_counts.size(), (int)perm_counts.size());
            fflush(stdout);
        }
    }

    printf("\n\n=== FREQUENCY TEST RESULTS ===\n");
    printf("n = %d\n", n);
    printf("Samples attempted: %d\n", num_samples);
    printf("Samples completed: %d\n", completed);
    printf("Timeouts: %d\n", timeouts);
    printf("Distinct BPD states seen: %d\n", (int)bpd_counts.size());
    printf("Distinct permutations seen: %d\n", (int)perm_counts.size());

    if (completed == 0) {
        printf("No completed samples - cannot do frequency analysis.\n");
        return;
    }

    // --- BPD-level chi-squared test ---
    int K = bpd_counts.size();
    double expected_per_bpd = (double)completed / K;
    printf("\nBPD-level analysis (assuming %d states total):\n", K);
    printf("Expected count per BPD if uniform: %.2f\n", expected_per_bpd);

    double chi2_bpd = 0;
    int min_count = completed, max_count = 0;
    for (auto& [key, count] : bpd_counts) {
        double diff = count - expected_per_bpd;
        chi2_bpd += diff * diff / expected_per_bpd;
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
    }
    int df_bpd = K - 1;
    // Approximate p-value using normal approximation of chi-squared
    double z_bpd = (chi2_bpd - df_bpd) / std::sqrt(2.0 * df_bpd);

    printf("Chi-squared statistic: %.2f (df=%d)\n", chi2_bpd, df_bpd);
    printf("Normalized (z-score): %.2f\n", z_bpd);
    printf("Min count: %d, Max count: %d, Ratio max/min: %.2f\n",
           min_count, max_count, (double)max_count / std::max(1, min_count));
    if (std::abs(z_bpd) < 3.0) {
        printf(">>> BPD frequencies CONSISTENT with uniform (|z| < 3). <<<\n");
    } else {
        printf(">>> BPD frequencies INCONSISTENT with uniform (|z| = %.2f). <<<\n",
               std::abs(z_bpd));
    }

    // --- Permutation-level analysis ---
    printf("\nPermutation frequencies (count, permutation):\n");
    // Sort by count descending
    std::vector<std::pair<int, std::vector<int>>> perm_list;
    for (auto& [perm, count] : perm_counts) {
        perm_list.push_back({count, perm});
    }
    std::sort(perm_list.begin(), perm_list.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int show = std::min((int)perm_list.size(), 30);
    for (int i = 0; i < show; i++) {
        printf("  %5d  (", perm_list[i].first);
        for (int k = 0; k < n; k++)
            printf("%d%s", perm_list[i].second[k], k < n-1 ? "," : "");
        printf(")\n");
    }
    if ((int)perm_list.size() > show)
        printf("  ... (%d more permutations)\n", (int)perm_list.size() - show);

    // Expected permutation frequency = S_w(1^n) / total_RBPDs
    // We don't know S_w(1^n) here, but we can note the count distribution
    printf("\nNote: Permutation w should appear with frequency proportional to S_w(1^n).\n");
    printf("Identity (1,...,n) should appear exactly once per occurrence (1 RBPD).\n");
    printf("Check: identity count = %d, expected ~= %d / (total RBPDs)\n",
           perm_counts.count(std::vector<int>{}) > 0 ? 0 : 0, completed);

    // Check identity count specifically
    std::vector<int> id_perm(n);
    for (int i = 0; i < n; i++) id_perm[i] = i + 1;
    if (perm_counts.count(id_perm)) {
        printf("Identity permutation count: %d\n", perm_counts[id_perm]);
    } else {
        printf("Identity permutation: not seen\n");
    }

    printf("\n");
}

// =============================================================================
// Proper Backward CFTP (Propp-Wilson protocol)
// =============================================================================

struct CftpUpdate {
    int i, j;   // position (1..n-1)
    bool isUp;  // direction
};

// Apply one update to a BPD grid - UNCONSTRAINED (no reducedness check).
// This is the standard ASM chain update: provably monotone.
void apply_unconstrained(BPDEngine& eng, std::vector<uint8_t>& grid, const CftpUpdate& u) {
    std::vector<uint8_t> temp(grid);
    bool moved = u.isUp ? eng.try_upmove(temp, u.i, u.j)
                        : eng.try_downmove(temp, u.i, u.j);
    if (moved) grid.swap(temp);
}

// Apply one update with per-chain reducedness check (internal rejection).
// Each chain independently rejects moves that break reducedness.
void apply_with_reduced_check(BPDEngine& eng, std::vector<uint8_t>& grid,
                               const CftpUpdate& u) {
    int n = eng.get_n();
    std::vector<uint8_t> temp(grid);
    bool moved = u.isUp ? eng.try_upmove(temp, u.i, u.j)
                        : eng.try_downmove(temp, u.i, u.j);
    if (!moved) return;

    // Check if cross involved
    bool has_cross = (grid[(u.i-1)*n+(u.j-1)] == 1 || temp[(u.i-1)*n+(u.j-1)] == 1 ||
                      grid[u.i*n+u.j] == 1 || temp[u.i*n+u.j] == 1);
    if (has_cross) {
        if (eng.bpd2perm_check(temp)) grid.swap(temp);
        // else: reject (non-reduced) -> stay
    } else {
        grid.swap(temp);  // drip/undrip always preserves reducedness
    }
}

// Run one round of backward CFTP: replay all updates from extremes, check coalescence.
// mode 0 = unconstrained, mode 1 = with reducedness check
bool run_cftp_round(BPDEngine& eng, const std::vector<CftpUpdate>& updates,
                    std::vector<uint8_t>& bot, std::vector<uint8_t>& top, int mode) {
    int n = eng.get_n();
    std::vector<int> id_perm(n), w0_perm(n);
    for (int i = 0; i < n; i++) { id_perm[i] = i + 1; w0_perm[i] = n - i; }

    bot.resize(n * n);
    top.resize(n * n);
    eng.generateRotheBPD(bot, id_perm);
    eng.generateRotheBPD(top, w0_perm);

    for (const auto& u : updates) {
        if (mode == 0) {
            apply_unconstrained(eng, bot, u);
            apply_unconstrained(eng, top, u);
        } else {
            apply_with_reduced_check(eng, bot, u);
            apply_with_reduced_check(eng, top, u);
        }
    }

    return bot == top;
}

// Proper backward CFTP (Propp-Wilson protocol).
// mode 0 = unconstrained ASM chain (external rejection of non-reduced output)
// mode 1 = RBPD chain with internal per-chain reducedness rejection
// Returns the coalesced BPD grid.
std::vector<uint8_t> backward_cftp_sample(BPDEngine& eng, std::mt19937& rng, int mode,
                                           int& rounds_out, int& total_updates_out) {
    int n = eng.get_n();
    std::uniform_int_distribution<int> pos(1, n - 1);
    std::uniform_int_distribution<int> dir(0, 1);

    // Initial window size (tuned for quick coalescence)
    int initial_window = std::max(16, 2 * n * n);

    // Stored updates in chronological order: [0] = earliest time, [last] = time 0
    std::vector<CftpUpdate> updates;
    updates.reserve(initial_window * 8);

    // Generate initial window
    for (int t = 0; t < initial_window; t++) {
        updates.push_back({pos(rng), pos(rng), dir(rng) == 0});
    }

    std::vector<uint8_t> bot, top;
    int round = 0;

    while (round < 40) {  // safety limit (2^40 updates = way more than needed)
        round++;

        if (run_cftp_round(eng, updates, bot, top, mode)) {
            rounds_out = round;
            total_updates_out = (int)updates.size();
            return bot;  // coalesced!
        }

        // Double: prepend updates.size() new random updates
        int extend = (int)updates.size();
        std::vector<CftpUpdate> earlier;
        earlier.reserve(extend + updates.size());
        for (int t = 0; t < extend; t++) {
            earlier.push_back({pos(rng), pos(rng), dir(rng) == 0});
        }
        earlier.insert(earlier.end(), updates.begin(), updates.end());
        updates = std::move(earlier);
    }

    rounds_out = round;
    total_updates_out = (int)updates.size();
    return {};  // failed
}

// Print chi-squared test results for BPD-level frequencies
void print_chi2_results(const std::map<BPDKey, int>& bpd_counts, int completed) {
    int K = bpd_counts.size();
    double expected = (double)completed / K;
    printf("\nBPD-level analysis (%d distinct states):\n", K);
    printf("Expected count per BPD if uniform: %.2f\n", expected);

    double chi2 = 0;
    int min_count = completed, max_count = 0;
    for (auto& [key, count] : bpd_counts) {
        double diff = count - expected;
        chi2 += diff * diff / expected;
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
    }
    int df = K - 1;
    double z = (chi2 - df) / std::sqrt(2.0 * df);

    printf("Chi-squared: %.2f (df=%d)\n", chi2, df);
    printf("Normalized z-score: %.2f\n", z);
    printf("Min count: %d, Max count: %d, Ratio max/min: %.2f\n",
           min_count, max_count, (double)max_count / std::max(1, min_count));
    if (std::abs(z) < 3.0) {
        printf(">>> CONSISTENT with uniform (|z| = %.2f < 3). <<<\n", std::abs(z));
    } else {
        printf(">>> NOT UNIFORM (|z| = %.2f). <<<\n", std::abs(z));
    }
}

// Print permutation frequency table
void print_perm_freqs(const std::map<std::vector<int>, int>& perm_counts, int n, int completed) {
    printf("\nPermutation frequencies (count, permutation):\n");
    std::vector<std::pair<int, std::vector<int>>> perm_list;
    for (auto& [perm, count] : perm_counts) {
        perm_list.push_back({count, perm});
    }
    std::sort(perm_list.begin(), perm_list.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int show = std::min((int)perm_list.size(), 20);
    for (int i = 0; i < show; i++) {
        printf("  %5d  (", perm_list[i].first);
        for (int k = 0; k < n; k++)
            printf("%d%s", perm_list[i].second[k], k < n-1 ? "," : "");
        printf(")\n");
    }
    if ((int)perm_list.size() > show)
        printf("  ... (%d more)\n", (int)perm_list.size() - show);

    // Identity count
    std::vector<int> id(n);
    for (int i = 0; i < n; i++) id[i] = i + 1;
    printf("\nIdentity count: %d (expected if uniform over %d BPDs: %.1f)\n",
           perm_counts.count(id) ? perm_counts.at(id) : 0,
           (int)perm_list.size(), // rough, since we don't know total BPDs
           (double)completed / 41.0); // only correct for n=4
}

// =============================================================================
// Proper CFTP frequency test
// =============================================================================

void run_proper_cftp_freq(int n, int num_samples, int mode) {
    const char* mode_name = (mode == 0) ? "EXTERNAL rejection (unconstrained ASM chain)"
                                        : "INTERNAL rejection (RBPD chain)";
    printf("=== PROPER BACKWARD CFTP (%s) ===\n", mode_name);
    printf("n=%d, %d samples\n\n", n, num_samples);

    BPDEngine eng(n);
    std::mt19937 rng(std::random_device{}());

    std::map<BPDKey, int> bpd_counts;
    std::map<std::vector<int>, int> perm_counts;
    int completed = 0, rejected = 0, failed = 0;
    int64_t total_rounds = 0, total_updates = 0;

    while (completed < num_samples) {
        int rounds, updates;
        auto result = backward_cftp_sample(eng, rng, mode, rounds, updates);

        if (result.empty()) {
            failed++;
            continue;
        }

        total_rounds += rounds;
        total_updates += updates;

        // For external rejection: check if output is reduced
        if (mode == 0) {
            if (!eng.bpd2perm_check(result)) {
                rejected++;
                continue;
            }
        }

        completed++;
        bpd_counts[BPDKey{result}]++;
        perm_counts[eng.compute_perm(result)]++;

        if (completed % std::max(1, num_samples/20) == 0 || completed == num_samples) {
            double avg_rounds = (double)total_rounds / (completed + rejected + failed);
            double avg_updates = (double)total_updates / (completed + rejected + failed);
            printf("\r%d/%d done (rejected=%d, avg_rounds=%.1f, avg_updates=%.0f, BPDs=%d)   ",
                   completed, num_samples, rejected, avg_rounds, avg_updates,
                   (int)bpd_counts.size());
            fflush(stdout);
        }
    }

    printf("\n\n=== RESULTS: PROPER BACKWARD CFTP (%s) ===\n", mode_name);
    printf("n = %d\n", n);
    printf("Completed samples: %d\n", completed);
    printf("Rejected (non-reduced, mode 0 only): %d\n", rejected);
    printf("Failed (timeout): %d\n", failed);
    printf("Distinct BPD states: %d\n", (int)bpd_counts.size());
    printf("Distinct permutations: %d\n", (int)perm_counts.size());
    printf("Avg rounds per attempt: %.2f\n",
           (double)total_rounds / (completed + rejected + failed));
    printf("Avg updates per attempt: %.1f\n",
           (double)total_updates / (completed + rejected + failed));

    print_chi2_results(bpd_counts, completed);
    print_perm_freqs(perm_counts, n, completed);
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("CFTP Diagnostic Tool\n\n");
        printf("Usage:\n");
        printf("  %s ordering <n> <num_samples>   Check chain ordering at every step\n", argv[0]);
        printf("  %s freq <n> <num_samples>       Frequency test (FORWARD coupling, biased)\n", argv[0]);
        printf("  %s cftp_ext <n> <num_samples>   Backward CFTP, external rejection (provably correct)\n", argv[0]);
        printf("  %s cftp_int <n> <num_samples>   Backward CFTP, internal rejection (empirically correct)\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s ordering 4 10000\n", argv[0]);
        printf("  %s cftp_ext 4 100000    # Gold standard: proper backward CFTP\n", argv[0]);
        printf("  %s cftp_int 4 100000    # Same but with per-chain reducedness check\n", argv[0]);
        printf("  %s freq 4 100000        # Forward coupling (shown to be biased)\n", argv[0]);
        return 1;
    }

    const char* mode = argv[1];
    int n = atoi(argv[2]);
    int num_samples = atoi(argv[3]);

    if (n < 2) {
        fprintf(stderr, "Error: n must be >= 2\n");
        return 1;
    }

    if (strcmp(mode, "ordering") == 0) {
        run_ordering_test(n, num_samples);
    } else if (strcmp(mode, "freq") == 0) {
        run_freq_test(n, num_samples);
    } else if (strcmp(mode, "cftp_ext") == 0) {
        run_proper_cftp_freq(n, num_samples, 0);
    } else if (strcmp(mode, "cftp_int") == 0) {
        run_proper_cftp_freq(n, num_samples, 1);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    return 0;
}
