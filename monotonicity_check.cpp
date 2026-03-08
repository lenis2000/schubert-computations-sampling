// monotonicity_check.cpp
// Verifies the "Contact Shielding" monotonicity claim for BPD CFTP
//
// Tests monotonicity in the chosen height convention:
// if X <= Y in the order (equivalently h_X >= h_Y pointwise),
// then after applying the same BPD update to both X and Y (with
// independent reducedness rejection), the order should still hold.
//
// Modes:
//   ./monotonicity_check cftp <n> <trials>    - Monitor CFTP chains
//   ./monotonicity_check exhaust <n>          - Exhaustive (n<=5)
//   ./monotonicity_check both <n> <trials>    - Both tests
//
// Compile:
//   clang++ -O3 -std=c++17 monotonicity_check.cpp -o monotonicity_check

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <cassert>
#include <chrono>

// Tile types: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz
static const char* TILE_NAMES[] = {"blank","cross","r-elb","j-elb","vert ","horiz"};

// ============================================================================
// BPD State
// ============================================================================

struct BPD {
    int n;
    std::vector<uint8_t> grid;

    BPD() : n(0) {}
    BPD(int sz) : n(sz), grid(sz * sz, 0) {}

    uint8_t get(int r, int c) const { return grid[r * n + c]; }
    void set(int r, int c, uint8_t v) { grid[r * n + c] = v; }

    bool operator==(const BPD& o) const { return n == o.n && grid == o.grid; }
    bool operator<(const BPD& o) const { return grid < o.grid; }

    // Generate Rothe BPD for permutation w (1-indexed values)
    void initRothe(const std::vector<int>& w) {
        int minW = *std::min_element(w.begin(), w.end());
        for (int j = 1; j <= n; j++) {
            if (j + minW - 1 < w[0]) set(0, j-1, 0);
            else if (j + minW - 1 == w[0]) set(0, j-1, 2);
            else set(0, j-1, 5);
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (j + minW - 1 < w[i-1]) {
                    uint8_t up = get(i-2, j-1);
                    set(i-1, j-1, (up==2||up==4||up==1) ? 4 : 0);
                } else if (j + minW - 1 == w[i-1]) {
                    set(i-1, j-1, 2);
                } else {
                    uint8_t up = get(i-2, j-1);
                    set(i-1, j-1, (up==4||up==2||up==1) ? 1 : 5);
                }
            }
        }
    }

    void initIdentity() {
        std::vector<int> id(n);
        for (int i = 0; i < n; i++) id[i] = i + 1;
        initRothe(id);
    }

    void initW0() {
        std::vector<int> w0(n);
        for (int i = 0; i < n; i++) w0[i] = n - i;
        initRothe(w0);
    }

    int get_sk(int i, int j) const {
        if (get(i,j) != 1) return -1;
        int k = 1;
        for (int s = 1; s <= std::min(i,j); s++) {
            uint8_t t = get(i-s, j-s);
            if (t==2||t==3||t==4||t==5) k++;
            else if (t==1) k += 2;
        }
        return k;
    }

    bool isReduced() const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;
        for (int s = 1-n; s < 1; s++) {
            for (int t = 0; t < s+n; t++) {
                int i = n-t-1, j = n+s-t-1;
                int k = get_sk(i, j);
                if (k != -1) {
                    if (ww[k-1] > ww[k]) return false;
                    std::swap(ww[k-1], ww[k]);
                }
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n-s; t++) {
                int i = n-s-t-1, j = n-t-1;
                int k = get_sk(i, j);
                if (k != -1) {
                    if (ww[k-1] > ww[k]) return false;
                    std::swap(ww[k-1], ww[k]);
                }
            }
        }
        return true;
    }

    std::vector<int> perm() const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;
        for (int s = 1-n; s < 1; s++) {
            for (int t = 0; t < s+n; t++) {
                int i = n-t-1, j = n+s-t-1;
                int k = get_sk(i, j);
                if (k != -1 && k < n) std::swap(ww[k-1], ww[k]);
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n-s; t++) {
                int i = n-s-t-1, j = n-t-1;
                int k = get_sk(i, j);
                if (k != -1 && k < n) std::swap(ww[k-1], ww[k]);
            }
        }
        return ww;
    }

    int numCrosses() const {
        int c = 0;
        for (auto t : grid) if (t == 1) c++;
        return c;
    }

    void print() const {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%d ", get(i,j));
            printf("\n");
        }
    }
};

// ============================================================================
// Height function on dual-lattice vertices (0..n)x(0..n).
//
// Convention used here: six-vertex lines are contour lines of h.
// Moving right across an edge:
//   - if that edge carries a vertical line segment, h increases by 1
//   - otherwise h is unchanged
//
// Boundary normalization: h=0 on top and left boundaries.
// ============================================================================

std::vector<int> computeHeight(const BPD& bpd) {
    int n = bpd.n;
    std::vector<int> h((n + 1) * (n + 1), 0);
    auto idx = [n](int i, int j) { return i * (n + 1) + j; };

    for (int i = 0; i <= n; i++) {
        int crossings = 0;
        for (int j = 0; j < n; j++) {
            h[idx(i, j)] = crossings;
            bool has_vertical_line = false;  // vertical segment at row i, column j
            if (i == 0) {
                has_vertical_line = false;
            } else if (i == n) {
                has_vertical_line = true;
            } else {
                uint8_t t = bpd.get(i - 1, j);
                has_vertical_line = (t == 1 || t == 2 || t == 4);
            }
            if (has_vertical_line) crossings++;
        }
        h[idx(i, n)] = crossings;
    }
    return h;
}

// Order check in this height convention: X <= Y iff h_X >= h_Y pointwise.
bool heightLeq(const std::vector<int>& hX, const std::vector<int>& hY, int n) {
    for (int i = 0; i <= n; i++)
        for (int j = 0; j <= n; j++)
            if (hX[i*(n+1)+j] < hY[i*(n+1)+j])
                return false;
    return true;
}

// Find first violation of order X <= Y, i.e. a vertex where h_X < h_Y.
bool findViolation(const std::vector<int>& hX, const std::vector<int>& hY,
                   int n, int& vi, int& vj, int& diff) {
    for (int i = 0; i <= n; i++)
        for (int j = 0; j <= n; j++) {
            int d = hY[i*(n+1)+j] - hX[i*(n+1)+j];
            if (d > 0) { vi = i; vj = j; diff = d; return true; }
        }
    return false;
}

// Count contact points where h_X(i,j) == h_Y(i,j), excluding boundary
int countContacts(const std::vector<int>& hX, const std::vector<int>& hY, int n) {
    int ct = 0;
    for (int i = 1; i < n; i++)
        for (int j = 1; j < n; j++)
            if (hX[i*(n+1)+j] == hY[i*(n+1)+j]) ct++;
    return ct;
}

// ============================================================================
// BPD Moves (same as bpd_sampler.cpp)
// ============================================================================

bool tryDrip(BPD& b, int i, int j) {
    uint8_t nw=b.get(i-1,j-1), ne=b.get(i-1,j), sw=b.get(i,j-1), se=b.get(i,j);
    if (nw==2 && se==0) {
        b.set(i-1,j-1, 0); b.set(i,j, 3);
        b.set(i-1,j, (ne==5)?2:4); b.set(i,j-1, (sw==4)?2:5);
        return true;
    }
    return false;
}

bool tryUndrip(BPD& b, int i, int j) {
    uint8_t nw=b.get(i-1,j-1), ne=b.get(i-1,j), sw=b.get(i,j-1), se=b.get(i,j);
    if (nw==0 && se==3 && (ne==2||ne==4) && (sw==2||sw==5)) {
        b.set(i-1,j-1, 2); b.set(i,j, 0);
        b.set(i-1,j, (ne==2)?5:3); b.set(i,j-1, (sw==2)?4:3);
        return true;
    }
    return false;
}

bool tryCrossDrip(BPD& b, int i, int j) {
    uint8_t nw=b.get(i-1,j-1), ne=b.get(i-1,j), sw=b.get(i,j-1), se=b.get(i,j);
    if (nw==2 && se==2) {
        b.set(i-1,j-1, 0); b.set(i,j, 1);
        b.set(i-1,j, (ne==3)?4:2); b.set(i,j-1, (sw==3)?5:2);
        return true;
    }
    if (nw==1 && se==0) {
        b.set(i-1,j-1, 3); b.set(i,j, 3);
        b.set(i-1,j, (ne==3)?4:2); b.set(i,j-1, (sw==3)?5:2);
        return true;
    }
    if (nw==1 && se==2) {
        b.set(i-1,j-1, 3); b.set(i,j, 1);
        b.set(i-1,j, (ne==3)?4:2); b.set(i,j-1, (sw==3)?5:2);
        return true;
    }
    return false;
}

bool tryCrossUndrip(BPD& b, int i, int j) {
    uint8_t nw=b.get(i-1,j-1), ne=b.get(i-1,j), sw=b.get(i,j-1), se=b.get(i,j);
    if (nw==0 && se==1) {
        b.set(i-1,j-1, 2); b.set(i,j, 2);
        b.set(i-1,j, (ne==4)?3:5); b.set(i,j-1, (sw==5)?3:4);
        return true;
    }
    if (nw==3 && se==3) {
        b.set(i-1,j-1, 1); b.set(i,j, 0);
        b.set(i-1,j, (ne==4)?3:5); b.set(i,j-1, (sw==5)?3:4);
        return true;
    }
    if (nw==3 && se==1) {
        b.set(i-1,j-1, 1); b.set(i,j, 2);
        b.set(i-1,j, (ne==4)?3:5); b.set(i,j-1, (sw==5)?3:4);
        return true;
    }
    return false;
}

bool tryUp(BPD& b, int i, int j) {
    if (tryDrip(b, i, j)) return true;
    return tryCrossDrip(b, i, j);
}

bool tryDown(BPD& b, int i, int j) {
    if (tryUndrip(b, i, j)) return true;
    return tryCrossUndrip(b, i, j);
}

// Apply update with reducedness rejection. Returns the new state.
BPD applyUpdate(const BPD& state, int i, int j, bool isUp) {
    BPD proposed = state;
    bool moved = isUp ? tryUp(proposed, i, j) : tryDown(proposed, i, j);
    if (!moved) return state;

    // Check if cross involved
    bool has_cross = (state.get(i-1,j-1)==1 || proposed.get(i-1,j-1)==1 ||
                      state.get(i,j)==1 || proposed.get(i,j)==1);
    if (has_cross) {
        if (proposed.isReduced()) return proposed;
        return state; // reject
    }
    return proposed; // drip/undrip always safe
}

// ============================================================================
// TEST 1: CFTP Monitoring
// Run backward CFTP, check h(b0) <= h(b1) after every single update.
// ============================================================================

struct CFTPMonitorResult {
    int64_t total_updates;
    int64_t violations;
    int64_t contact_updates;      // updates where some h values are equal
    int64_t asymmetric_accepts;   // X accepts but Y rejects (or vice versa)
    int64_t both_accept;
    int64_t both_reject;
    int64_t both_noop;
    int coalesced_round;
};

CFTPMonitorResult monitorCFTP(int n, uint32_t seed) {
    CFTPMonitorResult res = {};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pos(1, n-1);
    std::uniform_int_distribution<int> dir(0, 1);

    BPD b0(n), b1(n);
    b0.initIdentity();
    b1.initW0();

    // Verify initial ordering
    auto h0 = computeHeight(b0);
    auto h1 = computeHeight(b1);
    assert(heightLeq(h0, h1, n));

    int initial_window = std::max(16, 2 * n * n);

    struct Update { int8_t i, j; bool isUp; };
    std::vector<Update> updates;
    updates.reserve(initial_window * 8);

    for (int t = 0; t < initial_window; t++)
        updates.push_back({(int8_t)pos(rng), (int8_t)pos(rng), dir(rng)==0});

    for (int round = 1; round <= 40; round++) {
        b0.initIdentity();
        b1.initW0();

        for (const auto& u : updates) {
            BPD new_b0 = applyUpdate(b0, u.i, u.j, u.isUp);
            BPD new_b1 = applyUpdate(b1, u.i, u.j, u.isUp);

            bool b0_moved = !(new_b0 == b0);
            bool b1_moved = !(new_b1 == b1);

            res.total_updates++;

            if (b0_moved && b1_moved) res.both_accept++;
            else if (!b0_moved && !b1_moved) res.both_noop++;
            else res.asymmetric_accepts++;

            b0 = new_b0;
            b1 = new_b1;

            // Check monotonicity
            h0 = computeHeight(b0);
            h1 = computeHeight(b1);

            int vi, vj, diff;
            if (findViolation(h0, h1, n, vi, vj, diff)) {
                res.violations++;
                if (res.violations <= 5) {
                    printf("  VIOLATION at step %lld: h_b0(%d,%d)=%d > h_b1(%d,%d)=%d\n",
                           (long long)res.total_updates, vi, vj,
                           h0[vi*(n+1)+vj], vi, vj, h1[vi*(n+1)+vj]);
                    printf("    Update: pos=(%d,%d) dir=%s, b0_moved=%d b1_moved=%d\n",
                           u.i, u.j, u.isUp?"UP":"DOWN", b0_moved, b1_moved);
                    printf("    b0 perm: ");
                    auto p0 = b0.perm();
                    for (int x : p0) printf("%d ", x);
                    printf("\n    b1 perm: ");
                    auto p1 = b1.perm();
                    for (int x : p1) printf("%d ", x);
                    printf("\n");
                }
            }

            // Count contacts (interior only)
            int contacts = countContacts(h0, h1, n);
            if (contacts > 0) res.contact_updates++;
        }

        if (b0 == b1) {
            res.coalesced_round = round;
            return res;
        }

        // Double window
        int extend = (int)updates.size();
        std::vector<Update> earlier;
        earlier.reserve(extend + updates.size());
        for (int t = 0; t < extend; t++)
            earlier.push_back({(int8_t)pos(rng), (int8_t)pos(rng), dir(rng)==0});
        earlier.insert(earlier.end(), updates.begin(), updates.end());
        updates = std::move(earlier);
    }

    res.coalesced_round = -1;
    return res;
}

void testCFTPMonitoring(int n, int trials) {
    printf("=== CFTP Monotonicity Monitor: n=%d, trials=%d ===\n", n, trials);

    int64_t total_violations = 0;
    int64_t total_updates = 0;
    int64_t total_asymmetric = 0;
    int64_t total_contacts = 0;
    int coalesced = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 seed_rng(std::random_device{}());

    for (int trial = 0; trial < trials; trial++) {
        uint32_t seed = seed_rng();
        auto res = monitorCFTP(n, seed);

        total_violations += res.violations;
        total_updates += res.total_updates;
        total_asymmetric += res.asymmetric_accepts;
        total_contacts += res.contact_updates;
        if (res.coalesced_round > 0) coalesced++;

        if ((trial+1) % std::max(1, trials/10) == 0 || trial == trials-1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            printf("  [%d/%d] violations=%lld, updates=%lld, asymmetric=%lld, "
                   "contacts=%lld, coalesced=%d (%.1fs)\n",
                   trial+1, trials, (long long)total_violations,
                   (long long)total_updates, (long long)total_asymmetric,
                   (long long)total_contacts, coalesced, elapsed);
        }
    }

    printf("\n--- CFTP Monitor Summary (n=%d) ---\n", n);
    printf("Trials:              %d\n", trials);
    printf("Total updates:       %lld\n", (long long)total_updates);
    printf("VIOLATIONS:          %lld\n", (long long)total_violations);
    printf("Asymmetric accepts:  %lld (%.2f%%)\n",
           (long long)total_asymmetric, 100.0*total_asymmetric/total_updates);
    printf("Updates with contacts: %lld (%.2f%%)\n",
           (long long)total_contacts, 100.0*total_contacts/total_updates);
    printf("Coalesced:           %d/%d\n", coalesced, trials);

    if (total_violations == 0)
        printf("\n*** MONOTONICITY PRESERVED in all %lld updates ***\n\n",
               (long long)total_updates);
    else
        printf("\n!!! MONOTONICITY VIOLATED %lld times !!!\n\n",
               (long long)total_violations);
}

// ============================================================================
// TEST 2: Exhaustive enumeration for small n
// Enumerate all reduced BPDs, check all ordered pairs x all moves.
// ============================================================================

void enumerateAllReducedBPDs(int n, std::vector<BPD>& allBPDs) {
    std::set<std::vector<uint8_t>> visited;
    std::queue<BPD> frontier;

    // Start from identity
    BPD id(n);
    id.initIdentity();
    frontier.push(id);
    visited.insert(id.grid);

    // Also start from w0
    BPD w0(n);
    w0.initW0();
    if (visited.find(w0.grid) == visited.end()) {
        frontier.push(w0);
        visited.insert(w0.grid);
    }

    while (!frontier.empty()) {
        BPD cur = frontier.front();
        frontier.pop();

        // Try all moves at all positions
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                for (int d = 0; d < 2; d++) {
                    BPD next = applyUpdate(cur, i, j, d == 0);
                    if (visited.find(next.grid) == visited.end()) {
                        visited.insert(next.grid);
                        frontier.push(next);
                    }
                }
            }
        }
    }

    allBPDs.clear();
    allBPDs.reserve(visited.size());
    for (auto& g : visited) {
        BPD b(n);
        b.grid = g;
        allBPDs.push_back(b);
    }
}

void testExhaustive(int n) {
    printf("=== Exhaustive Monotonicity Test: n=%d ===\n", n);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: Enumerate all reduced BPDs
    printf("Enumerating all reduced BPDs for n=%d...\n", n);
    std::vector<BPD> allBPDs;
    enumerateAllReducedBPDs(n, allBPDs);
    printf("Found %zu reduced BPDs\n", allBPDs.size());

    // Step 2: Compute all height functions
    printf("Computing height functions...\n");
    std::vector<std::vector<int>> heights(allBPDs.size());
    for (size_t i = 0; i < allBPDs.size(); i++)
        heights[i] = computeHeight(allBPDs[i]);

    // Step 3: Find all ordered pairs (X <= Y)
    printf("Finding ordered pairs...\n");
    int64_t num_pairs = 0;
    int64_t num_moves = 2 * (n-1) * (n-1); // positions x directions

    // Step 4: For each pair and each move, check monotonicity
    int64_t total_checks = 0;
    int64_t violations = 0;
    int64_t contact_violations = 0; // violations specifically at contact points
    int64_t asymmetric_count = 0;   // X accepts, Y rejects or vice versa

    for (size_t a = 0; a < allBPDs.size(); a++) {
        for (size_t b = a; b < allBPDs.size(); b++) {
            // Check both orderings (a <= b) and (b <= a)
            for (int order = 0; order < 2; order++) {
                size_t xi = (order == 0) ? a : b;
                size_t yi = (order == 0) ? b : a;
                if (xi == yi && order == 1) continue; // skip duplicate diagonal

                if (!heightLeq(heights[xi], heights[yi], n)) continue;
                num_pairs++;

                // Test all possible moves
                for (int mi = 1; mi < n; mi++) {
                    for (int mj = 1; mj < n; mj++) {
                        for (int d = 0; d < 2; d++) {
                            bool isUp = (d == 0);
                            BPD newX = applyUpdate(allBPDs[xi], mi, mj, isUp);
                            BPD newY = applyUpdate(allBPDs[yi], mi, mj, isUp);

                            auto hNewX = computeHeight(newX);
                            auto hNewY = computeHeight(newY);

                            total_checks++;

                            bool x_moved = !(newX == allBPDs[xi]);
                            bool y_moved = !(newY == allBPDs[yi]);
                            if (x_moved != y_moved) asymmetric_count++;

                            int vi, vj, diff;
                            if (findViolation(hNewX, hNewY, n, vi, vj, diff)) {
                                violations++;
                                if (violations <= 10) {
                                    printf("  VIOLATION #%lld:\n", (long long)violations);
                                    printf("    X perm: ");
                                    auto px = allBPDs[xi].perm();
                                    for (int x : px) printf("%d ", x);
                                    printf("  (ell=%d)\n", allBPDs[xi].numCrosses());
                                    printf("    Y perm: ");
                                    auto py = allBPDs[yi].perm();
                                    for (int x : py) printf("%d ", x);
                                    printf("  (ell=%d)\n", allBPDs[yi].numCrosses());
                                    printf("    Move: pos=(%d,%d) dir=%s\n",
                                           mi, mj, isUp ? "UP" : "DOWN");
                                    printf("    X moved=%d, Y moved=%d\n",
                                           x_moved, y_moved);
                                    printf("    Violation at h(%d,%d): "
                                           "h_X'=%d < h_Y'=%d\n",
                                           vi, vj, hNewX[vi*(n+1)+vj],
                                           hNewY[vi*(n+1)+vj]);

                                    // Check if this was a contact point before
                                    int hx_before = heights[xi][vi*(n+1)+vj];
                                    int hy_before = heights[yi][vi*(n+1)+vj];
                                    printf("    Before: h_X(%d,%d)=%d, h_Y(%d,%d)=%d %s\n",
                                           vi, vj, hx_before, vi, vj, hy_before,
                                           (hx_before == hy_before) ? "[CONTACT]" : "");
                                    if (hx_before == hy_before) contact_violations++;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Progress
        if ((a+1) % std::max((size_t)1, allBPDs.size()/10) == 0) {
            printf("  Progress: %zu/%zu states as X...\n", a+1, allBPDs.size());
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    printf("\n--- Exhaustive Test Summary (n=%d) ---\n", n);
    printf("Reduced BPDs:        %zu\n", allBPDs.size());
    printf("Ordered pairs:       %lld\n", (long long)num_pairs);
    printf("Total move checks:   %lld\n", (long long)total_checks);
    printf("Asymmetric accepts:  %lld\n", (long long)asymmetric_count);
    printf("VIOLATIONS:          %lld\n", (long long)violations);
    printf("  at contact points: %lld\n", (long long)contact_violations);
    printf("Time:                %.2fs\n", elapsed);

    if (violations == 0)
        printf("\n*** MONOTONICITY PROVED for n=%d: 0 violations in %lld checks ***\n\n",
               n, (long long)total_checks);
    else
        printf("\n!!! VIOLATIONS FOUND: %lld !!!\n\n", (long long)violations);
}

void testCanonicalCounterexample() {
    const int n = 4;
    const int move_i = 3;
    const int move_j = 3;
    const bool isUp = true;

    BPD X(n), Y(n);
    X.initRothe(std::vector<int>{3, 1, 2, 4});
    Y.initRothe(std::vector<int>{3, 2, 1, 4});

    auto hX = computeHeight(X);
    auto hY = computeHeight(Y);

    printf("=== Canonical n=4 counterexample check ===\n");
    printf("X perm: ");
    auto px = X.perm();
    for (int v : px) printf("%d ", v);
    printf("(ell=%d)\n", X.numCrosses());
    printf("Y perm: ");
    auto py = Y.perm();
    for (int v : py) printf("%d ", v);
    printf("(ell=%d)\n\n", Y.numCrosses());

    printf("Initial order X <= Y (height order): %s\n", heightLeq(hX, hY, n) ? "YES" : "NO");
    printf("Initial contact at v=(3,3): h_X=%d, h_Y=%d\n",
           hX[3*(n+1)+3], hY[3*(n+1)+3]);

    BPD Xp = applyUpdate(X, move_i, move_j, isUp);
    BPD Yp = applyUpdate(Y, move_i, move_j, isUp);

    bool x_moved = !(Xp == X);
    bool y_moved = !(Yp == Y);
    printf("\nMove: UP at (%d,%d)\n", move_i, move_j);
    printf("X moved=%d, Y moved=%d\n", x_moved, y_moved);

    auto hXp = computeHeight(Xp);
    auto hYp = computeHeight(Yp);
    bool ok = heightLeq(hXp, hYp, n);
    printf("After move X' <= Y' (height order): %s\n", ok ? "YES" : "NO");
    printf("At v=(3,3): h_X'=%d, h_Y'=%d\n",
           hXp[3*(n+1)+3], hYp[3*(n+1)+3]);

    int vi, vj, diff;
    if (findViolation(hXp, hYp, n, vi, vj, diff)) {
        printf("First violation at v=(%d,%d): h_X'=%d < h_Y'=%d (diff=%d)\n",
               vi, vj, hXp[vi*(n+1)+vj], hYp[vi*(n+1)+vj], diff);
    } else {
        printf("No violation detected for this move.\n");
    }
    printf("\nX before (tiles):\n"); X.print();
    printf("Y before (tiles):\n"); Y.print();
    printf("X after (tiles):\n"); Xp.print();
    printf("Y after (tiles):\n"); Yp.print();
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    printf("Monotonicity Check for BPD CFTP\n\n");
    printf("Usage:\n");
    printf("  %s cftp <n> [trials]    CFTP monitoring (default: 100 trials)\n", prog);
    printf("  %s exhaust <n>          Exhaustive test (n <= 5 recommended)\n", prog);
    printf("  %s both <n> [trials]    Both tests\n", prog);
    printf("  %s height <n>           Print height functions for identity and w0\n", prog);
    printf("  %s counterexample       Reproduce canonical n=4 monotonicity violation\n", prog);
    printf("\nExamples:\n");
    printf("  %s cftp 6 200           200 CFTP trials at n=6\n", prog);
    printf("  %s exhaust 4            Exhaustive proof for n=4\n", prog);
    printf("  %s both 4 100           Both tests for n=4\n", prog);
}

void printHeights(int n) {
    printf("=== Height Functions for n=%d ===\n\n", n);

    BPD id(n); id.initIdentity();
    BPD w0(n); w0.initW0();

    printf("Identity BPD (tiles):\n"); id.print();
    printf("Permutation: ");
    for (int x : id.perm()) printf("%d ", x);
    printf("\n\n");

    printf("w0 BPD (tiles):\n"); w0.print();
    printf("Permutation: ");
    for (int x : w0.perm()) printf("%d ", x);
    printf("\n\n");

    auto hId = computeHeight(id);
    auto hW0 = computeHeight(w0);

    printf("h_identity:\n");
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++)
            printf("%3d", hId[i*(n+1)+j]);
        printf("\n");
    }

    printf("\nh_w0:\n");
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++)
            printf("%3d", hW0[i*(n+1)+j]);
        printf("\n");
    }

    printf("\nh_identity - h_w0 (should be >= 0):\n");
    bool ok = true;
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            int d = hId[i*(n+1)+j] - hW0[i*(n+1)+j];
            printf("%3d", d);
            if (d < 0) ok = false;
        }
        printf("\n");
    }
    printf("\nh_identity >= h_w0: %s\n", ok ? "YES" : "NO (BUG!)");
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    if (strcmp(argv[1], "height") == 0) {
        int n = (argc >= 3) ? atoi(argv[2]) : 4;
        printHeights(n);
        return 0;
    }

    if (strcmp(argv[1], "cftp") == 0) {
        int n = (argc >= 3) ? atoi(argv[2]) : 6;
        int trials = (argc >= 4) ? atoi(argv[3]) : 100;
        testCFTPMonitoring(n, trials);
        return 0;
    }

    if (strcmp(argv[1], "exhaust") == 0) {
        int n = (argc >= 3) ? atoi(argv[2]) : 3;
        testExhaustive(n);
        return 0;
    }

    if (strcmp(argv[1], "both") == 0) {
        int n = (argc >= 3) ? atoi(argv[2]) : 4;
        int trials = (argc >= 4) ? atoi(argv[3]) : 100;
        testExhaustive(n);
        testCFTPMonitoring(n, trials);
        return 0;
    }

    if (strcmp(argv[1], "counterexample") == 0) {
        testCanonicalCounterexample();
        return 0;
    }

    printUsage(argv[0]);
    return 1;
}
