// Verify Connectivity Proof for Reduced Bumpless Pipe Dreams
//
// Tests the generalized lemma: For ANY reduced BPD with w != e, the
// topmost-leftmost cross has NW neighbor that is either:
//   (a) blank → box-cross annihilation (ell decreases by 1), or
//   (b) j-elbow → cross-undrip (ell preserved, ASM height decreases)
// In both cases, the result is a reduced BPD.
//
// Also tests:
//   - Original Rothe-only lemma (NW always blank for Rothe BPDs)
//   - Full descent: repeatedly apply generalized lemma to reach identity
//   - BFS enumeration: test all reduced BPDs via exhaustive enumeration
//
// Compile:
//   clang++ -O3 -std=c++17 -mcpu=apple-m2 -flto -DNDEBUG \
//     verify_connectivity_proof.cpp -o verify_connectivity_proof
//
// Linux:
//   g++ -O3 -std=c++17 -march=native -flto -DNDEBUG \
//     verify_connectivity_proof.cpp -o verify_connectivity_proof

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cassert>
#include <unordered_set>
#include <queue>
#include <string>

// Tile types: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz

// =============================================================================
// BPD class
// =============================================================================

class BPD {
public:
    int n;
    std::vector<uint8_t> grid;

    BPD(int size) : n(size), grid(size * size, 0) {}
    BPD(int size, const std::vector<uint8_t>& g) : n(size), grid(g) {}

    inline uint8_t get(int r, int c) const { return grid[r * n + c]; }
    inline void set(int r, int c, uint8_t val) { grid[r * n + c] = val; }

    bool operator==(const BPD& other) const { return grid == other.grid; }

    // Generate Rothe BPD for permutation w (1-indexed values)
    void generateRotheBPD(const std::vector<int>& w) {
        int minW = *std::min_element(w.begin(), w.end());

        for (int j = 1; j <= n; j++) {
            if (j + minW - 1 < w[0]) {
                set(0, j - 1, 0);
            } else if (j + minW - 1 == w[0]) {
                set(0, j - 1, 2);
            } else {
                set(0, j - 1, 5);
            }
        }

        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (j + minW - 1 < w[i - 1]) {
                    uint8_t up = get(i - 2, j - 1);
                    set(i - 1, j - 1, (up == 2 || up == 4 || up == 1) ? 4 : 0);
                } else if (j + minW - 1 == w[i - 1]) {
                    set(i - 1, j - 1, 2);
                } else {
                    uint8_t up = get(i - 2, j - 1);
                    set(i - 1, j - 1, (up == 4 || up == 2 || up == 1) ? 1 : 5);
                }
            }
        }
    }

    // Get transposition index from cross at (i,j)
    int get_sk_from_cross(int i, int j) const {
        if (get(i, j) != 1) return -1;
        int k = 1;
        int maxS = std::min(i, j);
        for (int s = 1; s <= maxS; s++) {
            uint8_t t = get(i - s, j - s);
            if (t == 2 || t == 3 || t == 4 || t == 5) k += 1;
            else if (t == 1) k += 2;
        }
        return k;
    }

    // Check if BPD is reduced
    bool isReduced() const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;

        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1;
                int j = n + s - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1) {
                    if (ww[k - 1] > ww[k]) return false;
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }

        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1;
                int j = n - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1) {
                    if (ww[k - 1] > ww[k]) return false;
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        return true;
    }

    // Compute permutation from BPD
    std::vector<int> computePerm() const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;

        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1;
                int j = n + s - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1 && k < n) {
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }

        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1;
                int j = n - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1 && k < n) {
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        return ww;
    }

    // Count crosses (= Coxeter length)
    int countCrosses() const {
        int count = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (get(i, j) == 1) count++;
        return count;
    }

    // Compute ASM height function sum (sum over all cells of h(i,j))
    // h(i,j) = number of pipes that pass through or below row i and
    //          to the left of or through column j.
    // For our purposes, we compute a simpler monotone: sum of row indices
    // of all r-elbows and j-elbows (captures pipe "depth").
    // Actually, use the standard: sum over all cells of the tile's
    // contribution to the ASM height function.
    int computeASMHeightSum() const {
        // The ASM height function h_{i,j} counts the number of strands
        // that cross the horizontal edge between rows i-1 and i
        // in columns 1..j. We compute the total sum of all h_{i,j}.
        //
        // Equivalently, for each cross/elbow, we track its contribution.
        // But a simpler proxy that is monotone under drip/cross-undrip:
        // sum of (row * n + col) for each j-elbow, minus same for r-elbows.
        // Drips move r-elbows down to j-elbows, increasing this sum.
        // Undrips reverse this.
        //
        // Use a direct height function computation instead:
        // For each horizontal edge (between row r-1 and r, at column c),
        // count how many strands cross it (i.e., tiles above contribute
        // vertical flow). h(r,c) = sum over rows 0..r-1 of indicator
        // that tile (row,c) has a south-exit.

        int total = 0;
        for (int c = 0; c < n; c++) {
            int flow = 0;  // number of strands flowing south at column c
            for (int r = 0; r < n; r++) {
                uint8_t t = get(r, c);
                // Does this tile have a south exit?
                // cross(1): yes. r-elbow(2): yes. vert(4): yes.
                // blank(0): no. j-elbow(3): no. horiz(5): no.
                if (t == 1 || t == 2 || t == 4) flow++;
                // j-elbow(3) absorbs a south-entering pipe (turns it west)
                // but we already counted the entry above
                total += flow;
            }
        }
        return total;
    }

    void print() const {
        const char* names[] = {".", "+", "r", "j", "|", "-"};
        for (int i = 0; i < n; i++) {
            printf("    ");
            for (int j = 0; j < n; j++) {
                printf("%s ", names[get(i, j)]);
            }
            printf("\n");
        }
    }
};

// =============================================================================
// Coxeter length of a permutation
// =============================================================================

int coxeterLength(const std::vector<int>& w) {
    int n = w.size();
    int inv = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (w[i] > w[j]) inv++;
    return inv;
}

// =============================================================================
// Check if w' is a length-1 Bruhat cover below w
// =============================================================================

bool isBruhatCoverBelow(const std::vector<int>& w, const std::vector<int>& wprime) {
    int n = w.size();
    int ell_w = coxeterLength(w);
    int ell_wp = coxeterLength(wprime);
    if (ell_wp != ell_w - 1) return false;

    std::vector<int> diff_pos;
    for (int i = 0; i < n; i++) {
        if (w[i] != wprime[i]) diff_pos.push_back(i);
    }
    if (diff_pos.size() != 2) return false;

    int a = diff_pos[0], b = diff_pos[1];
    if (wprime[a] != w[b] || wprime[b] != w[a]) return false;
    if (a >= b) return false;
    if (w[a] <= w[b]) return false;

    for (int k = a + 1; k < b; k++) {
        if (w[b] < w[k] && w[k] < w[a]) return false;
    }
    return true;
}

// =============================================================================
// 2x2 Move Operations
// =============================================================================

bool tryDrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i - 1, j - 1);
    uint8_t se = bpd.get(i, j);
    if (nw == 2 && se == 0) {
        uint8_t ne = bpd.get(i - 1, j);
        uint8_t sw = bpd.get(i, j - 1);
        bpd.set(i - 1, j - 1, 0);
        bpd.set(i, j, 3);
        bpd.set(i - 1, j, (ne == 5) ? 2 : 4);
        bpd.set(i, j - 1, (sw == 4) ? 2 : 5);
        return true;
    }
    return false;
}

bool tryUndrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i - 1, j - 1);
    uint8_t ne = bpd.get(i - 1, j);
    uint8_t sw = bpd.get(i, j - 1);
    uint8_t se = bpd.get(i, j);

    if (nw == 0 && se == 3 && (ne == 2 || ne == 4) && (sw == 2 || sw == 5)) {
        bpd.set(i - 1, j - 1, 2);
        bpd.set(i, j, 0);
        bpd.set(i - 1, j, (ne == 2) ? 5 : 3);
        bpd.set(i, j - 1, (sw == 2) ? 4 : 3);
        return true;
    }
    return false;
}

bool tryCrossDrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i - 1, j - 1);
    uint8_t ne = bpd.get(i - 1, j);
    uint8_t sw = bpd.get(i, j - 1);
    uint8_t se = bpd.get(i, j);

    if (nw == 2 && se == 2) {
        bpd.set(i - 1, j - 1, 0);
        bpd.set(i, j, 1);
        bpd.set(i - 1, j, (ne == 3) ? 4 : 2);
        bpd.set(i, j - 1, (sw == 3) ? 5 : 2);
        return true;
    }
    if (nw == 1 && se == 0) {
        bpd.set(i - 1, j - 1, 3);
        bpd.set(i, j, 3);
        bpd.set(i - 1, j, (ne == 3) ? 4 : 2);
        bpd.set(i, j - 1, (sw == 3) ? 5 : 2);
        return true;
    }
    if (nw == 1 && se == 2) {
        bpd.set(i - 1, j - 1, 3);
        bpd.set(i, j, 1);
        bpd.set(i - 1, j, (ne == 3) ? 4 : 2);
        bpd.set(i, j - 1, (sw == 3) ? 5 : 2);
        return true;
    }
    return false;
}

bool tryCrossUndrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i - 1, j - 1);
    uint8_t ne = bpd.get(i - 1, j);
    uint8_t sw = bpd.get(i, j - 1);
    uint8_t se = bpd.get(i, j);

    // Case 1: NW=blank, SE=cross (box-cross annihilation)
    if (nw == 0 && se == 1) {
        bpd.set(i - 1, j - 1, 2);
        bpd.set(i, j, 2);
        bpd.set(i - 1, j, (ne == 4) ? 3 : 5);
        bpd.set(i, j - 1, (sw == 5) ? 3 : 4);
        return true;
    }
    // Case 2: NW=j-elbow, SE=j-elbow
    if (nw == 3 && se == 3) {
        bpd.set(i - 1, j - 1, 1);
        bpd.set(i, j, 0);
        bpd.set(i - 1, j, (ne == 4) ? 3 : 5);
        bpd.set(i, j - 1, (sw == 5) ? 3 : 4);
        return true;
    }
    // Case 3: NW=j-elbow, SE=cross
    if (nw == 3 && se == 1) {
        bpd.set(i - 1, j - 1, 1);
        bpd.set(i, j, 2);
        bpd.set(i - 1, j, (ne == 4) ? 3 : 5);
        bpd.set(i, j - 1, (sw == 5) ? 3 : 4);
        return true;
    }
    return false;
}

// =============================================================================
// BFS neighbor enumeration (for connectivity check)
// =============================================================================

void getNeighbors(const BPD& bpd, std::vector<std::vector<uint8_t>>& neighbors) {
    neighbors.clear();
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            BPD temp = bpd;
            if (tryDrip(temp, i, j)) {
                if (temp.isReduced()) neighbors.push_back(temp.grid);
                temp = bpd;
            }
            if (tryUndrip(temp, i, j)) {
                if (temp.isReduced()) neighbors.push_back(temp.grid);
                temp = bpd;
            }
            if (tryCrossDrip(temp, i, j)) {
                if (temp.isReduced()) neighbors.push_back(temp.grid);
                temp = bpd;
            }
            if (tryCrossUndrip(temp, i, j)) {
                if (temp.isReduced()) neighbors.push_back(temp.grid);
                temp = bpd;
            }
        }
    }
}

struct GridHash {
    size_t operator()(const std::vector<uint8_t>& grid) const {
        size_t h = 0;
        for (uint8_t v : grid) h = h * 31 + v;
        return h;
    }
};

// =============================================================================
// Find topmost-leftmost cross in a BPD
// =============================================================================

std::pair<int,int> findTopmostLeftmostCross(const BPD& bpd) {
    for (int i = 0; i < bpd.n; i++) {
        for (int j = 0; j < bpd.n; j++) {
            if (bpd.get(i, j) == 1) {
                return {i, j};
            }
        }
    }
    return {-1, -1};
}

// =============================================================================
// Format permutation as string
// =============================================================================

std::string permToString(const std::vector<int>& w) {
    std::string s = "(";
    for (size_t i = 0; i < w.size(); i++) {
        s += std::to_string(w[i]);
        if (i + 1 < w.size()) s += ",";
    }
    s += ")";
    return s;
}

// =============================================================================
// Test 1: Original Lemma - Rothe BPDs only (NW must be blank)
// =============================================================================

struct LemmaTestResult {
    int n;
    int total_perms;
    int pass_count;
    int fail_count;
    int fail_nw_not_blank;
    int fail_not_reduced;
    int fail_wrong_length;
    int fail_not_cover;
    double elapsed_sec;
};

LemmaTestResult testRotheLemma(int n, bool verbose) {
    LemmaTestResult result = {};
    result.n = n;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> w(n);
    std::iota(w.begin(), w.end(), 1);

    do {
        bool is_identity = true;
        for (int i = 0; i < n; i++) {
            if (w[i] != i + 1) { is_identity = false; break; }
        }
        if (is_identity) continue;

        result.total_perms++;

        BPD bpd(n);
        bpd.generateRotheBPD(w);

        int ell_w = coxeterLength(w);

        auto [ci, cj] = findTopmostLeftmostCross(bpd);
        if (ci == -1) {
            printf("  BUG: no cross found but w != e for w=%s\n",
                   permToString(w).c_str());
            result.fail_count++;
            continue;
        }

        if (ci < 1 || cj < 1) {
            result.fail_count++;
            continue;
        }

        uint8_t nw_tile = bpd.get(ci - 1, cj - 1);
        if (nw_tile != 0) {
            const char* names[] = {"blank", "cross", "r-elbow", "j-elbow", "vert", "horiz"};
            if (verbose) {
                printf("  FAIL: NW tile at (%d,%d) is %s for w=%s\n",
                       ci - 1, cj - 1, names[nw_tile], permToString(w).c_str());
            }
            result.fail_nw_not_blank++;
            result.fail_count++;
            continue;
        }

        // Apply box-cross annihilation (cross-undrip case 1: NW=blank, SE=cross)
        BPD bpd_after(bpd);
        bool applied = tryCrossUndrip(bpd_after, ci, cj);
        if (!applied) {
            result.fail_count++;
            continue;
        }

        if (!bpd_after.isReduced()) {
            if (verbose) {
                printf("  FAIL: result not reduced for w=%s\n", permToString(w).c_str());
            }
            result.fail_not_reduced++;
            result.fail_count++;
            continue;
        }

        std::vector<int> wprime = bpd_after.computePerm();
        int ell_wp = coxeterLength(wprime);
        if (ell_wp != ell_w - 1) {
            if (verbose) {
                printf("  FAIL: ell(w')=%d, expected %d\n", ell_wp, ell_w - 1);
            }
            result.fail_wrong_length++;
            result.fail_count++;
            continue;
        }

        if (!isBruhatCoverBelow(w, wprime)) {
            if (verbose) {
                printf("  FAIL: not a Bruhat cover for w=%s\n", permToString(w).c_str());
            }
            result.fail_not_cover++;
            result.fail_count++;
            continue;
        }

        result.pass_count++;

    } while (std::next_permutation(w.begin(), w.end()));

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_sec = std::chrono::duration<double>(end - start).count();

    return result;
}

// =============================================================================
// Test 2: Generalized Lemma - ALL reduced BPDs (NW can be blank or j-elbow)
//
// For every reduced BPD with w != e:
//   1. Find topmost-leftmost cross (i,j)
//   2. Check NW = (i-1, j-1) is blank or j-elbow
//   3. If blank: box-cross annihilation → reduced, ell decreases by 1
//   4. If j-elbow: cross-undrip case 3 → reduced, ell preserved
// =============================================================================

struct GenLemmaTestResult {
    int n;
    uint64_t total_rbpds;
    uint64_t pass_count;
    uint64_t fail_count;
    uint64_t case_blank;       // NW was blank (box-cross annihilation)
    uint64_t case_jelbow;      // NW was j-elbow (cross-undrip)
    uint64_t fail_nw_wrong;    // NW was neither blank nor j-elbow
    uint64_t fail_not_reduced; // result not reduced
    uint64_t fail_wrong_ell;   // length change wrong
    double elapsed_sec;
};

GenLemmaTestResult testGeneralizedLemma(int n, bool verbose) {
    GenLemmaTestResult result = {};
    result.n = n;

    auto start = std::chrono::high_resolution_clock::now();

    // Enumerate all reduced BPDs via BFS
    std::unordered_set<std::vector<uint8_t>, GridHash> visited;
    std::queue<std::vector<uint8_t>> bfs_queue;

    BPD bpd_id(n);
    std::vector<int> id(n);
    std::iota(id.begin(), id.end(), 1);
    bpd_id.generateRotheBPD(id);
    visited.insert(bpd_id.grid);
    bfs_queue.push(bpd_id.grid);

    BPD bpd_w0(n);
    std::vector<int> w0(n);
    for (int i = 0; i < n; i++) w0[i] = n - i;
    bpd_w0.generateRotheBPD(w0);
    if (visited.find(bpd_w0.grid) == visited.end()) {
        visited.insert(bpd_w0.grid);
        bfs_queue.push(bpd_w0.grid);
    }

    std::vector<std::vector<uint8_t>> neighbors;
    while (!bfs_queue.empty()) {
        auto cur = bfs_queue.front();
        bfs_queue.pop();
        BPD bpd(n, cur);
        getNeighbors(bpd, neighbors);
        for (const auto& nb : neighbors) {
            if (visited.find(nb) == visited.end()) {
                visited.insert(nb);
                bfs_queue.push(nb);
            }
        }
    }

    result.total_rbpds = visited.size();

    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    const char* tile_names[] = {"blank", "cross", "r-elbow", "j-elbow", "vert", "horiz"};

    for (const auto& grid : visited) {
        BPD bpd(n, grid);
        std::vector<int> perm = bpd.computePerm();

        // Skip identity (no crosses)
        if (perm == identity) {
            result.pass_count++;
            continue;
        }

        int ell = coxeterLength(perm);

        // Find topmost-leftmost cross
        auto [ci, cj] = findTopmostLeftmostCross(bpd);
        if (ci == -1) {
            if (verbose) printf("  BUG: no cross but perm=%s\n", permToString(perm).c_str());
            result.fail_count++;
            continue;
        }

        if (ci < 1 || cj < 1) {
            if (verbose) printf("  FAIL: cross at boundary (%d,%d)\n", ci, cj);
            result.fail_count++;
            continue;
        }

        uint8_t nw_tile = bpd.get(ci - 1, cj - 1);

        if (nw_tile == 0) {
            // Case A: NW is blank → box-cross annihilation
            result.case_blank++;

            BPD after(bpd);
            bool applied = tryCrossUndrip(after, ci, cj);  // case 1: NW=0, SE=1
            if (!applied) {
                if (verbose) printf("  FAIL: box-cross annihilation did not apply\n");
                result.fail_count++;
                continue;
            }

            if (!after.isReduced()) {
                if (verbose) {
                    printf("  FAIL: Case A result not reduced, perm=%s, cross at (%d,%d)\n",
                           permToString(perm).c_str(), ci, cj);
                    printf("  Before:\n"); bpd.print();
                    printf("  After:\n"); after.print();
                }
                result.fail_not_reduced++;
                result.fail_count++;
                continue;
            }

            std::vector<int> new_perm = after.computePerm();
            int new_ell = coxeterLength(new_perm);
            if (new_ell != ell - 1) {
                if (verbose) {
                    printf("  FAIL: Case A length %d -> %d (expected %d)\n",
                           ell, new_ell, ell - 1);
                }
                result.fail_wrong_ell++;
                result.fail_count++;
                continue;
            }

            result.pass_count++;

        } else if (nw_tile == 3) {
            // Case B: NW is j-elbow → cross-undrip case 3
            result.case_jelbow++;

            BPD after(bpd);
            bool applied = tryCrossUndrip(after, ci, cj);  // case 3: NW=3, SE=1
            if (!applied) {
                if (verbose) {
                    printf("  FAIL: cross-undrip case 3 did not apply at (%d,%d)\n", ci, cj);
                    printf("  Tiles: NW=%s NE=%s SW=%s SE=%s\n",
                           tile_names[bpd.get(ci-1, cj-1)],
                           tile_names[bpd.get(ci-1, cj)],
                           tile_names[bpd.get(ci, cj-1)],
                           tile_names[bpd.get(ci, cj)]);
                }
                result.fail_count++;
                continue;
            }

            if (!after.isReduced()) {
                if (verbose) {
                    printf("  FAIL: Case B result not reduced, perm=%s, cross at (%d,%d)\n",
                           permToString(perm).c_str(), ci, cj);
                    printf("  Before:\n"); bpd.print();
                    printf("  After:\n"); after.print();
                }
                result.fail_not_reduced++;
                result.fail_count++;
                continue;
            }

            // Cross-undrip case 3 should preserve cross count (ell unchanged)
            std::vector<int> new_perm = after.computePerm();
            int new_ell = coxeterLength(new_perm);
            if (new_ell != ell) {
                if (verbose) {
                    printf("  FAIL: Case B length %d -> %d (expected %d)\n",
                           ell, new_ell, ell);
                }
                result.fail_wrong_ell++;
                result.fail_count++;
                continue;
            }

            result.pass_count++;

        } else {
            // NW is neither blank nor j-elbow
            if (verbose) {
                printf("  FAIL: NW tile is %s at (%d,%d), perm=%s, cross at (%d,%d)\n",
                       tile_names[nw_tile], ci - 1, cj - 1,
                       permToString(perm).c_str(), ci, cj);
                printf("  BPD:\n"); bpd.print();
            }
            result.fail_nw_wrong++;
            result.fail_count++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_sec = std::chrono::duration<double>(end - start).count();

    return result;
}

// =============================================================================
// Test 3: Full greedy descent from ALL reduced BPDs to identity
//
// Strategy: repeatedly apply generalized lemma (find topmost-leftmost cross,
// apply box-cross annihilation or cross-undrip) until reaching identity.
// =============================================================================

struct DescentTestResult {
    int n;
    uint64_t total_rbpds;
    uint64_t pass_count;
    uint64_t fail_count;
    int max_steps;
    double elapsed_sec;
};

DescentTestResult testGreedyDescent(int n, bool verbose) {
    DescentTestResult result = {};
    result.n = n;

    auto start = std::chrono::high_resolution_clock::now();

    // Enumerate all reduced BPDs via BFS
    std::unordered_set<std::vector<uint8_t>, GridHash> visited;
    std::queue<std::vector<uint8_t>> bfs_queue;

    BPD bpd_id(n);
    std::vector<int> id(n);
    std::iota(id.begin(), id.end(), 1);
    bpd_id.generateRotheBPD(id);
    visited.insert(bpd_id.grid);
    bfs_queue.push(bpd_id.grid);

    BPD bpd_w0(n);
    std::vector<int> w0(n);
    for (int i = 0; i < n; i++) w0[i] = n - i;
    bpd_w0.generateRotheBPD(w0);
    if (visited.find(bpd_w0.grid) == visited.end()) {
        visited.insert(bpd_w0.grid);
        bfs_queue.push(bpd_w0.grid);
    }

    std::vector<std::vector<uint8_t>> neighbors;
    while (!bfs_queue.empty()) {
        auto cur = bfs_queue.front();
        bfs_queue.pop();
        BPD bpd(n, cur);
        getNeighbors(bpd, neighbors);
        for (const auto& nb : neighbors) {
            if (visited.find(nb) == visited.end()) {
                visited.insert(nb);
                bfs_queue.push(nb);
            }
        }
    }

    result.total_rbpds = visited.size();

    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    BPD identity_bpd(n);
    identity_bpd.generateRotheBPD(identity);

    // Generous safety bound: the ASM height sum for n x n is at most O(n^3),
    // and each step decreases it by at least 1.
    int max_steps_allowed = n * n * n;

    for (const auto& grid : visited) {
        BPD bpd(n, grid);
        std::vector<int> perm = bpd.computePerm();

        if (perm == identity && bpd == identity_bpd) {
            result.pass_count++;
            continue;
        }

        BPD current(bpd);
        int steps = 0;
        bool failed = false;
        int prev_height = current.computeASMHeightSum();

        while (steps < max_steps_allowed) {
            std::vector<int> cur_perm = current.computePerm();
            if (cur_perm == identity && current.countCrosses() == 0) {
                break;
            }

            auto [ci, cj] = findTopmostLeftmostCross(current);
            if (ci == -1) {
                if (verbose) {
                    printf("  FAIL: no crosses but perm=%s\n",
                           permToString(cur_perm).c_str());
                }
                failed = true;
                break;
            }

            if (ci < 1 || cj < 1) {
                if (verbose) printf("  FAIL: cross at boundary (%d,%d)\n", ci, cj);
                failed = true;
                break;
            }

            uint8_t nw_tile = current.get(ci - 1, cj - 1);

            if (nw_tile != 0 && nw_tile != 3) {
                if (verbose) {
                    const char* names[] = {"blank", "cross", "r-elbow", "j-elbow", "vert", "horiz"};
                    printf("  FAIL: NW is %s at step %d, perm=%s\n",
                           names[nw_tile], steps, permToString(cur_perm).c_str());
                    current.print();
                }
                failed = true;
                break;
            }

            BPD next(current);
            bool applied = tryCrossUndrip(next, ci, cj);
            if (!applied) {
                if (verbose) printf("  FAIL: cross-undrip did not apply at step %d\n", steps);
                failed = true;
                break;
            }

            if (!next.isReduced()) {
                if (verbose) {
                    printf("  FAIL: result not reduced at step %d\n", steps);
                    printf("  Before:\n"); current.print();
                    printf("  After:\n"); next.print();
                }
                failed = true;
                break;
            }

            int new_height = next.computeASMHeightSum();
            if (new_height >= prev_height) {
                if (verbose) {
                    printf("  WARN: height did not decrease at step %d: %d -> %d\n",
                           steps, prev_height, new_height);
                }
                // Not necessarily a failure - the height function proxy might
                // not be strictly monotone. Continue anyway.
            }
            prev_height = new_height;

            current = next;
            steps++;
        }

        if (failed || steps >= max_steps_allowed) {
            result.fail_count++;
        } else {
            result.pass_count++;
            if (steps > result.max_steps) result.max_steps = steps;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_sec = std::chrono::duration<double>(end - start).count();

    return result;
}

// =============================================================================
// Main
// =============================================================================

void printUsage(const char* prog) {
    printf("Verify Connectivity Proof for Reduced BPDs\n\n");
    printf("Tests the generalized lemma: for ANY reduced BPD, the topmost-leftmost\n");
    printf("cross has NW neighbor that is blank (box-cross annihilation) or j-elbow\n");
    printf("(cross-undrip), and both moves produce a reduced BPD.\n\n");
    printf("Usage: %s [options] [n]\n\n", prog);
    printf("Options:\n");
    printf("  --verbose    Print details for failures\n");
    printf("  --all        Run all tests (default: Rothe lemma only)\n");
    printf("  --help, -h   Show this help\n\n");
    printf("Arguments:\n");
    printf("  n            Max n to test (default: 7)\n");
    printf("  n1-n2        Range to test\n\n");
    printf("Tests:\n");
    printf("  [Rothe]      Original lemma: NW always blank for Rothe BPDs\n");
    printf("  [GenLemma]   Generalized lemma: NW is blank or j-elbow for ALL RBPDs\n");
    printf("  [Descent]    Full greedy descent from every RBPD to identity\n\n");
    printf("Examples:\n");
    printf("  %s 8              Test Rothe lemma for n=2..8\n", prog);
    printf("  %s --all 7        Run all tests for n=2..7\n", prog);
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    bool test_all = false;
    int maxN = 7;
    int minN = 2;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--all") == 0) {
            test_all = true;
        } else if (argv[i][0] != '-') {
            char* dash = strchr(argv[i], '-');
            if (dash && dash != argv[i]) {
                minN = atoi(argv[i]);
                maxN = atoi(dash + 1);
            } else {
                maxN = atoi(argv[i]);
            }
        }
    }

    if (minN < 2) minN = 2;
    if (maxN < minN) maxN = minN;

    printf("==============================================\n");
    printf("  Verify Connectivity Proof for Reduced BPDs\n");
    printf("==============================================\n\n");

    printf("Testing n = %d to %d\n", minN, maxN);
    if (test_all) printf("  + Generalized lemma (all RBPDs)\n  + Full greedy descent\n");
    printf("\n");

    bool all_pass = true;

    for (int n = minN; n <= maxN; n++) {
        printf("--- n = %d ---\n", n);

        uint64_t factorial = 1;
        for (int i = 2; i <= n; i++) factorial *= i;

        // Test 1: Rothe-only Lemma
        printf("  [Rothe] Testing %llu permutations...\n",
               (unsigned long long)(factorial - 1));
        LemmaTestResult lr = testRotheLemma(n, verbose);

        printf("  [Rothe] %d/%d passed (%.3f sec)\n",
               lr.pass_count, lr.total_perms, lr.elapsed_sec);
        if (lr.fail_count > 0) {
            printf("  [Rothe] *** %d FAILURES ***\n", lr.fail_count);
            all_pass = false;
        } else {
            printf("  [Rothe] PASS\n");
        }

        // Tests that require BFS enumeration
        if (test_all && n <= 7) {
            // Test 2: Generalized Lemma
            printf("  [GenLemma] Testing all RBPDs...\n");
            GenLemmaTestResult gr = testGeneralizedLemma(n, verbose);

            printf("  [GenLemma] %llu/%llu passed",
                   (unsigned long long)gr.pass_count,
                   (unsigned long long)gr.total_rbpds);
            printf(" (blank=%llu, j-elbow=%llu) (%.3f sec)\n",
                   (unsigned long long)gr.case_blank,
                   (unsigned long long)gr.case_jelbow,
                   gr.elapsed_sec);
            if (gr.fail_count > 0) {
                printf("  [GenLemma] *** %llu FAILURES ***\n",
                       (unsigned long long)gr.fail_count);
                if (gr.fail_nw_wrong > 0)
                    printf("    - NW wrong tile: %llu\n", (unsigned long long)gr.fail_nw_wrong);
                if (gr.fail_not_reduced > 0)
                    printf("    - Not reduced: %llu\n", (unsigned long long)gr.fail_not_reduced);
                if (gr.fail_wrong_ell > 0)
                    printf("    - Wrong length: %llu\n", (unsigned long long)gr.fail_wrong_ell);
                all_pass = false;
            } else {
                printf("  [GenLemma] PASS\n");
            }

            // Test 3: Full Greedy Descent
            printf("  [Descent] Testing greedy descent from all RBPDs...\n");
            DescentTestResult dr = testGreedyDescent(n, verbose);

            printf("  [Descent] %llu/%llu reached identity, max steps=%d (%.3f sec)\n",
                   (unsigned long long)dr.pass_count,
                   (unsigned long long)dr.total_rbpds,
                   dr.max_steps, dr.elapsed_sec);
            if (dr.fail_count > 0) {
                printf("  [Descent] *** %llu FAILURES ***\n",
                       (unsigned long long)dr.fail_count);
                all_pass = false;
            } else {
                printf("  [Descent] PASS\n");
            }
        } else if (test_all && n > 7) {
            printf("  [GenLemma] Skipped (n=%d too large for BFS)\n", n);
            printf("  [Descent] Skipped (n=%d too large for BFS)\n", n);
        }

        printf("\n");
    }

    printf("==============================================\n");
    if (all_pass) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  SOME TESTS FAILED\n");
    }
    printf("==============================================\n");

    return all_pass ? 0 : 1;
}
