// cftp_bias_test.cpp
//
// Test whether naive CFTP (internal rejection, extremal chains only)
// produces the correct distribution on permutations.
//
// The correct distribution on RBPDs is uniform, which induces a distribution
// on permutations proportional to S_w(1^n) (= number of RBPDs for w).
//
// We enumerate all RBPDs to get the true S_w(1^n) values, then run many
// naive CFTP trials and chi-squared test the permutation frequencies.
//
// Compile:
//   clang++ -O3 -std=c++17 cftp_bias_test.cpp -o cftp_bias_test

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <set>
#include <map>
#include <queue>
#include <cassert>
#include <chrono>

// Tile types: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz

struct BPD {
    int n;
    std::vector<uint8_t> grid;

    BPD() : n(0) {}
    BPD(int sz) : n(sz), grid(sz * sz, 0) {}

    uint8_t get(int r, int c) const { return grid[r * n + c]; }
    void set(int r, int c, uint8_t v) { grid[r * n + c] = v; }

    bool operator==(const BPD& o) const { return n == o.n && grid == o.grid; }
    bool operator<(const BPD& o) const { return grid < o.grid; }

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
};

// --- Moves with internal per-chain rejection ---

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

BPD applyUpdate(const BPD& state, int i, int j, bool isUp) {
    BPD proposed = state;
    bool moved = isUp ? tryUp(proposed, i, j) : tryDown(proposed, i, j);
    if (!moved) return state;

    bool has_cross = (state.get(i-1,j-1)==1 || proposed.get(i-1,j-1)==1 ||
                      state.get(i,j)==1 || proposed.get(i,j)==1);
    if (has_cross) {
        if (proposed.isReduced()) return proposed;
        return state;
    }
    return proposed;
}

// --- Enumerate all reduced BPDs via BFS ---

void enumerateAllReducedBPDs(int n, std::vector<BPD>& allBPDs) {
    std::set<std::vector<uint8_t>> visited;
    std::queue<BPD> frontier;

    BPD id(n); id.initIdentity();
    frontier.push(id);
    visited.insert(id.grid);

    BPD w0(n); w0.initW0();
    if (visited.find(w0.grid) == visited.end()) {
        frontier.push(w0);
        visited.insert(w0.grid);
    }

    while (!frontier.empty()) {
        BPD cur = frontier.front();
        frontier.pop();
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
    for (auto& g : visited) {
        BPD b(n);
        b.grid = g;
        allBPDs.push_back(b);
    }
}

// --- CFTP ---

struct Update { int8_t i, j; bool isUp; };

BPD replayUpdates(const BPD& start, const std::vector<Update>& updates) {
    BPD state = start;
    for (const auto& u : updates)
        state = applyUpdate(state, u.i, u.j, u.isUp);
    return state;
}

// Format permutation as string
std::string permStr(const std::vector<int>& p) {
    std::string s = "(";
    for (size_t i = 0; i < p.size(); i++) {
        if (i > 0) s += ",";
        s += std::to_string(p[i]);
    }
    s += ")";
    return s;
}

int main(int argc, char* argv[]) {
    int n = (argc >= 2) ? atoi(argv[1]) : 4;
    int trials = (argc >= 3) ? atoi(argv[2]) : 200000;

    if (n < 3 || n > 6) {
        printf("Usage: %s <n> [trials]  (3 <= n <= 6)\n", argv[0]);
        return 1;
    }

    printf("=== CFTP BIAS TEST (permutation level): n=%d, %d trials ===\n\n", n, trials);

    // Step 1: Enumerate all RBPDs and compute S_w(1^n) for each permutation
    std::vector<BPD> allBPDs;
    enumerateAllReducedBPDs(n, allBPDs);
    int totalBPDs = (int)allBPDs.size();
    printf("Total RBPDs: %d\n", totalBPDs);

    // Count RBPDs per permutation = S_w(1^n)
    std::map<std::vector<int>, int> S_w;  // true Schubert values
    for (auto& b : allBPDs)
        S_w[b.perm()]++;

    int numPerms = (int)S_w.size();
    printf("Permutations with S_w > 0: %d\n", numPerms);

    // Print true distribution
    printf("\nTrue S_w(1^%d) values:\n", n);
    std::vector<std::pair<std::vector<int>, int>> sw_list(S_w.begin(), S_w.end());
    std::sort(sw_list.begin(), sw_list.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (auto& [w, val] : sw_list)
        printf("  %-12s  S_w = %d  (%.4f)\n", permStr(w).c_str(), val, (double)val / totalBPDs);

    // Step 2: Run naive CFTP (internal rejection, extremal chains only)
    std::map<std::vector<int>, int> naive_perm_counts;
    int naive_total = 0;
    int false_coalescences = 0;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> pos(1, n-1);
    std::uniform_int_distribution<int> dir(0, 1);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int trial = 0; trial < trials; trial++) {
        int window = std::max(16, 2 * n * n);
        std::vector<Update> updates;
        updates.reserve(window * 8);
        for (int t = 0; t < window; t++)
            updates.push_back({(int8_t)pos(rng), (int8_t)pos(rng), dir(rng)==0});

        bool coalesced = false;
        for (int round = 0; round < 40; round++) {
            BPD b_id(n); b_id.initIdentity();
            BPD b_w0(n); b_w0.initW0();
            b_id = replayUpdates(b_id, updates);
            b_w0 = replayUpdates(b_w0, updates);

            if (b_id == b_w0) {
                coalesced = true;
                auto w = b_id.perm();
                naive_perm_counts[w]++;
                naive_total++;

                // Also check universality for false coalescence rate
                bool universal = true;
                for (int s = 0; s < totalBPDs; s++) {
                    BPD fs = replayUpdates(allBPDs[s], updates);
                    if (!(fs == b_id)) { universal = false; break; }
                }
                if (!universal) false_coalescences++;

                break;
            }

            // Double
            int extend = (int)updates.size();
            std::vector<Update> earlier;
            earlier.reserve(extend + updates.size());
            for (int t = 0; t < extend; t++)
                earlier.push_back({(int8_t)pos(rng), (int8_t)pos(rng), dir(rng)==0});
            earlier.insert(earlier.end(), updates.begin(), updates.end());
            updates = std::move(earlier);
        }

        if ((trial+1) % std::max(1, trials/20) == 0 || trial == trials-1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            printf("\r[%d/%d] samples=%d, false_coal=%d (%.1fs)   ",
                   trial+1, trials, naive_total, false_coalescences, elapsed);
            fflush(stdout);
        }
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    printf("\n\n=== RESULTS ===\n");
    printf("n = %d, %d RBPDs, %d permutations\n", n, totalBPDs, numPerms);
    printf("Naive CFTP samples: %d\n", naive_total);
    printf("False coalescences: %d (%.1f%%)\n",
           false_coalescences, 100.0 * false_coalescences / naive_total);
    printf("Time: %.1fs\n\n", elapsed);

    // Step 3: Chi-squared test of naive permutation frequencies
    // against the true distribution proportional to S_w(1^n)
    printf("=== Chi-squared: naive CFTP vs S_w(1^n) distribution ===\n\n");
    printf("%-12s  %6s  %10s  %10s  %10s  %10s\n",
           "perm", "S_w", "expected", "observed", "obs/exp", "(O-E)^2/E");

    double chi2 = 0;
    int df = numPerms - 1;

    for (auto& [w, sw_val] : sw_list) {
        double expected = (double)naive_total * sw_val / totalBPDs;
        int observed = naive_perm_counts.count(w) ? naive_perm_counts[w] : 0;
        double contrib = (observed - expected) * (observed - expected) / expected;
        chi2 += contrib;
        printf("%-12s  %6d  %10.1f  %10d  %10.4f  %10.2f\n",
               permStr(w).c_str(), sw_val, expected, observed,
               observed / expected, contrib);
    }

    double z = (chi2 - df) / std::sqrt(2.0 * df);
    printf("\nChi-squared = %.2f, df = %d, z-score = %.2f\n", chi2, df, z);
    if (std::abs(z) < 3.0)
        printf(">>> CONSISTENT with S_w distribution (|z| = %.2f < 3) <<<\n", std::abs(z));
    else
        printf(">>> REJECTED: not proportional to S_w (|z| = %.2f) <<<\n", std::abs(z));

    // Total variation distance
    double tv = 0;
    for (auto& [w, sw_val] : sw_list) {
        double p_true = (double)sw_val / totalBPDs;
        double p_naive = (double)(naive_perm_counts.count(w) ? naive_perm_counts[w] : 0) / naive_total;
        tv += std::abs(p_true - p_naive);
    }
    tv /= 2.0;
    printf("TV distance (naive vs true): %.6f\n", tv);

    printf("\n");
    return 0;
}
