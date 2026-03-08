// cftp_universality_check.cpp
//
// THE CRITICAL EXPERIMENT: Does b0=b1 imply ALL chains coalesced?
//
// For each CFTP trial:
//   1. Generate the update sequence that makes extremal chains coalesce
//   2. Replay the SAME updates starting from EVERY reduced BPD
//   3. Check if ALL starting states reach the same final state
//
// If yes: CFTP is correct despite height-function violations.
// If no:  CFTP is BROKEN (produces biased samples).
//
// Compile:
//   clang++ -O3 -std=c++17 cftp_universality_check.cpp -o cftp_universality_check

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
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

    int numCrosses() const {
        int c = 0;
        for (auto t : grid) if (t == 1) c++;
        return c;
    }
};

// --- Moves ---

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
    printf("Enumerated %zu reduced BPDs for n=%d\n", allBPDs.size(), n);
}

// --- The Critical Experiment ---

struct Update { int8_t i, j; bool isUp; };

// Generate CFTP update sequence that coalesces the extremes.
// Returns the update sequence and the coalesced state.
bool generateCFTPSequence(int n, std::mt19937& rng,
                           std::vector<Update>& updates, BPD& result) {
    std::uniform_int_distribution<int> pos(1, n-1);
    std::uniform_int_distribution<int> dir(0, 1);

    int initial_window = std::max(16, 2 * n * n);
    updates.clear();
    updates.reserve(initial_window * 8);

    for (int t = 0; t < initial_window; t++)
        updates.push_back({(int8_t)pos(rng), (int8_t)pos(rng), dir(rng)==0});

    for (int round = 1; round <= 40; round++) {
        BPD b0(n), b1(n);
        b0.initIdentity();
        b1.initW0();

        for (const auto& u : updates) {
            b0 = applyUpdate(b0, u.i, u.j, u.isUp);
            b1 = applyUpdate(b1, u.i, u.j, u.isUp);
        }

        if (b0 == b1) {
            result = b0;
            return true;
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
    return false;
}

// Replay the update sequence on a given starting state
BPD replayUpdates(const BPD& start, const std::vector<Update>& updates) {
    BPD state = start;
    for (const auto& u : updates)
        state = applyUpdate(state, u.i, u.j, u.isUp);
    return state;
}

void runExperiment(int n, int trials) {
    printf("================================================================\n");
    printf("UNIVERSALITY TEST: n=%d, trials=%d\n", n, trials);
    printf("Does b0=b1 imply ALL %d-state chains coalesced?\n", n);
    printf("================================================================\n\n");

    // Step 1: Enumerate all reduced BPDs
    std::vector<BPD> allBPDs;
    enumerateAllReducedBPDs(n, allBPDs);

    int total_states = (int)allBPDs.size();
    int failures = 0;          // trials where some chain didn't coalesce
    int total_coalesced = 0;   // trials where extremes coalesced
    int64_t total_stragglers = 0; // total chains that disagreed with extremes

    std::mt19937 rng(std::random_device{}());
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int trial = 0; trial < trials; trial++) {
        // Generate CFTP sequence
        std::vector<Update> updates;
        BPD extremal_result;
        if (!generateCFTPSequence(n, rng, updates, extremal_result)) {
            printf("  Trial %d: extremes didn't coalesce (skipping)\n", trial);
            continue;
        }
        total_coalesced++;

        // Replay on ALL starting states
        int stragglers = 0;
        std::set<std::vector<uint8_t>> distinct_outcomes;
        distinct_outcomes.insert(extremal_result.grid);

        for (int s = 0; s < total_states; s++) {
            BPD final_state = replayUpdates(allBPDs[s], updates);
            if (!(final_state == extremal_result)) {
                stragglers++;
                distinct_outcomes.insert(final_state.grid);
            }
        }

        total_stragglers += stragglers;

        if (stragglers > 0) {
            failures++;
            printf("  Trial %d: FAILURE! %d/%d states did NOT coalesce "
                   "(%zu distinct outcomes, %zu updates)\n",
                   trial, stragglers, total_states,
                   distinct_outcomes.size(), updates.size());

            if (failures <= 3) {
                // Show details for first few failures
                printf("    Extremal result perm: ");
                auto ep = extremal_result.perm();
                for (int x : ep) printf("%d ", x);
                printf("(ell=%d)\n", extremal_result.numCrosses());

                for (int s = 0; s < total_states; s++) {
                    BPD fs = replayUpdates(allBPDs[s], updates);
                    if (!(fs == extremal_result)) {
                        auto sp = allBPDs[s].perm();
                        auto fp = fs.perm();
                        printf("    Start perm: ");
                        for (int x : sp) printf("%d ", x);
                        printf("(ell=%d) -> Final perm: ", allBPDs[s].numCrosses());
                        for (int x : fp) printf("%d ", x);
                        printf("(ell=%d)\n", fs.numCrosses());
                    }
                }
            }
        }

        // Progress
        if ((trial+1) % std::max(1, trials/10) == 0 || trial == trials-1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            printf("  [%d/%d] failures=%d, coalesced=%d (%.1fs)\n",
                   trial+1, trials, failures, total_coalesced, elapsed);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    printf("\n================================================================\n");
    printf("RESULT for n=%d:\n", n);
    printf("================================================================\n");
    printf("  Reduced BPDs:        %d\n", total_states);
    printf("  Trials:              %d\n", trials);
    printf("  Extremes coalesced:  %d\n", total_coalesced);
    printf("  Universal coalesce:  %d (ALL %d states agreed)\n",
           total_coalesced - failures, total_states);
    printf("  FAILURES:            %d (some state disagreed)\n", failures);
    printf("  Total stragglers:    %lld\n", (long long)total_stragglers);
    printf("  Time:                %.2fs\n", elapsed);

    if (failures == 0) {
        printf("\n  *** ALL %d states coalesced in every trial ***\n", total_states);
        printf("  *** CFTP produces correct samples despite h-function violations ***\n\n");
    } else {
        printf("\n  !!! CFTP IS BROKEN: %d trials had non-universal coalescence !!!\n",
               failures);
        printf("  !!! The sampler produces BIASED samples !!!\n\n");
    }
}

int main(int argc, char* argv[]) {
    int n = (argc >= 2) ? atoi(argv[1]) : 4;
    int trials = (argc >= 3) ? atoi(argv[2]) : 1000;

    if (n < 3 || n > 7) {
        printf("Usage: %s <n> [trials]  (3 <= n <= 7)\n", argv[0]);
        return 1;
    }

    runExperiment(n, trials);
    return 0;
}
