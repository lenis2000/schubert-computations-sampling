// Explore Allowed 2x2 Moves on Reduced BPDs
//
// For every reduced BPD with w != e, enumerate ALL available 2x2 down-moves,
// check which are allowed (preserve reducedness), and look for patterns.
//
// Key questions:
//   - Is there ALWAYS at least one allowed down-move from any non-identity RBPD?
//   - If so, what characterizes the allowed ones?
//   - What cross positions admit allowed box-cross annihilation?
//   - Can we find a deterministic strategy that always works?
//
// Compile:
//   clang++ -O3 -std=c++17 -mcpu=apple-m2 -flto -DNDEBUG \
//     explore_allowed_moves.cpp -o explore_allowed_moves

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <string>
#include <map>
#include <set>

// Tile types: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz

class BPD {
public:
    int n;
    std::vector<uint8_t> grid;

    BPD(int size) : n(size), grid(size * size, 0) {}
    BPD(int size, const std::vector<uint8_t>& g) : n(size), grid(g) {}

    inline uint8_t get(int r, int c) const { return grid[r * n + c]; }
    inline void set(int r, int c, uint8_t val) { grid[r * n + c] = val; }

    bool operator==(const BPD& other) const { return grid == other.grid; }

    void generateRotheBPD(const std::vector<int>& w) {
        int minW = *std::min_element(w.begin(), w.end());
        for (int j = 1; j <= n; j++) {
            if (j + minW - 1 < w[0]) set(0, j - 1, 0);
            else if (j + minW - 1 == w[0]) set(0, j - 1, 2);
            else set(0, j - 1, 5);
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

    bool isReduced() const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;
        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1, j = n + s - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1) {
                    if (ww[k - 1] > ww[k]) return false;
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1, j = n - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1) {
                    if (ww[k - 1] > ww[k]) return false;
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        return true;
    }

    std::vector<int> computePerm() const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;
        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1, j = n + s - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1 && k < n) std::swap(ww[k - 1], ww[k]);
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1, j = n - t - 1;
                int k = get_sk_from_cross(i, j);
                if (k != -1 && k < n) std::swap(ww[k - 1], ww[k]);
            }
        }
        return ww;
    }

    // Find the first cross that causes non-reducedness, reporting details
    struct DoubleCrossingInfo {
        bool found;
        int row, col;           // position of the bad cross
        int pipe_a, pipe_b;     // the two pipe labels that cross twice
        int first_cross_row, first_cross_col; // where they first crossed
    };

    DoubleCrossingInfo findDoubleCrossing() const {
        DoubleCrossingInfo info{false, -1, -1, -1, -1, -1, -1};
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;

        // Track where each pair first crosses
        // pair (a,b) with a<b: first cross position
        std::map<std::pair<int,int>, std::pair<int,int>> first_cross;

        auto process_cross = [&](int ci, int cj) -> bool {
            int k = get_sk_from_cross(ci, cj);
            if (k == -1 || k >= n) return true;
            int a = ww[k-1], b = ww[k];
            if (a > b) {
                // This is the double crossing!
                info.found = true;
                info.row = ci; info.col = cj;
                info.pipe_a = std::min(a,b);
                info.pipe_b = std::max(a,b);
                auto key = std::make_pair(info.pipe_a, info.pipe_b);
                if (first_cross.count(key))  {
                    info.first_cross_row = first_cross[key].first;
                    info.first_cross_col = first_cross[key].second;
                }
                return false;
            }
            auto key = std::make_pair(std::min(a,b), std::max(a,b));
            if (!first_cross.count(key)) {
                first_cross[key] = {ci, cj};
            }
            std::swap(ww[k-1], ww[k]);
            return true;
        };

        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int ci = n - t - 1, cj = n + s - t - 1;
                if (!process_cross(ci, cj)) return info;
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int ci = n - s - t - 1, cj = n - t - 1;
                if (!process_cross(ci, cj)) return info;
            }
        }
        return info;
    }

    // Trace all pipes through the BPD.
    // Returns pipe_at[i][j] = pair of pipe labels (south-pipe, west-pipe) at each cell.
    // Pipe k enters from the bottom of column (k-1) and exits on the right of some row.
    void tracePipes(std::vector<std::vector<std::pair<int,int>>>& pipe_at) const {
        // pipe_at[i][j] = (pipe going vertically/south-to-north, pipe going horizontally/west-to-east)
        // Convention: pipe k enters from bottom at column k-1 (0-indexed)
        // At each cell, one or two pipes pass through
        pipe_at.assign(n, std::vector<std::pair<int,int>>(n, {0, 0}));

        // Track which pipe is on each edge
        // vert_pipe[i][j] = pipe label on the edge between row i-1 and row i at column j
        // horiz_pipe[i][j] = pipe label on the edge between col j-1 and col j at row i
        std::vector<std::vector<int>> vert_pipe(n+1, std::vector<int>(n, 0));
        std::vector<std::vector<int>> horiz_pipe(n, std::vector<int>(n+1, 0));

        // Bottom boundary: pipe k enters at column k-1
        for (int j = 0; j < n; j++) vert_pipe[n][j] = j + 1;
        // Left boundary: no pipes enter from the left
        for (int i = 0; i < n; i++) horiz_pipe[i][0] = 0;

        // Process from bottom-left to top-right
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j < n; j++) {
                int from_south = vert_pipe[i+1][j];
                int from_west = horiz_pipe[i][j];
                int tile = get(i, j);

                switch (tile) {
                    case 0: // box: no pipes
                        vert_pipe[i][j] = 0;
                        horiz_pipe[i][j+1] = 0;
                        break;
                    case 1: // cross: pipes cross
                        pipe_at[i][j] = {from_south, from_west};
                        vert_pipe[i][j] = from_west;
                        horiz_pipe[i][j+1] = from_south;
                        break;
                    case 2: // r-elbow: south→east
                        pipe_at[i][j] = {from_south, 0};
                        vert_pipe[i][j] = 0;
                        horiz_pipe[i][j+1] = from_south;
                        break;
                    case 3: // j-elbow: west→north... wait
                        // j-elbow: pipe enters from west, goes north
                        // Actually in BPD: j-elbow connects south and east? No...
                        // In BPD tile semantics for bumpless pipe dreams:
                        // j-elbow (╮): connects top and left (pipe from west turns north)
                        // Wait, I need to get the tile semantics right.
                        // Tile 3 = j-elbow = ╮: connects W→N (pipe from west exits north)
                        pipe_at[i][j] = {0, from_west};
                        vert_pipe[i][j] = from_west;
                        horiz_pipe[i][j+1] = 0;
                        break;
                    case 4: // vert: south→north
                        pipe_at[i][j] = {from_south, 0};
                        vert_pipe[i][j] = from_south;
                        horiz_pipe[i][j+1] = 0;
                        break;
                    case 5: // horiz: west→east
                        pipe_at[i][j] = {0, from_west};
                        vert_pipe[i][j] = 0;
                        horiz_pipe[i][j+1] = from_west;
                        break;
                }
            }
        }
    }

    // Print grid with pipe labels at crosses
    void printWithPipes() const {
        std::vector<std::vector<std::pair<int,int>>> pipe_at;
        tracePipes(pipe_at);
        const char* names[] = {".", "+", "r", "j", "|", "-"};
        for (int i = 0; i < n; i++) {
            printf("    ");
            for (int j = 0; j < n; j++) {
                int tile = get(i, j);
                if (tile == 1) { // cross: show pipe labels
                    printf("%d%d", pipe_at[i][j].first, pipe_at[i][j].second);
                } else {
                    printf("%s ", names[tile]);
                }
            }
            printf("\n");
        }
    }

    int countCrosses() const {
        int c = 0;
        for (auto t : grid) if (t == 1) c++;
        return c;
    }

    bool isRothe() const {
        // A Rothe BPD has no j-elbows
        for (auto t : grid) if (t == 3) return false;
        return true;
    }

    void print() const {
        const char* names[] = {".", "+", "r", "j", "|", "-"};
        for (int i = 0; i < n; i++) {
            printf("    ");
            for (int j = 0; j < n; j++) printf("%s ", names[get(i, j)]);
            printf("\n");
        }
    }
};

int coxeterLength(const std::vector<int>& w) {
    int n = w.size(), inv = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (w[i] > w[j]) inv++;
    return inv;
}

std::string permToString(const std::vector<int>& w) {
    std::string s = "(";
    for (size_t i = 0; i < w.size(); i++) {
        s += std::to_string(w[i]);
        if (i + 1 < w.size()) s += ",";
    }
    return s + ")";
}

// =============================================================================
// Move operations - return true if move was applied
// =============================================================================

bool tryUndrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i-1,j-1), ne = bpd.get(i-1,j);
    uint8_t sw = bpd.get(i,j-1), se = bpd.get(i,j);
    if (nw == 0 && se == 3 && (ne == 2 || ne == 4) && (sw == 2 || sw == 5)) {
        bpd.set(i-1,j-1, 2); bpd.set(i,j, 0);
        bpd.set(i-1,j, (ne == 2) ? 5 : 3);
        bpd.set(i,j-1, (sw == 2) ? 4 : 3);
        return true;
    }
    return false;
}

bool tryCrossUndrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i-1,j-1), ne = bpd.get(i-1,j);
    uint8_t sw = bpd.get(i,j-1), se = bpd.get(i,j);
    if (nw == 0 && se == 1) {
        bpd.set(i-1,j-1, 2); bpd.set(i,j, 2);
        bpd.set(i-1,j, (ne == 4) ? 3 : 5);
        bpd.set(i,j-1, (sw == 5) ? 3 : 4);
        return true;
    }
    if (nw == 3 && se == 3) {
        bpd.set(i-1,j-1, 1); bpd.set(i,j, 0);
        bpd.set(i-1,j, (ne == 4) ? 3 : 5);
        bpd.set(i,j-1, (sw == 5) ? 3 : 4);
        return true;
    }
    if (nw == 3 && se == 1) {
        bpd.set(i-1,j-1, 1); bpd.set(i,j, 2);
        bpd.set(i-1,j, (ne == 4) ? 3 : 5);
        bpd.set(i,j-1, (sw == 5) ? 3 : 4);
        return true;
    }
    return false;
}

bool tryDrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i-1,j-1), se = bpd.get(i,j);
    if (nw == 2 && se == 0) {
        uint8_t ne = bpd.get(i-1,j), sw = bpd.get(i,j-1);
        bpd.set(i-1,j-1, 0); bpd.set(i,j, 3);
        bpd.set(i-1,j, (ne == 5) ? 2 : 4);
        bpd.set(i,j-1, (sw == 4) ? 2 : 5);
        return true;
    }
    return false;
}

bool tryCrossDrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i-1,j-1), ne = bpd.get(i-1,j);
    uint8_t sw = bpd.get(i,j-1), se = bpd.get(i,j);
    if (nw == 2 && se == 2) {
        bpd.set(i-1,j-1, 0); bpd.set(i,j, 1);
        bpd.set(i-1,j, (ne == 3) ? 4 : 2);
        bpd.set(i,j-1, (sw == 3) ? 5 : 2);
        return true;
    }
    if (nw == 1 && se == 0) {
        bpd.set(i-1,j-1, 3); bpd.set(i,j, 3);
        bpd.set(i-1,j, (ne == 3) ? 4 : 2);
        bpd.set(i,j-1, (sw == 3) ? 5 : 2);
        return true;
    }
    if (nw == 1 && se == 2) {
        bpd.set(i-1,j-1, 3); bpd.set(i,j, 1);
        bpd.set(i-1,j, (ne == 3) ? 4 : 2);
        bpd.set(i,j-1, (sw == 3) ? 5 : 2);
        return true;
    }
    return false;
}

// =============================================================================
// Enumerate all reduced BPDs via BFS
// =============================================================================

struct GridHash {
    size_t operator()(const std::vector<uint8_t>& g) const {
        size_t h = 0;
        for (uint8_t v : g) h = h * 31 + v;
        return h;
    }
};

void getNeighbors(const BPD& bpd, std::vector<std::vector<uint8_t>>& neighbors) {
    neighbors.clear();
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            BPD temp = bpd;
            if (tryDrip(temp, i, j) && temp.isReduced()) { neighbors.push_back(temp.grid); temp = bpd; }
            if (tryUndrip(temp, i, j) && temp.isReduced()) { neighbors.push_back(temp.grid); temp = bpd; }
            if (tryCrossDrip(temp, i, j) && temp.isReduced()) { neighbors.push_back(temp.grid); temp = bpd; }
            if (tryCrossUndrip(temp, i, j) && temp.isReduced()) { neighbors.push_back(temp.grid); temp = bpd; }
        }
    }
}

std::vector<std::vector<uint8_t>> enumerateAllRBPDs(int n) {
    std::unordered_set<std::vector<uint8_t>, GridHash> visited;
    std::queue<std::vector<uint8_t>> q;

    BPD bpd_id(n);
    std::vector<int> id(n);
    std::iota(id.begin(), id.end(), 1);
    bpd_id.generateRotheBPD(id);
    visited.insert(bpd_id.grid);
    q.push(bpd_id.grid);

    BPD bpd_w0(n);
    std::vector<int> w0(n);
    for (int i = 0; i < n; i++) w0[i] = n - i;
    bpd_w0.generateRotheBPD(w0);
    if (visited.find(bpd_w0.grid) == visited.end()) {
        visited.insert(bpd_w0.grid);
        q.push(bpd_w0.grid);
    }

    std::vector<std::vector<uint8_t>> neighbors;
    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        BPD bpd(n, cur);
        getNeighbors(bpd, neighbors);
        for (const auto& nb : neighbors) {
            if (visited.find(nb) == visited.end()) {
                visited.insert(nb);
                q.push(nb);
            }
        }
    }

    return std::vector<std::vector<uint8_t>>(visited.begin(), visited.end());
}

// =============================================================================
// Move descriptor
// =============================================================================

struct MoveInfo {
    int i, j;              // position (bottom-right of 2x2 window)
    int type;              // 0=undrip, 1=cross-undrip
    int subcase;           // for cross-undrip: 1=blank+cross, 2=jelbow+jelbow, 3=jelbow+cross
    bool is_reduced;       // does the result preserve reducedness?
    int delta_crosses;     // change in cross count
    int delta_ell;         // change in Coxeter length
    int cross_row, cross_col;  // position of the cross involved (SE for case 1,3; NW for case 2)
};

// Find all available down-moves and classify them
std::vector<MoveInfo> findAllDownMoves(const BPD& bpd) {
    std::vector<MoveInfo> moves;
    int n = bpd.n;
    int orig_crosses = bpd.countCrosses();
    std::vector<int> orig_perm = bpd.computePerm();
    int orig_ell = coxeterLength(orig_perm);

    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            // Try undrip
            {
                BPD temp(bpd);
                if (tryUndrip(temp, i, j)) {
                    MoveInfo m;
                    m.i = i; m.j = j;
                    m.type = 0; m.subcase = 0;
                    m.is_reduced = temp.isReduced();
                    m.delta_crosses = temp.countCrosses() - orig_crosses;
                    m.delta_ell = coxeterLength(temp.computePerm()) - orig_ell;
                    m.cross_row = -1; m.cross_col = -1;
                    moves.push_back(m);
                }
            }

            // Try cross-undrip (3 subcases)
            {
                uint8_t nw = bpd.get(i-1,j-1);
                uint8_t se = bpd.get(i,j);

                int subcase = 0;
                int cr = -1, cc = -1;
                if (nw == 0 && se == 1) { subcase = 1; cr = i; cc = j; }       // box-cross annihilation
                else if (nw == 3 && se == 3) { subcase = 2; cr = i-1; cc = j-1; } // jelbow+jelbow -> creates cross at NW
                else if (nw == 3 && se == 1) { subcase = 3; cr = i; cc = j; }   // jelbow+cross -> moves cross to NW

                if (subcase > 0) {
                    BPD temp(bpd);
                    if (tryCrossUndrip(temp, i, j)) {
                        MoveInfo m;
                        m.i = i; m.j = j;
                        m.type = 1; m.subcase = subcase;
                        m.is_reduced = temp.isReduced();
                        m.delta_crosses = temp.countCrosses() - orig_crosses;
                        m.delta_ell = coxeterLength(temp.computePerm()) - orig_ell;
                        m.cross_row = cr; m.cross_col = cc;
                        moves.push_back(m);
                    }
                }
            }
        }
    }
    return moves;
}

// =============================================================================
// Analysis
// =============================================================================

void analyze(int n, bool verbose) {
    printf("=== n = %d ===\n", n);

    auto all_rbpds = enumerateAllRBPDs(n);
    printf("Total RBPDs: %zu\n", all_rbpds.size());

    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    int total_nonid = 0;
    int has_allowed_annihilation = 0;  // has at least one allowed box-cross annihilation
    int has_allowed_crossundrip = 0;   // has at least one allowed cross-undrip (any subcase)
    int has_any_allowed_down = 0;      // has at least one allowed down-move (any type)
    int no_allowed_down = 0;           // NO allowed down-moves at all
    int no_allowed_cross_reducing = 0; // no allowed cross-reducing move

    // Track failure patterns for box-cross annihilation
    int annihilation_available = 0;
    int annihilation_allowed = 0;
    int annihilation_blocked = 0;

    // For BPDs with no allowed annihilation, track what IS available
    struct StuckCase {
        std::vector<uint8_t> grid;
        std::vector<int> perm;
        int ell;
        bool is_rothe;
        std::vector<MoveInfo> all_moves;
    };
    std::vector<StuckCase> stuck_cases;

    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        std::vector<int> perm = bpd.computePerm();
        if (perm == identity) continue;
        total_nonid++;

        auto moves = findAllDownMoves(bpd);

        bool found_allowed_annihilation = false;
        bool found_allowed_crossundrip = false;
        bool found_any_allowed_down = false;
        bool found_cross_reducing = false;

        for (const auto& m : moves) {
            if (m.type == 0 && m.is_reduced) {
                // Undrip - always allowed, doesn't reduce crosses
                found_any_allowed_down = true;
            }
            if (m.type == 1) {
                if (m.subcase == 1) {
                    annihilation_available++;
                    if (m.is_reduced) {
                        annihilation_allowed++;
                        found_allowed_annihilation = true;
                        found_any_allowed_down = true;
                        found_cross_reducing = true;
                    } else {
                        annihilation_blocked++;
                    }
                }
                if (m.is_reduced) {
                    found_allowed_crossundrip = true;
                    found_any_allowed_down = true;
                    if (m.delta_crosses < 0) found_cross_reducing = true;
                }
            }
        }

        if (found_allowed_annihilation) has_allowed_annihilation++;
        if (found_allowed_crossundrip) has_allowed_crossundrip++;
        if (found_any_allowed_down) has_any_allowed_down++;
        if (!found_any_allowed_down) no_allowed_down++;
        if (!found_cross_reducing) no_allowed_cross_reducing++;

        // Collect stuck cases (no allowed box-cross annihilation)
        if (!found_allowed_annihilation && verbose) {
            StuckCase sc;
            sc.grid = grid;
            sc.perm = perm;
            sc.ell = coxeterLength(perm);
            sc.is_rothe = bpd.isRothe();
            sc.all_moves = moves;
            stuck_cases.push_back(sc);
        }
    }

    printf("\nOf %d non-identity RBPDs:\n", total_nonid);
    printf("  Has allowed box-cross annihilation:  %d (%.1f%%)\n",
           has_allowed_annihilation, 100.0 * has_allowed_annihilation / total_nonid);
    printf("  Has allowed cross-undrip (any):      %d (%.1f%%)\n",
           has_allowed_crossundrip, 100.0 * has_allowed_crossundrip / total_nonid);
    printf("  Has ANY allowed down-move:            %d (%.1f%%)\n",
           has_any_allowed_down, 100.0 * has_any_allowed_down / total_nonid);
    printf("  No allowed down-moves at all:         %d\n", no_allowed_down);
    printf("  No cross-reducing move:               %d (%.1f%%)\n",
           no_allowed_cross_reducing, 100.0 * no_allowed_cross_reducing / total_nonid);

    printf("\nBox-cross annihilation stats:\n");
    printf("  Available: %d, Allowed: %d, Blocked: %d (%.1f%% blocked)\n",
           annihilation_available, annihilation_allowed, annihilation_blocked,
           annihilation_available > 0 ? 100.0 * annihilation_blocked / annihilation_available : 0);

    if (verbose && !stuck_cases.empty()) {
        // Show a few stuck cases
        int show = std::min((int)stuck_cases.size(), 5);
        printf("\n--- First %d BPDs with NO allowed box-cross annihilation ---\n", show);
        for (int idx = 0; idx < show; idx++) {
            auto& sc = stuck_cases[idx];
            BPD bpd(n, sc.grid);
            printf("\n  Perm=%s, ell=%d, Rothe=%s\n",
                   permToString(sc.perm).c_str(), sc.ell,
                   sc.is_rothe ? "yes" : "no");
            bpd.print();

            printf("  Available moves:\n");
            for (const auto& m : sc.all_moves) {
                const char* type_str;
                if (m.type == 0) type_str = "undrip";
                else if (m.subcase == 1) type_str = "box-cross-annihil";
                else if (m.subcase == 2) type_str = "cross-undrip(jj)";
                else type_str = "cross-undrip(j+)";

                printf("    (%d,%d) %-20s %s  dCross=%+d dEll=%+d\n",
                       m.i, m.j, type_str,
                       m.is_reduced ? "ALLOWED" : "blocked",
                       m.delta_crosses, m.delta_ell);
            }
        }
    }

    // === New analysis: for BPDs where annihilation exists but some are blocked,
    // which cross positions work? ===
    printf("\n--- Cross position analysis for box-cross annihilation ---\n");

    // For each RBPD with at least one blocked annihilation, check if there's
    // a pattern to which crosses work
    struct CrossAnalysis {
        int total_bpds_with_multiple_annihilations;
        int some_blocked_some_allowed;
        int all_allowed;
        int all_blocked;
    } ca = {};

    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        std::vector<int> perm = bpd.computePerm();
        if (perm == identity) continue;

        // Find all box-cross annihilation positions
        std::vector<std::pair<int,int>> allowed_pos, blocked_pos;
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                if (bpd.get(i-1,j-1) == 0 && bpd.get(i,j) == 1) {
                    BPD temp(bpd);
                    if (tryCrossUndrip(temp, i, j)) {
                        if (temp.isReduced()) allowed_pos.push_back({i, j});
                        else blocked_pos.push_back({i, j});
                    }
                }
            }
        }

        int total = allowed_pos.size() + blocked_pos.size();
        if (total >= 1) {
            ca.total_bpds_with_multiple_annihilations++;
            if (blocked_pos.empty()) ca.all_allowed++;
            else if (allowed_pos.empty()) ca.all_blocked++;
            else ca.some_blocked_some_allowed++;
        }
    }

    printf("  BPDs with at least 1 annihilation available: %d\n",
           ca.total_bpds_with_multiple_annihilations);
    printf("    All allowed: %d\n", ca.all_allowed);
    printf("    Some blocked, some allowed: %d\n", ca.some_blocked_some_allowed);
    printf("    All blocked: %d\n", ca.all_blocked);

    // === Key question: does EVERY non-identity RBPD have at least one
    //     allowed cross-REDUCING move (annihilation or cross-undrip that kills a cross)?
    //     Or at minimum, some allowed down-move? ===
    printf("\n*** KEY FINDING: ");
    if (no_allowed_down == 0) {
        printf("Every non-identity RBPD has at least one allowed down-move ***\n");
    } else {
        printf("There exist %d RBPDs with NO allowed down-move! ***\n", no_allowed_down);
    }

    printf("\n");
}

// =============================================================================
// Detailed analysis: for BPDs where annihilation is blocked, what property
// distinguishes blocked vs allowed crosses?
// =============================================================================

void analyzeBlockedAnnihilations(int n) {
    printf("=== Detailed blocked annihilation analysis for n=%d ===\n\n", n);

    auto all_rbpds = enumerateAllRBPDs(n);
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    // For each blocked annihilation, record the cross position and surrounding context
    int count = 0;
    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        std::vector<int> perm = bpd.computePerm();
        if (perm == identity) continue;

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                if (bpd.get(i-1,j-1) == 0 && bpd.get(i,j) == 1) {
                    BPD temp(bpd);
                    if (tryCrossUndrip(temp, i, j) && !temp.isReduced()) {
                        count++;
                        if (count <= 8) {
                            printf("Blocked annihilation #%d:\n", count);
                            printf("  Perm=%s, cross at (%d,%d)\n",
                                   permToString(perm).c_str(), i, j);

                            // Check: is there a cross ABOVE row i?
                            bool cross_above = false;
                            for (int r = 0; r < i; r++)
                                for (int c = 0; c < n; c++)
                                    if (bpd.get(r, c) == 1) cross_above = true;

                            // Check: is there a cross LEFT of col j in row i?
                            bool cross_left = false;
                            for (int c = 0; c < j; c++)
                                if (bpd.get(i, c) == 1) cross_left = true;

                            printf("  Cross above row %d: %s\n", i, cross_above ? "YES" : "no");
                            printf("  Cross left of col %d in row %d: %s\n", j, i, cross_left ? "YES" : "no");
                            printf("  Is topmost-leftmost: %s\n",
                                   (!cross_above && !cross_left) ? "YES" : "no");
                            printf("  Rothe: %s\n", bpd.isRothe() ? "yes" : "no");
                            bpd.print();
                            printf("\n");
                        }
                    }
                }
            }
        }
    }
    printf("Total blocked annihilations: %d\n\n", count);
}

// =============================================================================
// Test: does the BOTTOMMOST-RIGHTMOST cross always work for annihilation?
// Or some other selection strategy?
// =============================================================================

void testCrossSelectionStrategies(int n) {
    printf("=== Cross selection strategies for n=%d ===\n\n", n);

    auto all_rbpds = enumerateAllRBPDs(n);
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    // Strategies to test:
    // 1. Topmost-leftmost (already known to fail for non-Rothe)
    // 2. Bottommost-rightmost
    // 3. Any allowed (just check existence)
    // 4. Bottommost-leftmost
    // 5. Topmost-rightmost

    struct Strategy {
        const char* name;
        int pass, fail;
    };
    Strategy strategies[] = {
        {"topmost-leftmost", 0, 0},
        {"bottommost-rightmost", 0, 0},
        {"bottommost-leftmost", 0, 0},
        {"topmost-rightmost", 0, 0},
        {"any-allowed-exists", 0, 0},
    };

    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        std::vector<int> perm = bpd.computePerm();
        if (perm == identity) continue;

        // Collect all crosses
        std::vector<std::pair<int,int>> crosses;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (bpd.get(i, j) == 1) crosses.push_back({i, j});

        if (crosses.empty()) continue;

        // For each cross, check if annihilation (NW=blank) is available and allowed
        auto checkAnnihilation = [&](int ci, int cj) -> bool {
            if (ci < 1 || cj < 1) return false;
            if (bpd.get(ci-1, cj-1) != 0) return false;
            BPD temp(bpd);
            if (!tryCrossUndrip(temp, ci, cj)) return false;
            return temp.isReduced();
        };

        // Strategy 0: topmost-leftmost
        {
            auto [ci, cj] = crosses.front(); // already sorted by row then col
            if (checkAnnihilation(ci, cj)) strategies[0].pass++;
            else strategies[0].fail++;
        }

        // Strategy 1: bottommost-rightmost
        {
            auto [ci, cj] = crosses.back();
            if (checkAnnihilation(ci, cj)) strategies[1].pass++;
            else strategies[1].fail++;
        }

        // Strategy 2: bottommost-leftmost
        {
            int best_r = -1, best_c = n;
            for (auto [r, c] : crosses) {
                if (r > best_r || (r == best_r && c < best_c)) {
                    best_r = r; best_c = c;
                }
            }
            if (checkAnnihilation(best_r, best_c)) strategies[2].pass++;
            else strategies[2].fail++;
        }

        // Strategy 3: topmost-rightmost
        {
            int best_r = n, best_c = -1;
            for (auto [r, c] : crosses) {
                if (r < best_r || (r == best_r && c > best_c)) {
                    best_r = r; best_c = c;
                }
            }
            if (checkAnnihilation(best_r, best_c)) strategies[3].pass++;
            else strategies[3].fail++;
        }

        // Strategy 4: any allowed exists
        {
            bool found = false;
            for (auto [ci, cj] : crosses) {
                if (checkAnnihilation(ci, cj)) { found = true; break; }
            }
            if (found) strategies[4].pass++;
            else strategies[4].fail++;
        }
    }

    int total = strategies[0].pass + strategies[0].fail;
    printf("Strategy results (box-cross annihilation only, %d non-identity RBPDs):\n", total);
    for (auto& s : strategies) {
        printf("  %-25s pass=%d  fail=%d  (%.1f%% success)\n",
               s.name, s.pass, s.fail,
               total > 0 ? 100.0 * s.pass / total : 0);
    }
    printf("\n");
}

// =============================================================================
// Within-fiber reachability analysis
//
// Key question: for BPDs with no box-cross annihilation available,
// can within-fiber 2x2 moves always reach a BPD that does have one?
//
// Within-fiber moves (preserve permutation):
//   - drip:          NW=r-elbow, SE=blank     -> NW=blank, SE=j-elbow
//   - undrip:        NW=blank, SE=j-elbow     -> NW=r-elbow, SE=blank
//   - cross-drip:    NW=cross, SE=r-elbow     -> NW=j-elbow, SE=cross
//   - cross-undrip3: NW=j-elbow, SE=cross     -> NW=cross, SE=r-elbow
// =============================================================================

// Find all within-fiber neighbors (2x2 moves preserving permutation)
std::vector<std::vector<uint8_t>> withinFiberNeighbors(const BPD& bpd) {
    std::vector<std::vector<uint8_t>> result;
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            // Try drip
            { BPD temp(bpd); if (tryDrip(temp, i, j) && temp.isReduced()) result.push_back(temp.grid); }
            // Try undrip
            { BPD temp(bpd); if (tryUndrip(temp, i, j) && temp.isReduced()) result.push_back(temp.grid); }
            // Try cross-drip case 3 (NW=cross, SE=r-elbow)
            {
                uint8_t nw = bpd.get(i-1,j-1), se = bpd.get(i,j);
                if (nw == 1 && se == 2) {
                    BPD temp(bpd);
                    if (tryCrossDrip(temp, i, j) && temp.isReduced()) result.push_back(temp.grid);
                }
            }
            // Try cross-undrip case 3 (NW=j-elbow, SE=cross)
            {
                uint8_t nw = bpd.get(i-1,j-1), se = bpd.get(i,j);
                if (nw == 3 && se == 1) {
                    BPD temp(bpd);
                    if (tryCrossUndrip(temp, i, j) && temp.isReduced()) result.push_back(temp.grid);
                }
            }
        }
    }
    return result;
}

// Check if a BPD has any allowed box-cross annihilation
bool hasAllowedAnnihilation(const BPD& bpd) {
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            if (bpd.get(i-1,j-1) == 0 && bpd.get(i,j) == 1) {
                BPD temp(bpd);
                if (tryCrossUndrip(temp, i, j) && temp.isReduced()) return true;
            }
        }
    }
    return false;
}

void analyzeWithinFiberReachability(int n) {
    printf("=== Within-fiber reachability analysis for n=%d ===\n\n", n);

    auto all_rbpds = enumerateAllRBPDs(n);
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    // Group RBPDs by permutation
    struct PermKey {
        std::vector<int> perm;
        bool operator==(const PermKey& o) const { return perm == o.perm; }
    };
    struct PermHash {
        size_t operator()(const PermKey& p) const {
            size_t h = 0;
            for (int v : p.perm) h = h * 31 + v;
            return h;
        }
    };

    std::unordered_map<size_t, std::vector<size_t>> perm_to_bpds; // perm_hash -> indices
    std::vector<std::vector<int>> perms(all_rbpds.size());
    std::vector<size_t> perm_hashes(all_rbpds.size());

    for (size_t idx = 0; idx < all_rbpds.size(); idx++) {
        BPD bpd(n, all_rbpds[idx]);
        perms[idx] = bpd.computePerm();
        size_t h = 0;
        for (int v : perms[idx]) h = h * 31 + v;
        perm_hashes[idx] = h;
        perm_to_bpds[h].push_back(idx);
    }

    // For each non-identity permutation with "stuck" BPDs, check reachability
    int total_stuck = 0;
    int stuck_can_reach = 0;
    int stuck_cannot_reach = 0;
    int total_fibers = 0;
    int connected_fibers = 0;
    int disconnected_fibers = 0;
    int max_fiber_diameter = 0;
    int max_dist_to_annihilation = 0;
    std::vector<int> dist_histogram(20, 0); // dist_histogram[d] = how many stuck BPDs at distance d

    std::unordered_set<size_t> seen_perms;

    for (size_t idx = 0; idx < all_rbpds.size(); idx++) {
        if (perms[idx] == identity) continue;
        size_t ph = perm_hashes[idx];
        if (seen_perms.count(ph)) continue;
        seen_perms.insert(ph);

        const auto& fiber_indices = perm_to_bpds[ph];
        if (fiber_indices.size() <= 1) {
            total_fibers++;
            // Single BPD in fiber - check if it has annihilation (it should, it's the Rothe)
            BPD bpd(n, all_rbpds[fiber_indices[0]]);
            if (bpd.isRothe()) {
                connected_fibers++;
            } else {
                // Shouldn't happen for single-BPD fibers
                connected_fibers++;
            }
            // Check if stuck
            if (!hasAllowedAnnihilation(bpd)) {
                total_stuck++;
                stuck_cannot_reach++;
            }
            continue;
        }

        total_fibers++;

        // Build within-fiber graph
        std::unordered_map<size_t, int> grid_to_fiberidx;
        for (int fi = 0; fi < (int)fiber_indices.size(); fi++) {
            GridHash gh;
            grid_to_fiberidx[gh(all_rbpds[fiber_indices[fi]])] = fi;
        }

        // Build adjacency list using within-fiber moves
        std::vector<std::vector<int>> adj(fiber_indices.size());
        for (int fi = 0; fi < (int)fiber_indices.size(); fi++) {
            BPD bpd(n, all_rbpds[fiber_indices[fi]]);
            auto nbrs = withinFiberNeighbors(bpd);
            for (const auto& ng : nbrs) {
                GridHash gh;
                auto it = grid_to_fiberidx.find(gh(ng));
                if (it != grid_to_fiberidx.end()) {
                    adj[fi].push_back(it->second);
                }
            }
        }

        // Check fiber connectivity via BFS from node 0
        std::vector<int> dist(fiber_indices.size(), -1);
        std::queue<int> q;
        dist[0] = 0;
        q.push(0);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }

        bool all_reached = true;
        int diameter = 0;
        for (int fi = 0; fi < (int)fiber_indices.size(); fi++) {
            if (dist[fi] == -1) { all_reached = false; break; }
            diameter = std::max(diameter, dist[fi]);
        }

        if (all_reached) {
            connected_fibers++;
            // Full BFS from every node to get true diameter
            for (int src = 1; src < (int)fiber_indices.size(); src++) {
                std::vector<int> d2(fiber_indices.size(), -1);
                std::queue<int> q2;
                d2[src] = 0; q2.push(src);
                while (!q2.empty()) {
                    int u = q2.front(); q2.pop();
                    for (int v : adj[u]) {
                        if (d2[v] == -1) { d2[v] = d2[u] + 1; q2.push(v); }
                    }
                }
                for (int fi = 0; fi < (int)fiber_indices.size(); fi++)
                    diameter = std::max(diameter, d2[fi]);
            }
            max_fiber_diameter = std::max(max_fiber_diameter, diameter);
        } else {
            disconnected_fibers++;
        }

        // Check annihilation reachability
        // Mark which fiber nodes have annihilation
        std::vector<bool> has_ann(fiber_indices.size());
        for (int fi = 0; fi < (int)fiber_indices.size(); fi++) {
            BPD bpd(n, all_rbpds[fiber_indices[fi]]);
            has_ann[fi] = hasAllowedAnnihilation(bpd);
        }

        // For each stuck node, BFS to find nearest node with annihilation
        for (int fi = 0; fi < (int)fiber_indices.size(); fi++) {
            if (has_ann[fi]) continue; // not stuck

            total_stuck++;
            // BFS from fi
            std::vector<int> d3(fiber_indices.size(), -1);
            std::queue<int> q3;
            d3[fi] = 0; q3.push(fi);
            int min_dist_to_ann = -1;
            while (!q3.empty()) {
                int u = q3.front(); q3.pop();
                if (has_ann[u]) {
                    min_dist_to_ann = d3[u];
                    break;
                }
                for (int v : adj[u]) {
                    if (d3[v] == -1) { d3[v] = d3[u] + 1; q3.push(v); }
                }
            }

            if (min_dist_to_ann >= 0) {
                stuck_can_reach++;
                max_dist_to_annihilation = std::max(max_dist_to_annihilation, min_dist_to_ann);
                if (min_dist_to_ann < (int)dist_histogram.size())
                    dist_histogram[min_dist_to_ann]++;
            } else {
                stuck_cannot_reach++;
            }
        }
    }

    printf("Fiber statistics:\n");
    printf("  Total non-identity fibers: %d\n", total_fibers);
    printf("  Connected (within-fiber 2x2): %d\n", connected_fibers);
    printf("  Disconnected: %d\n", disconnected_fibers);
    printf("  Max fiber diameter: %d\n", max_fiber_diameter);
    printf("\n");

    printf("Stuck BPD reachability (BPDs with no direct annihilation):\n");
    printf("  Total stuck: %d\n", total_stuck);
    printf("  Can reach annihilation via within-fiber moves: %d\n", stuck_can_reach);
    printf("  CANNOT reach annihilation: %d\n", stuck_cannot_reach);
    printf("  Max distance to annihilation: %d\n", max_dist_to_annihilation);

    if (total_stuck > 0) {
        printf("  Distance distribution:\n");
        for (int d = 0; d < (int)dist_histogram.size(); d++) {
            if (dist_histogram[d] > 0)
                printf("    dist=%d: %d BPDs\n", d, dist_histogram[d]);
        }
    }

    if (stuck_cannot_reach == 0 && disconnected_fibers == 0)
        printf("\n*** WITHIN-FIBER 2x2 CONNECTIVITY HOLDS: all fibers connected, all stuck BPDs can reach annihilation ***\n");
    else if (stuck_cannot_reach > 0)
        printf("\n*** WARNING: %d stuck BPDs CANNOT reach annihilation via within-fiber 2x2 moves ***\n", stuck_cannot_reach);

    printf("\n");
}

// =============================================================================
// Investigate stuck-unreachable BPDs: what moves DO they have?
// For BPDs that can't reach annihilation within their fiber,
// explore what between-fiber moves are available and trace escape paths.
// =============================================================================

// Find ALL 2x2 neighbors (any move type, both within and between fiber)
std::vector<std::pair<std::vector<uint8_t>, std::string>> allNeighborsLabeled(const BPD& bpd) {
    std::vector<std::pair<std::vector<uint8_t>, std::string>> result;
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            char label[64];
            { BPD t(bpd); if (tryDrip(t, i, j) && t.isReduced()) {
                snprintf(label, sizeof(label), "drip(%d,%d)", i, j);
                result.push_back({t.grid, label});
            }}
            { BPD t(bpd); if (tryUndrip(t, i, j) && t.isReduced()) {
                snprintf(label, sizeof(label), "undrip(%d,%d)", i, j);
                result.push_back({t.grid, label});
            }}
            { BPD t(bpd); if (tryCrossDrip(t, i, j) && t.isReduced()) {
                snprintf(label, sizeof(label), "crossdrip(%d,%d)", i, j);
                result.push_back({t.grid, label});
            }}
            { BPD t(bpd); if (tryCrossUndrip(t, i, j) && t.isReduced()) {
                snprintf(label, sizeof(label), "crossundrip(%d,%d)", i, j);
                result.push_back({t.grid, label});
            }}
        }
    }
    return result;
}

bool hasAllowedUpMove(const BPD& bpd) {
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            BPD t1(bpd);
            if (tryDrip(t1, i, j) && t1.isReduced()) return true;
            BPD t2(bpd);
            if (tryCrossDrip(t2, i, j) && t2.isReduced()) return true;
        }
    }
    return false;
}

bool hasAllowedDownMove(const BPD& bpd) {
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            BPD t1(bpd);
            if (tryUndrip(t1, i, j) && t1.isReduced()) return true;
            BPD t2(bpd);
            if (tryCrossUndrip(t2, i, j) && t2.isReduced()) return true;
        }
    }
    return false;
}

struct EscapeResult {
    bool found = false;
    bool hit_limit = false;
    int nodes_visited = 0;
    int path_length = -1;
    std::vector<std::vector<uint8_t>> path_states;  // includes start and target
    std::vector<std::string> path_moves;            // move k takes state k -> k+1
};

EscapeResult findEscapePathToLowerEll(const BPD& start, int max_nodes = 1000000) {
    EscapeResult result;
    const int n = start.n;
    const int start_ell = coxeterLength(start.computePerm());

    struct ParentInfo {
        std::vector<uint8_t> parent_grid;
        std::string move;
        bool has_parent = false;
    };

    std::queue<std::vector<uint8_t>> q;
    std::unordered_set<std::vector<uint8_t>, GridHash> visited;
    std::unordered_map<std::vector<uint8_t>, ParentInfo, GridHash> parent;

    visited.insert(start.grid);
    parent[start.grid] = ParentInfo{};
    q.push(start.grid);
    result.nodes_visited = 1;

    std::vector<uint8_t> target_grid;

    while (!q.empty()) {
        auto cur_grid = q.front();
        q.pop();

        BPD cur(n, cur_grid);
        int cur_ell = coxeterLength(cur.computePerm());
        if (cur_grid != start.grid && cur_ell < start_ell) {
            result.found = true;
            target_grid = cur_grid;
            break;
        }

        auto nbrs = allNeighborsLabeled(cur);
        for (const auto& p : nbrs) {
            const auto& ng = p.first;
            const auto& label = p.second;
            if (visited.find(ng) != visited.end()) continue;

            visited.insert(ng);
            ParentInfo info;
            info.parent_grid = cur_grid;
            info.move = label;
            info.has_parent = true;
            parent[ng] = std::move(info);
            q.push(ng);
            result.nodes_visited++;

            if (result.nodes_visited >= max_nodes) {
                result.hit_limit = true;
                while (!q.empty()) q.pop();
                break;
            }
        }
    }

    if (!result.found) return result;

    std::vector<std::vector<uint8_t>> rev_states;
    std::vector<std::string> rev_moves;
    std::vector<uint8_t> cur = target_grid;
    while (cur != start.grid) {
        rev_states.push_back(cur);
        const auto it = parent.find(cur);
        if (it == parent.end() || !it->second.has_parent) break;
        rev_moves.push_back(it->second.move);
        cur = it->second.parent_grid;
    }
    rev_states.push_back(start.grid);

    std::reverse(rev_states.begin(), rev_states.end());
    std::reverse(rev_moves.begin(), rev_moves.end());

    result.path_states = std::move(rev_states);
    result.path_moves = std::move(rev_moves);
    result.path_length = (int)result.path_moves.size();
    return result;
}

void runTryApproach12(int n, bool verbose) {
    printf("=== TRY.md Approach 1 + 2 check for n=%d ===\n\n", n);
    auto all_rbpds = enumerateAllRBPDs(n);
    printf("Total RBPDs enumerated: %zu\n", all_rbpds.size());

    std::vector<int> identity(n), w0(n);
    std::iota(identity.begin(), identity.end(), 1);
    for (int i = 0; i < n; i++) w0[i] = n - i;

    int non_w0 = 0;
    int non_w0_has_up = 0;
    int non_w0_no_up = 0;

    int non_identity = 0;
    int non_identity_has_down = 0;
    int non_identity_no_down = 0;

    std::vector<BPD> stuck_no_down;

    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        auto perm = bpd.computePerm();

        if (perm != w0) {
            non_w0++;
            if (hasAllowedUpMove(bpd)) non_w0_has_up++;
            else {
                non_w0_no_up++;
                if (verbose && non_w0_no_up <= 5) {
                    printf("\n  non-w0 with NO up-move: perm=%s ell=%d\n",
                           permToString(perm).c_str(), coxeterLength(perm));
                    bpd.print();
                }
            }
        }

        if (perm != identity) {
            non_identity++;
            if (hasAllowedDownMove(bpd)) non_identity_has_down++;
            else {
                non_identity_no_down++;
                stuck_no_down.push_back(bpd);
            }
        }
    }

    BPD bpd_w0(n);
    bpd_w0.generateRotheBPD(w0);
    bool w0_has_up = hasAllowedUpMove(bpd_w0);

    printf("\nApproach 1 (no non-w0 local maxima under up-moves):\n");
    printf("  Non-w0 states: %d\n", non_w0);
    printf("  Has >=1 allowed up-move: %d\n", non_w0_has_up);
    printf("  Has NO allowed up-move: %d\n", non_w0_no_up);
    printf("  w0 has allowed up-move: %s (expected no)\n", w0_has_up ? "YES" : "no");
    if (non_w0_no_up == 0 && !w0_has_up) {
        printf("  RESULT: PASSES at n=%d\n", n);
    } else {
        printf("  RESULT: FAILS at n=%d\n", n);
    }

    printf("\nApproach 2 setup (stuck states with no down-move):\n");
    printf("  Non-identity states: %d\n", non_identity);
    printf("  Has >=1 allowed down-move: %d\n", non_identity_has_down);
    printf("  Has NO allowed down-move: %d\n", non_identity_no_down);

    int escapes_found = 0;
    int escapes_failed = 0;
    int max_path_len = 0;

    for (size_t idx = 0; idx < stuck_no_down.size(); idx++) {
        const BPD& start = stuck_no_down[idx];
        auto start_perm = start.computePerm();
        int start_ell = coxeterLength(start_perm);

        EscapeResult er = findEscapePathToLowerEll(start);

        printf("\n--- Stuck state %zu/%zu ---\n", idx + 1, stuck_no_down.size());
        printf("Start perm=%s ell=%d\n", permToString(start_perm).c_str(), start_ell);
        start.print();

        // Detailed: show all j-elbows and crosses, and what down-moves are available/blocked
        printf("  J-elbows at:");
        for (int ii = 0; ii < n; ii++)
            for (int jj = 0; jj < n; jj++)
                if (start.get(ii,jj) == 3) printf(" (%d,%d)", ii, jj);
        printf("\n  Crosses at:");
        for (int ii = 0; ii < n; ii++)
            for (int jj = 0; jj < n; jj++)
                if (start.get(ii,jj) == 1) printf(" (%d,%d)", ii, jj);
        printf("\n");

        // Show all available down-moves and why they fail
        printf("  Down-move attempts:\n");
        for (int ii = 1; ii < n; ii++) {
            for (int jj = 1; jj < n; jj++) {
                // Undrip
                {
                    uint8_t nw = start.get(ii-1,jj-1), se = start.get(ii,jj);
                    uint8_t ne = start.get(ii-1,jj), sw = start.get(ii,jj-1);
                    if (nw == 0 && se == 3 && (ne == 2 || ne == 4) && (sw == 2 || sw == 5)) {
                        BPD t(start);
                        tryUndrip(t, ii, jj);
                        bool red = t.isReduced();
                        printf("    undrip(%d,%d): available, reduced=%s\n", ii, jj, red?"YES":"NO");
                    }
                }
                // CrossUndrip
                {
                    uint8_t nw = start.get(ii-1,jj-1), se = start.get(ii,jj);
                    const char* subcase = nullptr;
                    if (nw == 0 && se == 1) subcase = "box+cross→annihil";
                    else if (nw == 3 && se == 3) subcase = "jelbow+jelbow";
                    else if (nw == 3 && se == 1) subcase = "jelbow+cross";
                    if (subcase) {
                        BPD t(start);
                        tryCrossUndrip(t, ii, jj);
                        bool red = t.isReduced();
                        printf("    crossundrip(%d,%d) [%s]: available, reduced=%s\n",
                               ii, jj, subcase, red?"YES":"NO");
                    }
                }
            }
        }
        if (!er.found) {
            escapes_failed++;
            printf("  Escape search FAILED (visited %d nodes%s)\n",
                   er.nodes_visited, er.hit_limit ? ", hit node limit" : "");
            continue;
        }

        escapes_found++;
        if (er.path_length > max_path_len) max_path_len = er.path_length;
        printf("  Escape found: path length = %d, nodes visited = %d\n",
               er.path_length, er.nodes_visited);

        for (size_t s = 0; s < er.path_states.size(); s++) {
            BPD step_bpd(n, er.path_states[s]);
            auto step_perm = step_bpd.computePerm();
            int step_ell = coxeterLength(step_perm);
            printf("  Step %zu: perm=%s ell=%d\n",
                   s, permToString(step_perm).c_str(), step_ell);
            step_bpd.print();
            if (s < er.path_moves.size()) {
                printf("    move -> %s\n", er.path_moves[s].c_str());
            }
        }
    }

    printf("\nApproach 2 summary:\n");
    printf("  Stuck states analyzed: %zu\n", stuck_no_down.size());
    printf("  Escapes to lower ell found: %d\n", escapes_found);
    printf("  Escapes failed: %d\n", escapes_failed);
    printf("  Max escape path length: %d\n", max_path_len);
    if (escapes_failed == 0) {
        printf("  RESULT: PASSES at n=%d (for all stuck states found)\n", n);
    } else {
        printf("  RESULT: INCONCLUSIVE/FAIL at n=%d\n", n);
    }
    printf("\n");
}

// Get all allowed up-move neighbors with descriptions
std::vector<std::pair<std::vector<uint8_t>, std::string>> getUpNeighbors(const BPD& bpd) {
    std::vector<std::pair<std::vector<uint8_t>, std::string>> result;
    int n = bpd.n;
    for (int i = 1; i < n; i++) for (int j = 1; j < n; j++) {
        {
            BPD t(bpd);
            if (tryDrip(t, i, j) && t.isReduced()) {
                char buf[64]; snprintf(buf, sizeof(buf), "drip(%d,%d)", i, j);
                result.push_back({t.grid, std::string(buf)});
            }
        }
        {
            BPD t(bpd);
            if (tryCrossDrip(t, i, j) && t.isReduced()) {
                char buf[64]; snprintf(buf, sizeof(buf), "crossdrip(%d,%d)", i, j);
                result.push_back({t.grid, std::string(buf)});
            }
        }
    }
    return result;
}

// Get all allowed down-move neighbors with descriptions
std::vector<std::pair<std::vector<uint8_t>, std::string>> getDownNeighbors(const BPD& bpd) {
    std::vector<std::pair<std::vector<uint8_t>, std::string>> result;
    int n = bpd.n;
    for (int i = 1; i < n; i++) for (int j = 1; j < n; j++) {
        {
            BPD t(bpd);
            if (tryUndrip(t, i, j) && t.isReduced()) {
                char buf[64]; snprintf(buf, sizeof(buf), "undrip(%d,%d)", i, j);
                result.push_back({t.grid, std::string(buf)});
            }
        }
        {
            BPD t(bpd);
            if (tryCrossUndrip(t, i, j) && t.isReduced()) {
                char buf[64]; snprintf(buf, sizeof(buf), "crossundrip(%d,%d)", i, j);
                result.push_back({t.grid, std::string(buf)});
            }
        }
    }
    return result;
}

// Helper: print detailed anatomy of a single stuck BPD
void printStuckAnatomy(const BPD& bpd, int idx) {
    int n = bpd.n;
    auto perm = bpd.computePerm();
    int ell = coxeterLength(perm);
    printf("========== STUCK STATE #%d ==========\n", idx);
    printf("Perm = %s, ell = %d\n", permToString(perm).c_str(), ell);
    bpd.print();
    printf("  Pipe labels at crosses (SouthPipe,WestPipe):\n");
    bpd.printWithPipes();

    // Count tile types
    int nc=0, nj=0, nr=0, nbox=0, nv=0, nh=0;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        switch(bpd.get(i,j)) {
            case 0: nbox++; break; case 1: nc++; break; case 2: nr++; break;
            case 3: nj++; break; case 4: nv++; break; case 5: nh++; break;
        }
    }
    printf("Tiles: %d cross, %d j-elbow, %d r-elbow, %d box, %d vert, %d horiz\n",
           nc, nj, nr, nbox, nv, nh);

    // J-elbow positions and their neighborhoods
    printf("\nJ-elbow neighborhoods:\n");
    const char* names[] = {".", "+", "r", "j", "|", "-", "X"};
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        if (bpd.get(i,j) != 3) continue;
        printf("  J-elbow at (%d,%d), anti-diag i+j=%d, diag i-j=%d\n", i, j, i+j, i-j);
        printf("    3x3 neighborhood:\n");
        for (int di = -1; di <= 1; di++) {
            printf("      ");
            for (int dj = -1; dj <= 1; dj++) {
                int ii = i+di, jj = j+dj;
                if (ii < 0 || ii >= n || jj < 0 || jj >= n) printf("X ");
                else printf("%s ", names[bpd.get(ii,jj)]);
            }
            printf("\n");
        }

        // What 2x2 windows include this J-elbow?
        printf("    As SE of window (%d,%d): ", i, j);
        if (i > 0 && j > 0) {
            uint8_t nw = bpd.get(i-1,j-1);
            printf("NW=%s → ", names[nw]);
            if (nw == 0) printf("undrip candidate");
            else if (nw == 1) printf("cross-undrip candidate");
            else printf("no down-move");
        } else printf("out of bounds");
        printf("\n");

        printf("    As NW of window (%d+1,%d+1): ", i, j);
        if (i+1 < n && j+1 < n) {
            uint8_t se = bpd.get(i+1,j+1);
            printf("SE=%s", names[se]);
        } else printf("out of bounds");
        printf("\n");

        // Check if NW has a cross (the "lock" pattern)
        if (i > 0 && j > 0) {
            uint8_t nw = bpd.get(i-1,j-1);
            if (nw == 1) printf("    *** LOCKED: cross at NW position (%d,%d)\n", i-1, j-1);
        }
    }

    // Which crosses could participate in annihilation? (box-cross → r+r)
    printf("\nCross+box annihilation candidates:\n");
    bool any_annihilation = false;
    for (int i = 1; i < n; i++) for (int j = 1; j < n; j++) {
        if (bpd.get(i,j) == 1 && bpd.get(i-1,j-1) == 0) {
            any_annihilation = true;
            BPD t(bpd);
            if (tryCrossUndrip(t, i, j)) {
                if (t.isReduced()) {
                    printf("  Cross(%d,%d)+Box(%d,%d): allowed\n", i, j, i-1, j-1);
                } else {
                    auto dc = t.findDoubleCrossing();
                    printf("  Cross(%d,%d)+Box(%d,%d): BLOCKED — pipes %d and %d cross twice\n",
                           i, j, i-1, j-1, dc.pipe_a, dc.pipe_b);
                    printf("      1st crossing at (%d,%d), 2nd (bad) at (%d,%d)\n",
                           dc.first_cross_row, dc.first_cross_col, dc.row, dc.col);
                }
            }
        }
    }
    if (!any_annihilation) printf("  (none)\n");

    // Show escape path: try each up-move, then check for down-moves
    printf("\nEscape paths (up-move → down-move):\n");
    auto up_neighbors = getUpNeighbors(bpd);
    int path_idx = 0;
    for (const auto& [up_grid, up_desc] : up_neighbors) {
        BPD up_bpd(n, up_grid);
        int up_ell = coxeterLength(up_bpd.computePerm());
        // Check if this up-neighbor has a down-move to lower ell
        auto down_neighbors = getDownNeighbors(up_bpd);
        for (const auto& [dn_grid, dn_desc] : down_neighbors) {
            BPD dn_bpd(n, dn_grid);
            int dn_ell = coxeterLength(dn_bpd.computePerm());
            if (dn_ell < ell) {
                path_idx++;
                printf("  Path %d: UP %s (ell %d→%d) → DOWN %s (ell %d→%d)\n",
                       path_idx, up_desc.c_str(), ell, up_ell,
                       dn_desc.c_str(), up_ell, dn_ell);
            }
        }
    }
    if (path_idx == 0) printf("  NO escape found!\n");
    printf("\n");
}

// Detailed anatomy of stuck BPDs
void anatomyOfStuckBPDs(int n) {
    printf("=== Anatomy of stuck BPDs for n=%d ===\n\n", n);
    auto all_rbpds = enumerateAllRBPDs(n);
    printf("Total RBPDs: %zu\n\n", all_rbpds.size());

    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    int idx = 0;
    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        auto perm = bpd.computePerm();
        if (perm == identity) continue;
        if (hasAllowedDownMove(bpd)) continue;
        idx++;
        printStuckAnatomy(bpd, idx);
    }

    if (idx == 0) printf("No stuck states found at n=%d.\n\n", n);
    else printf("Total stuck states: %d\n\n", idx);
}

bool has2StepEscape(const BPD& bpd, int ell); // forward declaration

// Generate stuck BPD candidates for larger n by exploring from Rothe BPDs
// of permutations that extend the known n=8 stuck permutations.
void generateStuckCandidates(int target_n) {
    printf("=== Generating stuck BPD candidates for n=%d ===\n\n", target_n);

    // The 10 stuck permutations at n=8 (1-indexed values)
    std::vector<std::vector<int>> stuck8 = {
        {5,4,1,7,3,2,8,6}, {4,6,3,1,8,2,7,5}, {3,6,5,2,8,1,7,4},
        {6,4,3,8,2,1,7,5}, {6,5,3,2,8,1,7,4}, {6,5,3,8,2,1,7,4},
        {3,6,5,2,1,8,4,7}, {6,5,3,2,1,8,4,7}, {5,4,3,7,2,1,8,6},
        {6,4,1,8,3,2,7,5}
    };

    std::unordered_set<std::vector<uint8_t>, GridHash> found_stuck;
    int total_bpds_checked = 0;
    int total_stuck = 0;

    // Generate extended permutations
    std::vector<std::vector<int>> candidates;

    for (const auto& w8 : stuck8) {
        // Extension 1: append fixed point(s) at end
        // w' = (w(1),...,w(8), 9) for n=9
        // w' = (w(1),...,w(8), 9, 10) for n=10
        {
            std::vector<int> w(w8);
            for (int k = 9; k <= target_n; k++) w.push_back(k);
            candidates.push_back(w);
        }
        // Extension 2: prepend fixed point(s) at start
        // w' = (1, w(1)+1,...,w(8)+1) for n=9
        {
            std::vector<int> w;
            for (int k = 0; k < target_n - 8; k++) w.push_back(k + 1);
            for (int v : w8) w.push_back(v + (target_n - 8));
            candidates.push_back(w);
        }
        // Extension 3: insert fixed point in middle positions
        for (int pos = 1; pos < 8; pos++) {
            if (target_n != 9) continue; // only for n=9
            std::vector<int> w;
            for (int i = 0; i < 8; i++) {
                if (i == pos) w.push_back(pos + 1);
                int v = w8[i];
                w.push_back(v >= pos + 1 ? v + 1 : v);
            }
            if ((int)w.size() == 8) w.push_back(pos + 1);
            if ((int)w.size() != target_n) continue;
            // Verify it's a valid permutation
            std::vector<int> sorted_w(w);
            std::sort(sorted_w.begin(), sorted_w.end());
            bool valid = true;
            for (int i = 0; i < target_n; i++) if (sorted_w[i] != i+1) valid = false;
            if (valid) candidates.push_back(w);
        }
    }

    // Remove duplicate candidates
    std::set<std::vector<int>> seen;
    std::vector<std::vector<int>> unique_candidates;
    for (const auto& c : candidates) {
        if (seen.insert(c).second) unique_candidates.push_back(c);
    }
    candidates = unique_candidates;

    printf("Generated %zu candidate permutations in S_%d\n\n", candidates.size(), target_n);

    // For each candidate, explore BPDs via BFS from Rothe using ALL moves
    for (size_t ci = 0; ci < candidates.size(); ci++) {
        const auto& w = candidates[ci];
        int ell = coxeterLength(w);

        BPD rothe(target_n);
        rothe.generateRotheBPD(w);

        // BFS from Rothe using all moves
        std::unordered_set<std::vector<uint8_t>, GridHash> visited;
        std::queue<std::vector<uint8_t>> q;
        visited.insert(rothe.grid);
        q.push(rothe.grid);

        int bpds_this_perm = 0;
        int stuck_this_perm = 0;
        int max_bfs = 100000; // limit per permutation

        while (!q.empty() && (int)visited.size() < max_bfs) {
            auto cur_grid = q.front(); q.pop();
            BPD cur(target_n, cur_grid);
            bpds_this_perm++;
            total_bpds_checked++;

            // Check if this BPD is stuck (non-identity, no allowed down-move)
            auto cperm = cur.computePerm();
            std::vector<int> id(target_n);
            std::iota(id.begin(), id.end(), 1);
            if (cperm != id && !hasAllowedDownMove(cur)) {
                if (found_stuck.find(cur_grid) == found_stuck.end()) {
                    found_stuck.insert(cur_grid);
                    total_stuck++;
                    stuck_this_perm++;
                    printStuckAnatomy(cur, total_stuck);
                }
            }

            // Expand using all moves
            std::vector<std::vector<uint8_t>> neighbors;
            getNeighbors(cur, neighbors);
            for (const auto& nb : neighbors) {
                if (visited.find(nb) == visited.end()) {
                    visited.insert(nb);
                    q.push(nb);
                }
            }
        }

        if (ci % 20 == 0 || stuck_this_perm > 0) {
            printf("  Candidate %zu/%zu: perm=%s ell=%d, explored %d BPDs, %d stuck\n",
                   ci+1, candidates.size(), permToString(w).c_str(), ell,
                   bpds_this_perm, stuck_this_perm);
        }
    }

    printf("\n=== Phase 1 summary for n=%d ===\n", target_n);
    printf("Candidate permutations tested: %zu\n", candidates.size());
    printf("Total BPDs checked: %d\n", total_bpds_checked);
    printf("Stuck BPDs found: %d\n\n", total_stuck);

    // Phase 2: broad random search
    printf("=== Phase 2: broad random exploration for n=%d ===\n", target_n);
    int max_ell = target_n * (target_n - 1) / 2;
    int ell_lo = (int)(0.35 * max_ell);
    int ell_hi = (int)(0.65 * max_ell);
    printf("Targeting ell range [%d, %d] (max_ell=%d)\n", ell_lo, ell_hi, max_ell);

    srand(42);
    int phase2_checked = 0;
    int phase2_stuck = 0;
    int perms_tried = 0;
    int max_perms = 2000;
    int max_bfs2 = 50000;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (perms_tried = 0; perms_tried < max_perms; perms_tried++) {
        // Generate random permutation
        std::vector<int> w(target_n);
        std::iota(w.begin(), w.end(), 1);
        for (int i = target_n - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(w[i], w[j]);
        }
        int ell = coxeterLength(w);
        if (ell < ell_lo || ell > ell_hi) continue;

        BPD rothe(target_n);
        rothe.generateRotheBPD(w);

        std::unordered_set<std::vector<uint8_t>, GridHash> visited;
        std::queue<std::vector<uint8_t>> q;
        visited.insert(rothe.grid);
        q.push(rothe.grid);

        while (!q.empty() && (int)visited.size() < max_bfs2) {
            auto cur_grid = q.front(); q.pop();
            BPD cur(target_n, cur_grid);
            phase2_checked++;

            auto cperm = cur.computePerm();
            std::vector<int> id(target_n);
            std::iota(id.begin(), id.end(), 1);
            if (cperm != id && !hasAllowedDownMove(cur)) {
                if (found_stuck.find(cur_grid) == found_stuck.end()) {
                    found_stuck.insert(cur_grid);
                    total_stuck++;
                    phase2_stuck++;
                    printStuckAnatomy(cur, total_stuck);
                }
            }

            std::vector<std::vector<uint8_t>> neighbors;
            getNeighbors(cur, neighbors);
            for (const auto& nb : neighbors) {
                if (visited.find(nb) == visited.end()) {
                    visited.insert(nb);
                    q.push(nb);
                }
            }
        }

        if (perms_tried % 200 == 0) {
            auto t1 = std::chrono::high_resolution_clock::now();
            double secs = std::chrono::duration<double>(t1-t0).count();
            printf("\r  %d perms, %d BPDs checked, %d stuck (%.0f BPDs/s)   ",
                   perms_tried, phase2_checked, phase2_stuck, phase2_checked/secs);
            fflush(stdout);
        }
    }
    printf("\r                                                              \r");

    printf("\n=== Final summary for n=%d ===\n", target_n);
    printf("Phase 1: %zu candidate perms, %d BPDs\n", candidates.size(), total_bpds_checked);
    printf("Phase 2: %d random perms, %d BPDs\n", perms_tried, phase2_checked);
    printf("Total stuck BPDs found: %d\n", total_stuck);
    if (total_stuck > 0) {
        printf("*** Found %d stuck BPDs at n=%d! ***\n", total_stuck, target_n);
    }
    printf("\n");
}

// Analyze ALL non-identity BPDs without annihilation:
// - Distance to nearest BPD with annihilation (using ALL 2x2 moves)
// - Whether there's always a dEll=0 neighbor with annihilation
void analyzeStuckBPDs(int n, bool verbose) {
    printf("=== Stuck BPD analysis for n=%d ===\n\n", n);

    auto all_rbpds = enumerateAllRBPDs(n);
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);
    GridHash gh;

    int total_stuck = 0;
    int stuck_with_same_ell_ann_neighbor = 0;  // has dEll=0 neighbor with annihilation
    int stuck_needs_ell_increase = 0;          // only dEll>0 neighbors have annihilation
    int stuck_no_ann_neighbor = 0;             // no immediate neighbor has annihilation
    int max_min_dist = 0;

    // For BFS distance analysis
    std::vector<int> dist_histogram(20, 0);
    int shown = 0;

    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        auto perm = bpd.computePerm();
        if (perm == identity) continue;
        if (hasAllowedAnnihilation(bpd)) continue;

        total_stuck++;
        int ell = coxeterLength(perm);

        // Check all neighbors
        auto nbrs = allNeighborsLabeled(bpd);
        bool has_same_ell_ann = false;
        bool has_any_ann_neighbor = false;

        for (const auto& [ng, label] : nbrs) {
            BPD nbpd(n, ng);
            if (hasAllowedAnnihilation(nbpd)) {
                has_any_ann_neighbor = true;
                int nell = coxeterLength(nbpd.computePerm());
                if (nell == ell) has_same_ell_ann = true;
            }
        }

        if (has_same_ell_ann) {
            stuck_with_same_ell_ann_neighbor++;
            dist_histogram[1]++;
        } else if (has_any_ann_neighbor) {
            stuck_needs_ell_increase++;
            dist_histogram[1]++;
        } else {
            stuck_no_ann_neighbor++;
            // BFS to find distance
            std::unordered_map<size_t, int> visited;
            std::unordered_map<size_t, std::vector<uint8_t>> hash_to_grid;
            std::queue<size_t> bq;
            size_t sh = gh(grid);
            visited[sh] = 0; hash_to_grid[sh] = grid; bq.push(sh);
            int target_dist = -1;
            while (!bq.empty()) {
                size_t ch = bq.front(); bq.pop();
                int d = visited[ch];
                if (d > 6) continue;
                BPD cbpd(n, hash_to_grid[ch]);
                if (d > 0 && hasAllowedAnnihilation(cbpd)) {
                    target_dist = d; break;
                }
                auto nbs = allNeighborsLabeled(cbpd);
                for (const auto& [ng2, lbl] : nbs) {
                    size_t nh = gh(ng2);
                    if (!visited.count(nh)) {
                        visited[nh] = d + 1; hash_to_grid[nh] = ng2; bq.push(nh);
                    }
                }
            }
            if (target_dist >= 0) dist_histogram[target_dist]++;
            max_min_dist = std::max(max_min_dist, target_dist);

            if (verbose && shown < 5) {
                shown++;
                printf("  Stuck BPD needing >1 step: perm=(");
                for (int k = 0; k < n; k++) printf("%s%d", k?",":"", perm[k]);
                printf("), ell=%d, dist_to_ann=%d\n", ell, target_dist);
                bpd.print();
                printf("  Moves:\n");
                for (const auto& [ng, label] : nbrs) {
                    BPD nbpd(n, ng);
                    auto np = nbpd.computePerm();
                    int nell = coxeterLength(np);
                    printf("    %s -> ell=%d, has_ann=%s\n",
                           label.c_str(), nell, hasAllowedAnnihilation(nbpd) ? "YES" : "no");
                }
                printf("\n");
            }
        }
    }

    printf("Total stuck (no direct annihilation): %d\n", total_stuck);
    printf("  Has dEll=0 neighbor with annihilation: %d (%.1f%%)\n",
           stuck_with_same_ell_ann_neighbor,
           total_stuck > 0 ? 100.0 * stuck_with_same_ell_ann_neighbor / total_stuck : 0);
    printf("  Needs dEll>0 (only higher-ell neighbors have ann): %d (%.1f%%)\n",
           stuck_needs_ell_increase,
           total_stuck > 0 ? 100.0 * stuck_needs_ell_increase / total_stuck : 0);
    printf("  No immediate neighbor has annihilation: %d (%.1f%%)\n",
           stuck_no_ann_neighbor,
           total_stuck > 0 ? 100.0 * stuck_no_ann_neighbor / total_stuck : 0);
    printf("  Max min-distance to annihilation: %d\n\n", max_min_dist);

    if (stuck_no_ann_neighbor == 0 && stuck_needs_ell_increase == 0)
        printf("*** ALL stuck BPDs have a same-length neighbor with annihilation! ***\n\n");
    else if (stuck_no_ann_neighbor == 0)
        printf("*** All stuck BPDs reach annihilation in 1 step (some need ell increase) ***\n\n");
}

// =============================================================================
// Potential function verification
//
// Φ(D) = Σ 3^(i+j) for each (i,j) where tile is cross(1) or j-elbow(3)
//
// Claim: every allowed "down" move (undrip or cross-undrip preserving reducedness)
// strictly decreases Φ.
//
// Proof sketch: at window (i,j), the SE tile loses a cross/j-elbow (-3^{i+j}),
// NE/SW can each gain at most one j-elbow (+3^{i-1+j} + 3^{i+j-1} = (2/3)·3^{i+j}),
// NW either stays neutral or swaps j-elbow↔cross (Φ-neutral).
// Net: ΔΦ ≤ -(1/3)·3^{i+j} < 0.
// =============================================================================

double computePhi(const BPD& bpd) {
    double phi = 0;
    for (int i = 0; i < bpd.n; i++)
        for (int j = 0; j < bpd.n; j++) {
            uint8_t t = bpd.get(i, j);
            if (t == 1 || t == 3) { // cross or j-elbow
                phi += pow(3.0, i + j);
            }
        }
    return phi;
}

void verifyPhiProperty(int n) {
    printf("=== Φ potential function verification for n=%d ===\n\n", n);

    auto all_rbpds = enumerateAllRBPDs(n);
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    int total_moves = 0;
    int phi_decreased = 0;
    int phi_increased = 0;    // should be 0
    int phi_unchanged = 0;    // should be 0
    double max_ratio = 0;     // max(ΔΦ / 3^{i+j})
    int no_down_move = 0;
    int has_down_move = 0;

    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        auto perm = bpd.computePerm();
        if (perm == identity) continue;

        double phi_before = computePhi(bpd);
        bool found_down = false;

        // Try all down moves (undrip + cross-undrip at every position)
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                // Try undrip
                {
                    BPD temp(bpd);
                    if (tryUndrip(temp, i, j) && temp.isReduced()) {
                        found_down = true;
                        double phi_after = computePhi(temp);
                        double delta = phi_after - phi_before;
                        double expected_bound = -pow(3.0, i + j) / 3.0;
                        total_moves++;
                        if (delta < -0.5) phi_decreased++;
                        else if (delta > 0.5) phi_increased++;
                        else phi_unchanged++;
                        double ratio = delta / pow(3.0, i + j);
                        if (ratio > max_ratio) max_ratio = ratio;
                    }
                }
                // Try cross-undrip (all subcases)
                {
                    uint8_t nw = bpd.get(i-1,j-1), se = bpd.get(i,j);
                    if ((nw == 0 && se == 1) || (nw == 3 && se == 3) || (nw == 3 && se == 1)) {
                        BPD temp(bpd);
                        if (tryCrossUndrip(temp, i, j) && temp.isReduced()) {
                            found_down = true;
                            double phi_after = computePhi(temp);
                            double delta = phi_after - phi_before;
                            total_moves++;
                            if (delta < -0.5) phi_decreased++;
                            else if (delta > 0.5) phi_increased++;
                            else phi_unchanged++;
                            double ratio = delta / pow(3.0, i + j);
                            if (ratio > max_ratio) max_ratio = ratio;
                        }
                    }
                }
            }
        }

        if (found_down) has_down_move++;
        else {
            no_down_move++;
            if (no_down_move <= 20) {
                printf("\n!!! NO ALLOWED DOWN MOVE for perm (");
                for (int k = 0; k < n; k++) printf("%s%d", k?",":"", perm[k]+1);
                printf("), ell=%d:\n", bpd.countCrosses());
                bpd.print();
            }
        }
    }

    printf("Total allowed down moves tested: %d\n", total_moves);
    printf("  Φ decreased: %d\n", phi_decreased);
    printf("  Φ increased: %d  ← should be 0!\n", phi_increased);
    printf("  Φ unchanged: %d  ← should be 0!\n", phi_unchanged);
    printf("  Max ratio ΔΦ/3^(i+j): %.6f  ← should be ≤ -1/3 = -0.333...\n", max_ratio);
    printf("\nExistence of down moves:\n");
    printf("  Has at least one: %d\n", has_down_move);
    printf("  Has none: %d  ← should be 0!\n", no_down_move);

    if (phi_increased == 0 && phi_unchanged == 0 && no_down_move == 0)
        printf("\n*** Φ PROOF VERIFIED: every down move decreases Φ, and every non-id RBPD has one ***\n");
    else
        printf("\n*** VERIFICATION FAILED ***\n");

    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

// Check if BPD has a 2-step escape: one up-move then one down-move to lower ell
bool has2StepEscape(const BPD& bpd, int ell) {
    int n = bpd.n;
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            // Try drip up
            {
                BPD t(bpd);
                if (tryDrip(t, i, j) && t.isReduced()) {
                    for (int ii = 1; ii < n; ii++) {
                        for (int jj = 1; jj < n; jj++) {
                            { BPD t2(t); if (tryUndrip(t2, ii, jj) && t2.isReduced() && coxeterLength(t2.computePerm()) < ell) return true; }
                            { BPD t2(t); if (tryCrossUndrip(t2, ii, jj) && t2.isReduced() && coxeterLength(t2.computePerm()) < ell) return true; }
                        }
                    }
                }
            }
            // Try cross-drip up
            {
                BPD t(bpd);
                if (tryCrossDrip(t, i, j) && t.isReduced()) {
                    for (int ii = 1; ii < n; ii++) {
                        for (int jj = 1; jj < n; jj++) {
                            { BPD t2(t); if (tryUndrip(t2, ii, jj) && t2.isReduced() && coxeterLength(t2.computePerm()) < ell) return true; }
                            { BPD t2(t); if (tryCrossUndrip(t2, ii, jj) && t2.isReduced() && coxeterLength(t2.computePerm()) < ell) return true; }
                        }
                    }
                }
            }
        }
    }
    return false;
}

// Verify modified descent: every non-identity RBPD can reach a lower-ell
// RBPD in at most 2 moves (1 down, or 1 up + 1 down).
// Small-n version: enumerate all, store in memory.
void verifyModifiedDescent(int n) {
    printf("=== Modified descent check for n=%d ===\n", n);
    auto all_rbpds = enumerateAllRBPDs(n);
    printf("Total RBPDs: %zu\n", all_rbpds.size());

    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 1);

    int total_nonid = 0, direct_down = 0, escape_2step = 0, stuck_forever = 0;

    for (const auto& grid : all_rbpds) {
        BPD bpd(n, grid);
        auto perm = bpd.computePerm();
        if (perm == identity) continue;
        total_nonid++;

        if (hasAllowedDownMove(bpd)) { direct_down++; continue; }

        int ell = coxeterLength(perm);
        if (has2StepEscape(bpd, ell)) escape_2step++;
        else {
            stuck_forever++;
            if (stuck_forever <= 5) {
                printf("\n!!! NO 2-STEP ESCAPE for perm=%s ell=%d:\n",
                       permToString(perm).c_str(), ell);
                bpd.print();
            }
        }
    }

    printf("\nResults for n=%d:\n", n);
    printf("  Non-identity RBPDs: %d\n", total_nonid);
    printf("  Direct down-move (1 step): %d\n", direct_down);
    printf("  2-step escape (up+down): %d\n", escape_2step);
    printf("  No 2-step escape: %d\n", stuck_forever);
    if (stuck_forever == 0)
        printf("  *** MODIFIED DESCENT VERIFIED for n=%d ***\n\n", n);
    else
        printf("  *** MODIFIED DESCENT FAILS for n=%d ***\n\n", n);
}

// =============================================================================
// Packed grid key for memory-efficient BFS (3 bits per tile)
// =============================================================================

struct PackedGrid {
    uint64_t data[5];  // 320 bits, enough for n≤10 (100 tiles × 3 bits = 300 bits)
    PackedGrid() : data{0,0,0,0,0} {}
    bool operator==(const PackedGrid& o) const {
        return data[0]==o.data[0] && data[1]==o.data[1] && data[2]==o.data[2]
            && data[3]==o.data[3] && data[4]==o.data[4];
    }
};

struct PackedGridHash {
    size_t operator()(const PackedGrid& k) const {
        return k.data[0] ^ (k.data[1]*0x9e3779b97f4a7c15ULL)
             ^ (k.data[2]*0xc6a4a7935bd1e995ULL) ^ (k.data[3]*0xbf58476d1ce4e5b9ULL)
             ^ (k.data[4]*0x94d049bb133111ebULL);
    }
};

inline PackedGrid packGrid(const std::vector<uint8_t>& grid) {
    PackedGrid key;
    for (size_t i = 0; i < grid.size() && i < 106; i++) {
        size_t bp = i * 3, w = bp/64, b = bp%64;
        key.data[w] |= ((uint64_t)grid[i]) << b;
        if (b > 61 && w < 4) key.data[w+1] |= ((uint64_t)grid[i]) >> (64-b);
    }
    return key;
}

inline void unpackGrid(const PackedGrid& key, std::vector<uint8_t>& grid) {
    for (size_t i = 0; i < grid.size() && i < 106; i++) {
        size_t bp = i * 3, w = bp/64, b = bp%64;
        uint8_t val = (key.data[w] >> b) & 0x7;
        if (b > 61 && w < 4) val |= (key.data[w+1] << (64-b)) & 0x7;
        grid[i] = val;
    }
}

// Streaming BFS: enumerate RBPDs and check modified descent on the fly.
// Uses packed keys for visited set (40 bytes vs ~105 bytes per entry).
void verifyModifiedDescentStreaming(int n) {
    printf("=== Modified descent check (streaming) for n=%d ===\n", n);

    std::unordered_set<PackedGrid, PackedGridHash> visited;
    std::deque<PackedGrid> queue;

    // Start from identity and w0 Rothe BPDs
    std::vector<int> id_perm(n), w0_perm(n);
    std::iota(id_perm.begin(), id_perm.end(), 1);
    for (int i = 0; i < n; i++) w0_perm[i] = n - i;

    BPD bpd_id(n); bpd_id.generateRotheBPD(id_perm);
    BPD bpd_w0(n); bpd_w0.generateRotheBPD(w0_perm);

    auto pk_id = packGrid(bpd_id.grid);
    auto pk_w0 = packGrid(bpd_w0.grid);
    visited.insert(pk_id); queue.push_back(pk_id);
    if (!(pk_id == pk_w0)) { visited.insert(pk_w0); queue.push_back(pk_w0); }

    uint64_t total = 0, total_nonid = 0, direct_down = 0, escape_2step = 0, stuck_forever = 0;
    std::vector<uint8_t> buf(n*n);
    std::vector<std::vector<uint8_t>> neighbors;
    auto t0 = std::chrono::high_resolution_clock::now();

    while (!queue.empty()) {
        PackedGrid cur = queue.front(); queue.pop_front();
        unpackGrid(cur, buf);
        BPD bpd(n, buf);
        total++;

        // Check modified descent for this BPD
        auto perm = bpd.computePerm();
        if (perm != id_perm) {
            total_nonid++;
            if (hasAllowedDownMove(bpd)) {
                direct_down++;
            } else {
                int ell = coxeterLength(perm);
                if (has2StepEscape(bpd, ell)) {
                    escape_2step++;
                    printf("  2-step escape #%llu: perm=%s ell=%d\n",
                           escape_2step, permToString(perm).c_str(), ell);
                } else {
                    stuck_forever++;
                    printf("\n!!! NO 2-STEP ESCAPE for perm=%s ell=%d:\n",
                           permToString(perm).c_str(), ell);
                    bpd.print();
                }
            }
        }

        // Progress
        if (total % 500000 == 0) {
            auto t1 = std::chrono::high_resolution_clock::now();
            double secs = std::chrono::duration<double>(t1-t0).count();
            printf("\r  %llu RBPDs checked (%.0f/s), %llu stuck-down, %llu 2-step, %llu fail, queue=%zu   ",
                   total, total/secs, total_nonid-direct_down, escape_2step, stuck_forever, queue.size());
            fflush(stdout);
        }

        // Expand neighbors
        getNeighbors(bpd, neighbors);
        for (const auto& nb : neighbors) {
            auto pk = packGrid(nb);
            if (visited.find(pk) == visited.end()) {
                visited.insert(pk);
                queue.push_back(pk);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1-t0).count();
    printf("\r                                                                              \r");
    printf("Total RBPDs: %llu (%.1fs, %.0f/s)\n", total, secs, total/secs);
    printf("\nResults for n=%d:\n", n);
    printf("  Non-identity RBPDs: %llu\n", total_nonid);
    printf("  Direct down-move (1 step): %llu\n", direct_down);
    printf("  2-step escape (up+down): %llu\n", escape_2step);
    printf("  No 2-step escape: %llu\n", stuck_forever);
    if (stuck_forever == 0)
        printf("  *** MODIFIED DESCENT VERIFIED for n=%d ***\n\n", n);
    else
        printf("  *** MODIFIED DESCENT FAILS for n=%d ***\n\n", n);
}

int main(int argc, char* argv[]) {
    int maxN = 6;
    bool verbose = false;
    bool detailed = false;
    bool run_try = false;
    bool run_escape = false;
    bool run_anatomy = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) verbose = true;
        else if (strcmp(argv[i], "--detailed") == 0 || strcmp(argv[i], "-d") == 0) detailed = true;
        else if (strcmp(argv[i], "--try") == 0) run_try = true;
        else if (strcmp(argv[i], "--escape") == 0) run_escape = true;
        else if (strcmp(argv[i], "--anatomy") == 0) run_anatomy = true;
        else if (argv[i][0] != '-') maxN = atoi(argv[i]);
    }

    printf("Exploring Allowed 2x2 Moves on Reduced BPDs\n");
    printf("=============================================\n\n");

    if (run_try) {
        if (maxN < 3) maxN = 8;
        runTryApproach12(maxN, verbose);
        return 0;
    }

    if (run_anatomy) {
        if (maxN <= 8)
            anatomyOfStuckBPDs(maxN);
        else
            generateStuckCandidates(maxN);
        return 0;
    }

    if (run_escape) {
        for (int n = 3; n <= maxN; n++) {
            if (n <= 8)
                verifyModifiedDescent(n);
            else
                verifyModifiedDescentStreaming(n);
        }
        return 0;
    }

    for (int n = 3; n <= maxN; n++) {
        analyze(n, verbose);
        testCrossSelectionStrategies(n);
        if (detailed && n <= 6) {
            analyzeBlockedAnnihilations(n);
        }
        verifyPhiProperty(n);
    }

    return 0;
}
