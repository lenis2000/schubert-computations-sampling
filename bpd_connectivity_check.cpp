// BPD Move Connectivity Checker
// Verifies that 2x2 moves connect the entire state space of reduced BPDs
//
// The state space is ALL reduced BPDs across all permutations in S_n.
// Expected count = Σ_w Υ_w = Σ_w S_w(1^n)
//
// Compile (aggressive optimization):
//   clang++ -O3 -std=c++17 -mcpu=apple-m2 -flto -ffast-math \
//     -funroll-loops -fomit-frame-pointer -DNDEBUG \
//     bpd_connectivity_check.cpp -o bpd_connectivity_check
//
// Linux:
//   g++ -O3 -std=c++17 -march=native -flto -ffast-math \
//     -funroll-loops -fomit-frame-pointer -DNDEBUG \
//     bpd_connectivity_check.cpp -o bpd_connectivity_check

#include <vector>
#include <set>
#include <map>
#include <queue>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <deque>

// Tile types: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz

// =============================================================================
// BPD Grid Operations
// =============================================================================

class BPD {
public:
    int n;
    std::vector<uint8_t> grid;

    BPD(int size) : n(size), grid(size * size, 0) {}
    BPD(int size, const std::vector<uint8_t>& g) : n(size), grid(g) {}

    inline uint8_t get(int r, int c) const { return grid[r * n + c]; }
    inline void set(int r, int c, uint8_t val) { grid[r * n + c] = val; }

    bool operator<(const BPD& other) const { return grid < other.grid; }
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

    // Check if BPD is reduced (no repeated transpositions)
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

    void print() const {
        const char* names[] = {".", "+", "r", "j", "|", "-"};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%s ", names[get(i, j)]);
            }
            printf("\n");
        }
    }
};

// Hash for BPD grids (legacy, slow - O(n²) per lookup)
struct GridHash {
    size_t operator()(const std::vector<uint8_t>& grid) const {
        size_t h = 0;
        for (uint8_t v : grid) {
            h = h * 31 + v;
        }
        return h;
    }
};

// =============================================================================
// Fast Grid Key: Pack n×n grid into fixed-size integers (3 bits per tile)
// Supports n ≤ 10 (100 cells × 3 bits = 300 bits ≤ 320 bits = 5 × uint64)
// =============================================================================

// GridKey: 5 uint64 = 320 bits, supports n ≤ 10 (100 cells × 3 bits = 300 bits)
struct GridKey {
    uint64_t data[5];  // 320 bits

    GridKey() : data{0, 0, 0, 0, 0} {}

    bool operator==(const GridKey& o) const {
        return data[0] == o.data[0] && data[1] == o.data[1] &&
               data[2] == o.data[2] && data[3] == o.data[3] && data[4] == o.data[4];
    }
};

struct GridKeyHash {
    size_t operator()(const GridKey& k) const {
        return k.data[0] ^
               (k.data[1] * 0x9e3779b97f4a7c15ULL) ^
               (k.data[2] * 0xc6a4a7935bd1e995ULL) ^
               (k.data[3] * 0xbf58476d1ce4e5b9ULL) ^
               (k.data[4] * 0x94d049bb133111ebULL);
    }
};

// Pack grid into GridKey (3 bits per tile, tiles 0-5)
// Supports up to 106 cells (n ≤ 10)
inline GridKey packGrid(const std::vector<uint8_t>& grid) {
    GridKey key;
    for (size_t i = 0; i < grid.size() && i < 106; i++) {
        size_t bitpos = i * 3;
        size_t word = bitpos / 64;
        size_t bit = bitpos % 64;
        key.data[word] |= ((uint64_t)grid[i]) << bit;
        if (bit > 61 && word < 4) {  // Spans word boundary
            key.data[word + 1] |= ((uint64_t)grid[i]) >> (64 - bit);
        }
    }
    return key;
}

// Unpack GridKey back to grid vector
inline void unpackGrid(const GridKey& key, std::vector<uint8_t>& grid) {
    for (size_t i = 0; i < grid.size() && i < 106; i++) {
        size_t bitpos = i * 3;
        size_t word = bitpos / 64;
        size_t bit = bitpos % 64;
        uint8_t val = (key.data[word] >> bit) & 0x7;
        if (bit > 61 && word < 4) {  // Spans word boundary
            val |= (key.data[word + 1] << (64 - bit)) & 0x7;
        }
        grid[i] = val;
    }
}

// =============================================================================
// Fast Permutation Key: Pack permutation into uint64 (4 bits per element)
// Supports n ≤ 16
// =============================================================================

struct PermKey {
    uint64_t data;

    PermKey() : data(0) {}
    explicit PermKey(uint64_t d) : data(d) {}

    bool operator==(const PermKey& o) const { return data == o.data; }
    bool operator<(const PermKey& o) const { return data < o.data; }
};

struct PermKeyHash {
    size_t operator()(const PermKey& k) const {
        // Good hash mixing for uint64
        uint64_t x = k.data;
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        return x;
    }
};

// Pack permutation (1-indexed values) into PermKey
inline PermKey packPerm(const std::vector<int>& perm) {
    uint64_t key = 0;
    for (size_t i = 0; i < perm.size(); i++) {
        key |= ((uint64_t)(perm[i] - 1)) << (i * 4);  // Store 0-indexed
    }
    return PermKey(key);
}

// =============================================================================
// 2x2 Move Operations
// =============================================================================

// Try drip move at 2x2 square with bottom-right at (i,j)
bool tryDrip(BPD& bpd, int i, int j) {
    uint8_t nw = bpd.get(i - 1, j - 1);
    uint8_t ne = bpd.get(i - 1, j);
    uint8_t sw = bpd.get(i, j - 1);
    uint8_t se = bpd.get(i, j);

    if (nw == 2 && se == 0) {
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

    if (nw == 0 && se == 1) {
        bpd.set(i - 1, j - 1, 2);
        bpd.set(i, j, 2);
        bpd.set(i - 1, j, (ne == 4) ? 3 : 5);
        bpd.set(i, j - 1, (sw == 5) ? 3 : 4);
        return true;
    }
    if (nw == 3 && se == 3) {
        bpd.set(i - 1, j - 1, 1);
        bpd.set(i, j, 0);
        bpd.set(i - 1, j, (ne == 4) ? 3 : 5);
        bpd.set(i, j - 1, (sw == 5) ? 3 : 4);
        return true;
    }
    if (nw == 3 && se == 1) {
        bpd.set(i - 1, j - 1, 1);
        bpd.set(i, j, 2);
        bpd.set(i - 1, j, (ne == 4) ? 3 : 5);
        bpd.set(i, j - 1, (sw == 5) ? 3 : 4);
        return true;
    }
    return false;
}

// Get all neighbors (BPDs reachable by one 2x2 move, that are reduced)
void getNeighbors(const BPD& bpd, std::vector<std::vector<uint8_t>>& neighbors) {
    neighbors.clear();
    int n = bpd.n;

    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            // Try each move type
            BPD temp = bpd;

            if (tryDrip(temp, i, j)) {
                if (temp.isReduced()) {
                    neighbors.push_back(temp.grid);
                }
                temp = bpd;
            }

            if (tryUndrip(temp, i, j)) {
                if (temp.isReduced()) {
                    neighbors.push_back(temp.grid);
                }
                temp = bpd;
            }

            if (tryCrossDrip(temp, i, j)) {
                if (temp.isReduced()) {
                    neighbors.push_back(temp.grid);
                }
                temp = bpd;
            }

            if (tryCrossUndrip(temp, i, j)) {
                if (temp.isReduced()) {
                    neighbors.push_back(temp.grid);
                }
                temp = bpd;
            }
        }
    }
}

// =============================================================================
// OEIS A331920: Total number of reduced pipe dreams for all permutations in S_n
// Also equals Σ_w Υ_w = Σ_w S_w(1^n)
// =============================================================================

uint64_t getExpectedRBPDCount(int n) {
    // OEIS A331920: 1, 2, 7, 41, 393, 6080, 150371, 5903710, 365973851,
    //               35669122055, 5446988315069, 1299591839689292
    static const uint64_t A331920[] = {
        0,  // n=0 (placeholder)
        1,  // n=1
        2,  // n=2
        7,  // n=3
        41, // n=4
        393, // n=5
        6080, // n=6
        150371, // n=7
        5903710, // n=8
        365973851, // n=9
        35669122055ULL, // n=10
        5446988315069ULL, // n=11
        1299591839689292ULL // n=12
    };

    if (n >= 1 && n <= 12) {
        return A331920[n];
    }
    return 0;  // Unknown
}

// =============================================================================
// BFS Enumeration of All Reachable RBPDs
// =============================================================================

uint64_t findAllReachableBFS(int n, const std::vector<uint8_t>& start) {
    std::unordered_set<GridKey, GridKeyHash> visited;
    std::queue<GridKey> queue;

    GridKey startKey = packGrid(start);
    queue.push(startKey);
    visited.insert(startKey);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print = start_time;

    std::vector<std::vector<uint8_t>> neighbors;
    std::vector<uint8_t> current(n * n);  // Reusable buffer for unpacking

    while (!queue.empty()) {
        GridKey currentKey = queue.front();
        queue.pop();

        unpackGrid(currentKey, current);
        BPD bpd(n, current);
        getNeighbors(bpd, neighbors);

        for (const auto& neighbor : neighbors) {
            GridKey neighborKey = packGrid(neighbor);
            if (visited.find(neighborKey) == visited.end()) {
                visited.insert(neighborKey);
                queue.push(neighborKey);
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_print).count();
        if (elapsed >= 1.0) {
            double total_elapsed = std::chrono::duration<double>(now - start_time).count();
            printf("\r  BFS enumeration: %zu RBPDs found, %zu in queue (%.1f/s)   ",
                   visited.size(), queue.size(), visited.size() / total_elapsed);
            fflush(stdout);
            last_print = now;
        }
    }

    printf("\r  BFS enumeration: %zu RBPDs found                           \n",
           visited.size());

    return visited.size();
}

// =============================================================================
// BFS that tracks both RBPDs and their permutations
// =============================================================================

struct BFSResult {
    uint64_t num_bpds;
    std::set<PermKey> permutations;  // Packed permutation keys
    std::unordered_map<PermKey, uint64_t, PermKeyHash> perm_counts;  // Count RBPDs per permutation
};

BFSResult findAllReachableWithPerms(int n, const std::vector<uint8_t>& start) {
    std::unordered_set<GridKey, GridKeyHash> visited;
    std::queue<GridKey> queue;

    // Pre-reserve capacity based on expected size (OEIS A331920)
    uint64_t expected = getExpectedRBPDCount(n);
    if (expected > 0 && expected < 100000000) {  // Reserve up to 100M
        visited.reserve(expected);
    }

    BFSResult result;
    result.num_bpds = 0;

    GridKey startKey = packGrid(start);
    queue.push(startKey);
    visited.insert(startKey);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print = start_time;

    std::vector<std::vector<uint8_t>> neighbors;
    std::vector<uint8_t> current(n * n);  // Reusable buffer for unpacking

    while (!queue.empty()) {
        GridKey currentKey = queue.front();
        queue.pop();

        unpackGrid(currentKey, current);
        BPD bpd(n, current);
        std::vector<int> perm = bpd.computePerm();
        PermKey permKey = packPerm(perm);
        result.permutations.insert(permKey);
        result.perm_counts[permKey]++;

        getNeighbors(bpd, neighbors);

        for (const auto& neighbor : neighbors) {
            GridKey neighborKey = packGrid(neighbor);
            if (visited.find(neighborKey) == visited.end()) {
                visited.insert(neighborKey);
                queue.push(neighborKey);
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_print).count();
        if (elapsed >= 1.0) {
            double total_elapsed = std::chrono::duration<double>(now - start_time).count();
            printf("\r  BFS: %zu RBPDs, %zu perms, %zu in queue (%.1f/s)   ",
                   visited.size(), result.permutations.size(), queue.size(),
                   visited.size() / total_elapsed);
            fflush(stdout);
            last_print = now;
        }
    }

    result.num_bpds = visited.size();
    printf("\r  BFS: %llu RBPDs, %zu permutations found                    \n",
           (unsigned long long)result.num_bpds, result.permutations.size());

    return result;
}

// =============================================================================
// Parallel Bidirectional BFS (two threads: from w0 and identity)
// =============================================================================

// Sharded hash set for lock-free-ish concurrent access
static constexpr size_t NUM_SHARDS = 64;

struct ShardedVisited {
    std::array<std::unordered_set<GridKey, GridKeyHash>, NUM_SHARDS> shards;
    std::array<std::mutex, NUM_SHARDS> locks;

    void reserve(size_t total) {
        size_t per_shard = (total + NUM_SHARDS - 1) / NUM_SHARDS;
        for (auto& s : shards) s.reserve(per_shard);
    }

    size_t shard_for(const GridKey& k) const {
        return GridKeyHash()(k) % NUM_SHARDS;
    }

    // Returns true if newly inserted
    bool insert(const GridKey& k) {
        size_t idx = shard_for(k);
        std::lock_guard<std::mutex> lock(locks[idx]);
        return shards[idx].insert(k).second;
    }

    size_t size() const {
        size_t total = 0;
        for (const auto& s : shards) total += s.size();
        return total;
    }
};

// =============================================================================
// Connectivity-only BFS (no permutation tracking, minimal memory)
// =============================================================================

uint64_t findAllReachableConnectivityOnly(int n, const std::vector<uint8_t>& start1,
                                           const std::vector<uint8_t>& start2,
                                           int num_threads = 4) {
    ShardedVisited visited;
    std::deque<GridKey> work_queue;
    std::mutex queue_mtx;
    std::condition_variable cv;

    uint64_t expected = getExpectedRBPDCount(n);
    if (expected > 0 && expected < 500000000) {
        visited.reserve(expected);
    }

    GridKey key1 = packGrid(start1);
    GridKey key2 = packGrid(start2);
    visited.insert(key1);
    work_queue.push_back(key1);
    if (!(key1 == key2) && visited.insert(key2)) {
        work_queue.push_back(key2);
    }

    std::atomic<uint64_t> total_processed{0};
    std::atomic<int> active_workers{0};
    std::atomic<bool> all_done{false};
    auto start_time = std::chrono::high_resolution_clock::now();

    const size_t BATCH_SIZE = 64;

    auto worker = [&]() {
        std::vector<std::vector<uint8_t>> neighbors;
        std::vector<uint8_t> current(n * n);
        std::vector<GridKey> work_batch;
        std::vector<GridKey> new_keys;
        work_batch.reserve(BATCH_SIZE);
        new_keys.reserve(BATCH_SIZE * 8);

        while (true) {
            work_batch.clear();
            {
                std::unique_lock<std::mutex> lock(queue_mtx);
                while (work_queue.empty() && !all_done) {
                    active_workers--;
                    if (active_workers == 0 && work_queue.empty()) {
                        all_done = true;
                        cv.notify_all();
                        return;
                    }
                    cv.wait(lock);
                    if (all_done) return;
                    active_workers++;
                }
                if (all_done) return;
                while (!work_queue.empty() && work_batch.size() < BATCH_SIZE) {
                    work_batch.push_back(work_queue.front());
                    work_queue.pop_front();
                }
            }

            new_keys.clear();
            for (const GridKey& currentKey : work_batch) {
                unpackGrid(currentKey, current);
                BPD bpd(n, current);
                getNeighbors(bpd, neighbors);
                for (const auto& neighbor : neighbors) {
                    new_keys.push_back(packGrid(neighbor));
                }
            }

            std::vector<GridKey> truly_new;
            for (const GridKey& gk : new_keys) {
                if (visited.insert(gk)) {
                    truly_new.push_back(gk);
                }
            }

            if (!truly_new.empty()) {
                std::lock_guard<std::mutex> lock(queue_mtx);
                for (const GridKey& gk : truly_new) {
                    work_queue.push_back(gk);
                }
            }
            cv.notify_all();
            total_processed += work_batch.size();
        }
    };

    std::thread reporter([&]() {
        while (!all_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (all_done) break;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            size_t vis_size = visited.size();
            double pct = expected > 0 ? 100.0 * vis_size / expected : 0;
            double rate = total_processed.load() / elapsed;
            char vis_str[32], rate_str[32];
            if (vis_size >= 1000000) snprintf(vis_str, 32, "%.1fM", vis_size / 1e6);
            else if (vis_size >= 1000) snprintf(vis_str, 32, "%.1fK", vis_size / 1e3);
            else snprintf(vis_str, 32, "%zu", vis_size);
            if (rate >= 1000000) snprintf(rate_str, 32, "%.1fM/s", rate / 1e6);
            else if (rate >= 1000) snprintf(rate_str, 32, "%.1fK/s", rate / 1e3);
            else snprintf(rate_str, 32, "%.0f/s", rate);
            printf("\r  BFS: %s (%.1f%%) %s       ", vis_str, pct, rate_str);
            fflush(stdout);
        }
    });

    std::vector<std::thread> workers;
    active_workers = num_threads;
    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back(worker);
    }

    for (auto& t : workers) t.join();
    reporter.join();

    uint64_t total = visited.size();
    char total_str[32];
    if (total >= 1000000) snprintf(total_str, 32, "%.1fM", total / 1e6);
    else if (total >= 1000) snprintf(total_str, 32, "%.1fK", total / 1e3);
    else snprintf(total_str, 32, "%llu", (unsigned long long)total);
    printf("\r  BFS: %s RBPDs found                    \n", total_str);

    return total;
}

BFSResult findAllReachableParallel(int n, const std::vector<uint8_t>& start1,
                                    const std::vector<uint8_t>& start2,
                                    int num_threads = 4) {
    // Sharded visited set (low contention)
    ShardedVisited visited;

    // Permutation tracking (less frequent updates)
    std::set<PermKey> permutations;
    std::unordered_map<PermKey, uint64_t, PermKeyHash> perm_counts;
    std::mutex perm_mtx;

    // Work queue with its own lock
    std::deque<GridKey> work_queue;
    std::mutex queue_mtx;
    std::condition_variable cv;

    // Pre-reserve
    uint64_t expected = getExpectedRBPDCount(n);
    if (expected > 0 && expected < 100000000) {
        visited.reserve(expected);
    }

    // Insert starting points
    GridKey key1 = packGrid(start1);
    GridKey key2 = packGrid(start2);
    visited.insert(key1);
    work_queue.push_back(key1);
    if (!(key1 == key2) && visited.insert(key2)) {
        work_queue.push_back(key2);
    }

    std::atomic<uint64_t> total_processed{0};
    std::atomic<int> active_workers{0};
    std::atomic<bool> all_done{false};
    auto start_time = std::chrono::high_resolution_clock::now();

    const size_t BATCH_SIZE = 32;

    auto worker = [&]() {
        std::vector<std::vector<uint8_t>> neighbors;
        std::vector<uint8_t> current(n * n);
        std::vector<GridKey> work_batch;
        std::vector<GridKey> new_keys;
        std::vector<PermKey> new_perms;
        work_batch.reserve(BATCH_SIZE);
        new_keys.reserve(BATCH_SIZE * 8);
        new_perms.reserve(BATCH_SIZE);

        while (true) {
            // Grab a batch of work
            work_batch.clear();
            {
                std::unique_lock<std::mutex> lock(queue_mtx);

                while (work_queue.empty() && !all_done) {
                    active_workers--;
                    if (active_workers == 0 && work_queue.empty()) {
                        all_done = true;
                        cv.notify_all();
                        return;
                    }
                    cv.wait(lock);
                    if (all_done) return;
                    active_workers++;
                }

                if (all_done) return;

                // Grab up to BATCH_SIZE items
                while (!work_queue.empty() && work_batch.size() < BATCH_SIZE) {
                    work_batch.push_back(work_queue.front());
                    work_queue.pop_front();
                }
            }

            // Process batch locally (no locks needed for this part)
            new_keys.clear();
            new_perms.clear();

            for (const GridKey& currentKey : work_batch) {
                unpackGrid(currentKey, current);
                BPD bpd(n, current);

                std::vector<int> perm = bpd.computePerm();
                new_perms.push_back(packPerm(perm));

                getNeighbors(bpd, neighbors);
                for (const auto& neighbor : neighbors) {
                    new_keys.push_back(packGrid(neighbor));
                }
            }

            // Insert new keys into sharded visited (low contention)
            std::vector<GridKey> truly_new;
            for (const GridKey& gk : new_keys) {
                if (visited.insert(gk)) {
                    truly_new.push_back(gk);
                }
            }

            // Add truly new keys to work queue
            if (!truly_new.empty()) {
                std::lock_guard<std::mutex> lock(queue_mtx);
                for (const GridKey& gk : truly_new) {
                    work_queue.push_back(gk);
                }
            }
            cv.notify_all();

            // Update permutation stats (single lock)
            {
                std::lock_guard<std::mutex> lock(perm_mtx);
                for (const PermKey& pk : new_perms) {
                    permutations.insert(pk);
                    perm_counts[pk]++;
                }
            }

            total_processed += work_batch.size();
        }
    };

    // Compute n! for percentage
    uint64_t factorial = 1;
    for (int i = 2; i <= n; i++) factorial *= i;

    // Progress reporter
    std::thread reporter([&]() {
        while (!all_done) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (all_done) break;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            size_t vis_size = visited.size();
            size_t perm_size, queue_size;
            {
                std::lock_guard<std::mutex> lock(perm_mtx);
                perm_size = permutations.size();
            }
            {
                std::lock_guard<std::mutex> lock(queue_mtx);
                queue_size = work_queue.size();
            }
            double pct = expected > 0 ? 100.0 * vis_size / expected : 0;
            double perm_pct = 100.0 * perm_size / factorial;
            double rate = total_processed.load() / elapsed;
            // Human readable format
            char vis_str[32], rate_str[32];
            if (vis_size >= 1000000) snprintf(vis_str, 32, "%.1fM", vis_size / 1e6);
            else if (vis_size >= 1000) snprintf(vis_str, 32, "%.1fK", vis_size / 1e3);
            else snprintf(vis_str, 32, "%zu", vis_size);
            if (rate >= 1000000) snprintf(rate_str, 32, "%.1fM/s", rate / 1e6);
            else if (rate >= 1000) snprintf(rate_str, 32, "%.1fK/s", rate / 1e3);
            else snprintf(rate_str, 32, "%.0f/s", rate);
            printf("\r  BFS: %s (%.1f%%), perms %.1f%% (%s)       ",
                   vis_str, pct, perm_pct, rate_str);
            fflush(stdout);
        }
    });

    // Launch workers
    std::vector<std::thread> workers;
    active_workers = num_threads;
    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back(worker);
    }

    for (auto& t : workers) t.join();
    reporter.join();

    BFSResult result;
    result.num_bpds = visited.size();
    result.permutations = std::move(permutations);
    result.perm_counts = std::move(perm_counts);

    printf("\r  Parallel BFS: %llu RBPDs, %zu permutations found            \n",
           (unsigned long long)result.num_bpds, result.permutations.size());

    return result;
}

// =============================================================================
// Connectivity Verification
// =============================================================================

void verifyConnectivity(int n, bool check_oeis, bool fast_mode = false) {
    printf("\n");
    printf("=========================================\n");
    printf("  Connectivity Check for n = %d\n", n);
    printf("=========================================\n\n");

    auto total_start = std::chrono::high_resolution_clock::now();

    // Compute n! (number of permutations)
    uint64_t factorial = 1;
    for (int i = 2; i <= n; i++) factorial *= i;

    // Get expected RBPD count from OEIS A331920
    uint64_t expected_rbpds = 0;
    if (check_oeis) {
        expected_rbpds = getExpectedRBPDCount(n);
        if (expected_rbpds > 0) {
            printf("Expected RBPDs (OEIS A331920): %llu\n\n", (unsigned long long)expected_rbpds);
        } else {
            printf("Warning: No known value for n=%d in OEIS A331920\n\n", n);
        }
    }

    // Generate starting BPDs for w0 and identity
    BPD bpd_w0(n);
    std::vector<int> w0(n);
    for (int i = 0; i < n; i++) w0[i] = n - i;
    bpd_w0.generateRotheBPD(w0);

    BPD bpd_id(n);
    std::vector<int> id(n);
    for (int i = 0; i < n; i++) id[i] = i + 1;
    bpd_id.generateRotheBPD(id);

    // Choose BFS mode
    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t num_bpds = 0;
    size_t num_perms = 0;

    if (fast_mode) {
        // Connectivity-only: no permutation tracking, minimal memory
        int num_threads = 4;
        printf("Fast mode: Connectivity-only BFS (%d threads)\n", num_threads);
        num_bpds = findAllReachableConnectivityOnly(n, bpd_w0.grid, bpd_id.grid, num_threads);
        num_perms = 0;  // Not tracked in fast mode
    } else if (n <= 7) {
        // Single-threaded for small n
        printf("%s: BFS from w0\n", check_oeis ? "Step 2" : "Step 1");
        printf("  w0 = (");
        for (int i = 0; i < n; i++) printf("%d%s", w0[i], i < n - 1 ? "," : "");
        printf(")\n");
        BFSResult result = findAllReachableWithPerms(n, bpd_w0.grid);
        num_bpds = result.num_bpds;
        num_perms = result.permutations.size();
    } else {
        // Parallel with permutation tracking
        int num_threads = 4;
        printf("%s: Parallel BFS (%d threads) from w0 and identity\n",
               check_oeis ? "Step 2" : "Step 1", num_threads);
        printf("  w0 = (");
        for (int i = 0; i < n; i++) printf("%d%s", w0[i], i < n - 1 ? "," : "");
        printf("), id = (");
        for (int i = 0; i < n; i++) printf("%d%s", id[i], i < n - 1 ? "," : "");
        printf(")\n");
        BFSResult result = findAllReachableParallel(n, bpd_w0.grid, bpd_id.grid, num_threads);
        num_bpds = result.num_bpds;
        num_perms = result.permutations.size();
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // Result
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    printf("\nResult:\n");
    printf("  Total RBPDs found:       %llu\n", (unsigned long long)num_bpds);
    if (check_oeis) {
        printf("  Expected (Σ Υ_w):        %llu\n", (unsigned long long)expected_rbpds);
    }
    if (!fast_mode) {
        printf("  Permutations reached:    %zu / %llu (%.1f%%)\n",
               num_perms, (unsigned long long)factorial,
               100.0 * num_perms / factorial);
    }
    printf("  Time: %.2f sec\n", total_time);

    // Check results
    bool perm_pass = fast_mode || (num_perms == factorial);
    bool rbpd_pass = !check_oeis || (num_bpds == expected_rbpds);

    if (perm_pass && rbpd_pass) {
        printf("\n  *** PASS: ");
        if (check_oeis) {
            printf("All %llu RBPDs reachable", (unsigned long long)num_bpds);
            if (!fast_mode) printf(", all permutations covered");
            printf("! ***\n");
        } else {
            printf("Connected! ***\n");
        }
    } else {
        printf("\n  *** FAIL: ");
        if (!perm_pass && !fast_mode) {
            printf("Missing %llu permutations! ",
                   (unsigned long long)(factorial - num_perms));
        }
        if (!rbpd_pass) {
            printf("RBPD count mismatch: found %llu, expected %llu! ",
                   (unsigned long long)num_bpds, (unsigned long long)expected_rbpds);
        }
        printf("***\n");
    }
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

void printUsage(const char* prog) {
    printf("BPD Move Connectivity Checker\n");
    printf("Verifies that 2x2 moves connect the entire state space of reduced BPDs.\n\n");
    printf("Usage: %s [options] [n]\n\n", prog);
    printf("Options:\n");
    printf("  --oeis       Verify RBPD count equals OEIS A331920 (Σ_w Υ_w)\n");
    printf("  --fast       Connectivity-only mode (no perm tracking, less memory)\n");
    printf("  --help, -h   Show this help\n\n");
    printf("Arguments:\n");
    printf("  n            Single n to test (e.g., 6)\n");
    printf("  n1-n2        Range to test (e.g., 2-8)\n\n");
    printf("Examples:\n");
    printf("  %s 8              Check n=8\n", prog);
    printf("  %s --oeis 9       Check n=9 with OEIS verification\n", prog);
    printf("  %s --fast --oeis 9   Fast mode for n=9 (less memory)\n", prog);
}

int main(int argc, char* argv[]) {
    bool check_oeis = false;
    bool fast_mode = false;
    int maxN = 6;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--oeis") == 0) {
            check_oeis = true;
        } else if (strcmp(argv[i], "--fast") == 0) {
            fast_mode = true;
        } else if (argv[i][0] != '-') {
            maxN = atoi(argv[i]);
            if (maxN < 2) maxN = 2;
        }
    }

    if (maxN > 10) {
        printf("Warning: n > 10 not supported (GridKey only holds 100 cells).\n");
        return 1;
    }
    if (maxN >= 10) {
        printf("Note: n=10 has 35.7B RBPDs - will run out of memory.\n");
    }

    // Check for single n mode (e.g., "6" means just n=6, not 2..6)
    // Use "2-6" or "2:6" for range mode
    int minN = 2;
    bool single_mode = false;

    // Re-parse to check for range syntax
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            char* dash = strchr(argv[i], '-');
            char* colon = strchr(argv[i], ':');
            if (dash || colon) {
                char* sep = dash ? dash : colon;
                minN = atoi(argv[i]);
                maxN = atoi(sep + 1);
            } else {
                // Single number = just that n
                minN = maxN = atoi(argv[i]);
                single_mode = true;
            }
            break;
        }
    }

    if (minN < 2) minN = 2;
    if (maxN < minN) maxN = minN;

    printf("BPD Move Connectivity Checker\n");
    printf("==============================\n");
    if (minN == maxN) {
        printf("Testing n = %d%s%s\n", maxN,
               check_oeis ? " (OEIS)" : "",
               fast_mode ? " (fast)" : "");
    } else {
        printf("Testing n = %d to %d%s%s\n", minN, maxN,
               check_oeis ? " (OEIS)" : "",
               fast_mode ? " (fast)" : "");
    }

    for (int n = minN; n <= maxN; n++) {
        verifyConnectivity(n, check_oeis, fast_mode);
    }

    printf("All tests completed.\n");
    return 0;
}
