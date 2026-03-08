// BPD MCMC Sampler - Single-chain Metropolis for reduced bumpless pipe dreams
// Approximate uniform sampling via local flips + LLS droops with reducedness rejection.
//
// Usage:
//   ./bpd_mcmc <n> [options]
//   ./bpd_mcmc batch:<n>:<B> [options]
//
// Options:
//   --seed <S>       Random seed (default: random)
//   --burnin <K>     Burn-in steps; supports K/M/G/T suffixes (default: 10*n^5)
//   --thin <T>       Thinning interval; supports K/M/G/T suffixes (default: 10*n^4)
//   --droop <P>      Smart droop move probability 0.0-1.0 (default: 0.25)
//   --droop-dist <D> Rectangle size dist: geometric, uniform, loguniform, revlog (default: geometric)
//   --geom-p <P>     Geometric side-length parameter p (default: 0.5, mean side length 2)
//   --geom-mean <M>  Geometric mean side length 1/p (default: 2)
//   --anchor <A>     Droop anchor corner: nw or se (default: se)
//   --start <S>      Starting state: identity or w0 (default: identity)
//   --collect-droop-dist <D>  Droop dist for collection phase (default: same as --droop-dist)
//   --no-png         Suppress PNG output
//   --no-tikz        Suppress TikZ output
//   --no-height      Suppress height function output
//   -h, --help       Show help
//
// Compilation (single-threaded, macOS Apple Silicon):
//   clang++ -O3 -std=c++17 -mcpu=apple-m2 -flto \
//     -I/opt/homebrew/opt/libpng/include/libpng16 -L/opt/homebrew/opt/libpng/lib \
//     bpd_mcmc.cpp -o bpd_mcmc -lpng16
//
// Compilation (multi-threaded batch mode, macOS Apple Silicon):
//   clang++ -O3 -std=c++17 -mcpu=apple-m2 -Xclang -fopenmp \
//     -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp \
//     -I/opt/homebrew/opt/libpng/include/libpng16 -L/opt/homebrew/opt/libpng/lib \
//     bpd_mcmc.cpp -o bpd_mcmc -lpng16
//
// Compilation (multi-threaded, Linux):
//   g++ -O3 -std=c++17 -march=native -flto -fopenmp bpd_mcmc.cpp -o bpd_mcmc -lpng
//
// Note: -flto gives ~5% speedup. Other flags (-ffast-math, -funroll-loops, -DNDEBUG,
// -Ofast) have negligible effect. Do NOT use -flto with OpenMP on macOS (slow compile).

#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <string>
#include <ctime>
#include <atomic>
#include <csignal>

// Graceful stop: Ctrl+C during collection finishes early and outputs what's collected
static volatile sig_atomic_t g_stop_requested = 0;
static void handle_sigint(int) { g_stop_requested = 1; }

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int) {}
inline double omp_get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

// PNG support
#ifdef __has_include
#if __has_include(<png.h>)
#define HAS_PNG 1
#include <png.h>
#endif
#endif
#ifndef HAS_PNG
#define HAS_PNG 0
#endif

// Tile types: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz
static constexpr int MAX_N = 128;

// Global timestamp prefix for output filenames (set once in main)
static char g_ts[20] = "";  // "YYYYMMDD_HHMMSS_"
static char g_suffix[64] = "";  // "_dist_anchor_start" suffix for output files

// ========================================================================
// Run logger: appends command line and output files to bpd_mcmc_runs.log
// Buffers everything in memory, writes atomically on close to avoid
// interleaving when multiple processes run in parallel.
// ========================================================================
static std::string g_runlog_buf;

void runlog_open(int argc, char** argv) {
    time_t now = time(nullptr);
    char tbuf[32];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    g_runlog_buf += "\n=== ";
    g_runlog_buf += tbuf;
    g_runlog_buf += " ===\ncmd:";
    for (int i = 0; i < argc; i++) { g_runlog_buf += ' '; g_runlog_buf += argv[i]; }
    g_runlog_buf += '\n';
}

void runlog_file(const char* filename) {
    g_runlog_buf += "  -> ";
    g_runlog_buf += filename;
    g_runlog_buf += '\n';
}

void runlog_close() {
    if (g_runlog_buf.empty()) return;
    FILE* fp = fopen("bpd_mcmc_runs.log", "a");
    if (!fp) return;
    fwrite(g_runlog_buf.data(), 1, g_runlog_buf.size(), fp);
    fclose(fp);
    g_runlog_buf.clear();
}

// ========================================================================
// xoshiro256++ fast PRNG (replaces std::mt19937)
// ========================================================================
struct Xoshiro256pp {
    uint64_t s[4];

    static inline uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    void seed(uint64_t seed) {
        // SplitMix64 to initialize state from single seed
        for (int i = 0; i < 4; i++) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    inline uint64_t next() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Lemire's fast bounded random: avoids division in the common case.
    // Returns uniform in [0, range).
    inline uint32_t bounded(uint32_t range) {
        uint64_t m = (uint64_t)(uint32_t)next() * (uint64_t)range;
        uint32_t l = (uint32_t)m;
        if (l < range) {
            uint32_t t = (-range) % range;  // threshold
            while (l < t) {
                m = (uint64_t)(uint32_t)next() * (uint64_t)range;
                l = (uint32_t)m;
            }
        }
        return (uint32_t)(m >> 32);
    }

    // Fast coin flip
    inline bool coin() { return next() & 1; }
};

// Count inversions in a permutation (1-indexed values). O(n^2).
int countInversions(const int* perm, int n) {
    int count = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (perm[i] > perm[j]) count++;
    return count;
}

// =============================================================================
// BPD state and moves
// =============================================================================

class BPD {
    int n;
    std::vector<uint8_t> grid;
    int crossCount_;

public:
    BPD(int size) : n(size), grid(size * size, 0), crossCount_(0) {}
    BPD(const BPD&) = default;
    BPD& operator=(const BPD&) = default;

    int size() const { return n; }
    const std::vector<uint8_t>& data() const { return grid; }
    std::vector<uint8_t>& data_mut() { return grid; }
    int crossCount() const { return crossCount_; }
    void setCrossCount(int c) { crossCount_ = c; }

    inline uint8_t get(int r, int c) const { return grid[r * n + c]; }
    inline void set(int r, int c, uint8_t val) {
        uint8_t old = grid[r * n + c];
        if (old == 1) crossCount_--;
        if (val == 1) crossCount_++;
        grid[r * n + c] = val;
    }

    // Generate Rothe BPD for a permutation w (1-indexed values)
    void initRothe(const std::vector<int>& w) {
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

    // Get simple transposition index from cross at position (i,j)
    int get_sk(int i, int j) const {
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

    // Check if BPD is reduced using anti-diagonal word (original, slow method)
    bool isReducedSlow() const {
        int ww[MAX_N];
        for (int i = 0; i < n; i++) ww[i] = i + 1;
        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1, j = n + s - t - 1;
                int k = get_sk(i, j);
                if (k != -1) {
                    if (ww[k - 1] > ww[k]) return false;
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1, j = n - t - 1;
                int k = get_sk(i, j);
                if (k != -1) {
                    if (ww[k - 1] > ww[k]) return false;
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        return true;
    }

    // =========================================================================
    // Local 2x2 moves (drip/undrip/cross_drip/cross_undrip)
    // =========================================================================

    // Drip: r-elbow at NW, blank at SE → blank at NW, j-elbow at SE
    bool tryDrip(int i, int j) {
        uint8_t nw = get(i-1,j-1), ne = get(i-1,j), sw = get(i,j-1), se = get(i,j);
        if (nw == 2 && se == 0) {
            set(i-1,j-1, 0);  set(i,j, 3);
            set(i-1,j, (ne==5)?2:4);  set(i,j-1, (sw==4)?2:5);
            return true;
        }
        return false;
    }

    // Undrip: reverse of drip
    bool tryUndrip(int i, int j) {
        uint8_t nw = get(i-1,j-1), ne = get(i-1,j), sw = get(i,j-1), se = get(i,j);
        if (nw == 0 && se == 3 && (ne==2||ne==4) && (sw==2||sw==5)) {
            set(i-1,j-1, 2);  set(i,j, 0);
            set(i-1,j, (ne==2)?5:3);  set(i,j-1, (sw==2)?4:3);
            return true;
        }
        return false;
    }

    // Cross drip: creates/moves crosses
    bool tryCrossDrip(int i, int j) {
        uint8_t nw = get(i-1,j-1), ne = get(i-1,j), sw = get(i,j-1), se = get(i,j);
        if (nw == 2 && se == 2) {
            set(i-1,j-1, 0); set(i,j, 1);
            set(i-1,j, (ne==3)?4:2); set(i,j-1, (sw==3)?5:2);
            return true;
        }
        if (nw == 1 && se == 0) {
            set(i-1,j-1, 3); set(i,j, 3);
            set(i-1,j, (ne==3)?4:2); set(i,j-1, (sw==3)?5:2);
            return true;
        }
        if (nw == 1 && se == 2) {
            set(i-1,j-1, 3); set(i,j, 1);
            set(i-1,j, (ne==3)?4:2); set(i,j-1, (sw==3)?5:2);
            return true;
        }
        return false;
    }

    // Cross undrip: reverse of cross drip
    bool tryCrossUndrip(int i, int j) {
        uint8_t nw = get(i-1,j-1), ne = get(i-1,j), sw = get(i,j-1), se = get(i,j);
        if (nw == 0 && se == 1) {
            set(i-1,j-1, 2); set(i,j, 2);
            set(i-1,j, (ne==4)?3:5); set(i,j-1, (sw==5)?3:4);
            return true;
        }
        if (nw == 3 && se == 3) {
            set(i-1,j-1, 1); set(i,j, 0);
            set(i-1,j, (ne==4)?3:5); set(i,j-1, (sw==5)?3:4);
            return true;
        }
        if (nw == 3 && se == 1) {
            set(i-1,j-1, 1); set(i,j, 2);
            set(i-1,j, (ne==4)?3:5); set(i,j-1, (sw==5)?3:4);
            return true;
        }
        return false;
    }

    bool tryUp(int i, int j) {
        if (tryDrip(i,j)) return true;
        return tryCrossDrip(i,j);
    }

    bool tryDown(int i, int j) {
        if (tryUndrip(i,j)) return true;
        return tryCrossUndrip(i,j);
    }

    // =========================================================================
    // Non-local rectangular moves
    // =========================================================================

    bool canRectDroop(int i1, int j1, int i2, int j2) const {
        if (i2 < i1+1 || j2 < j1+1) return false;
        if (i1 < 0 || j1 < 0 || i2 >= n || j2 >= n) return false;
        if (get(i1,j1) != 2 || get(i2,j2) != 0) return false;
        // Check corners: NE must be horiz/j-elbow, SW must be vert/j-elbow
        uint8_t ne = get(i1,j2), sw = get(i2,j1);
        if (ne != 5 && ne != 3) return false;
        if (sw != 4 && sw != 3) return false;
        // Edge checks: match exactly the tiles applyRectDroop handles
        for (int j = j1+1; j < j2; j++) {
            uint8_t t1 = get(i1,j); if (t1 != 5 && t1 != 1) return false;
            uint8_t t2 = get(i2,j); if (t2 != 0 && t2 != 4) return false;
        }
        for (int i = i1+1; i < i2; i++) {
            uint8_t t1 = get(i,j1); if (t1 != 4 && t1 != 1) return false;
            uint8_t t2 = get(i,j2); if (t2 != 0 && t2 != 5) return false;
        }
        for (int i = i1+1; i < i2; i++)
            for (int j = j1+1; j < j2; j++)
                if (get(i,j)==2 || get(i,j)==3) return false;
        return true;
    }

    void applyRectDroop(int i1, int j1, int i2, int j2) {
        set(i1,j1, 0);  set(i2,j2, 3);
        uint8_t ne = get(i1,j2), sw = get(i2,j1);
        if (ne==5) set(i1,j2, 2); else if (ne==3) set(i1,j2, 4);
        if (sw==4) set(i2,j1, 2); else if (sw==3) set(i2,j1, 5);
        for (int i = i1+1; i < i2; i++) {
            uint8_t t = get(i,j1);
            if (t==4) set(i,j1, 0); else if (t==1) set(i,j1, 5);
        }
        for (int j = j1+1; j < j2; j++) {
            uint8_t t = get(i1,j);
            if (t==5) set(i1,j, 0); else if (t==1) set(i1,j, 4);
        }
        for (int i = i1+1; i < i2; i++) {
            uint8_t t = get(i,j2);
            if (t==0) set(i,j2, 4); else if (t==5) set(i,j2, 1);
        }
        for (int j = j1+1; j < j2; j++) {
            uint8_t t = get(i2,j);
            if (t==0) set(i2,j, 5); else if (t==4) set(i2,j, 1);
        }
    }

    bool canRectUndroop(int i1, int j1, int i2, int j2) const {
        if (i2 < i1+1 || j2 < j1+1) return false;
        if (i1 < 0 || j1 < 0 || i2 >= n || j2 >= n) return false;
        if (get(i1,j1) != 0 || get(i2,j2) != 3) return false;
        uint8_t ne = get(i1,j2), sw = get(i2,j1);
        if ((ne!=2&&ne!=4) || (sw!=2&&sw!=5)) return false;
        for (int j = j1+1; j < j2; j++) {
            uint8_t t = get(i1,j); if (t!=0&&t!=4) return false;
            t = get(i2,j); if (t!=5&&t!=1) return false;
        }
        for (int i = i1+1; i < i2; i++) {
            uint8_t t = get(i,j1); if (t!=0&&t!=5) return false;
            t = get(i,j2); if (t!=4&&t!=1) return false;
        }
        for (int i = i1+1; i < i2; i++)
            for (int j = j1+1; j < j2; j++)
                if (get(i,j)==2 || get(i,j)==3) return false;
        return true;
    }

    void applyRectUndroop(int i1, int j1, int i2, int j2) {
        set(i1,j1, 2);  set(i2,j2, 0);
        uint8_t ne = get(i1,j2), sw = get(i2,j1);
        if (ne==2) set(i1,j2, 5); else if (ne==4) set(i1,j2, 3);
        if (sw==2) set(i2,j1, 4); else if (sw==5) set(i2,j1, 3);
        for (int i = i1+1; i < i2; i++) {
            uint8_t t = get(i,j1);
            if (t==0) set(i,j1, 4); else if (t==5) set(i,j1, 1);
        }
        for (int j = j1+1; j < j2; j++) {
            uint8_t t = get(i1,j);
            if (t==0) set(i1,j, 5); else if (t==4) set(i1,j, 1);
        }
        for (int i = i1+1; i < i2; i++) {
            uint8_t t = get(i,j2);
            if (t==4) set(i,j2, 0); else if (t==1) set(i,j2, 5);
        }
        for (int j = j1+1; j < j2; j++) {
            uint8_t t = get(i2,j);
            if (t==5) set(i2,j, 0); else if (t==1) set(i2,j, 4);
        }
    }
};

// =============================================================================
// Utility
// =============================================================================

// Parse integer with optional K/M/G/T suffix (case-insensitive).
// E.g. "10G" = 10000000000, "500M" = 500000000, "2K" = 2000.
int64_t parseCount(const char* s) {
    char* end;
    double val = strtod(s, &end);
    if (end != s) {
        switch (*end) {
            case 'k': case 'K': val *= 1e3; break;
            case 'm': case 'M': val *= 1e6; break;
            case 'g': case 'G': val *= 1e9; break;
            case 't': case 'T': val *= 1e12; break;
        }
    }
    return (int64_t)val;
}

std::string fmtNum(int64_t n) {
    char buf[32];
    if (n >= 1000000000LL) snprintf(buf, sizeof(buf), "%.2fG", n / 1e9);
    else if (n >= 1000000) snprintf(buf, sizeof(buf), "%.2fM", n / 1e6);
    else if (n >= 1000) snprintf(buf, sizeof(buf), "%.2fK", n / 1e3);
    else snprintf(buf, sizeof(buf), "%lld", (long long)n);
    return std::string(buf);
}

std::string fmtETA(double secs) {
    char buf[64];
    int s = (int)secs;
    if (s >= 3600) snprintf(buf, sizeof(buf), "%dh%02dm", s / 3600, (s % 3600) / 60);
    else if (s >= 60) snprintf(buf, sizeof(buf), "%dm%02ds", s / 60, s % 60);
    else snprintf(buf, sizeof(buf), "%ds", s);
    return std::string(buf);
}

// =============================================================================
// MCMC Sampler
// =============================================================================

struct MCMCStats {
    int64_t proposals = 0;
    int64_t accepts = 0;
    int64_t local_proposals = 0;
    int64_t local_accepts = 0;
    int64_t droop_proposals = 0;
    int64_t droop_accepts = 0;
    int64_t droop_down_accepts = 0;   // droops (r-elbow→blank, moves pipes down/right)
    int64_t droop_up_accepts = 0;     // undroops (blank→j-elbow, moves pipes up/left)
    int64_t reducedness_rejects = 0;

    // Histogram of accepted droop/undroop sizes: index = max(di,dj)
    // (rectangle "radius" = max side length). Size MAX_N is enough.
    int64_t droop_size_hist[MAX_N] = {};    // droops by max(di,dj)
    int64_t undroop_size_hist[MAX_N] = {};  // undroops by max(di,dj)

    MCMCStats& operator+=(const MCMCStats& o) {
        proposals += o.proposals; accepts += o.accepts;
        local_proposals += o.local_proposals; local_accepts += o.local_accepts;
        droop_proposals += o.droop_proposals; droop_accepts += o.droop_accepts;
        droop_down_accepts += o.droop_down_accepts; droop_up_accepts += o.droop_up_accepts;
        reducedness_rejects += o.reducedness_rejects;
        for (int i = 0; i < MAX_N; i++) {
            droop_size_hist[i] += o.droop_size_hist[i];
            undroop_size_hist[i] += o.undroop_size_hist[i];
        }
        return *this;
    }
};

// BPDMcmc: MCMC sampler on reduced bumpless pipe dreams.
//
// Key algorithmic optimization: O(n) incremental reducedness checking.
//
// A BPD is reduced iff crossCount == inversions(perm(BPD)). Rather than
// recomputing the permutation from scratch after each move (O(n²) pipe
// tracing + O(n²) inversion count), we maintain:
//   1. Edge colors (pipe identities) at every cell boundary
//   2. The permutation read from right-boundary colors
//   3. The inversion count of that permutation
//
// After a local 2×2 move, we propagate edge-color changes from the
// modified cells outward (O(n) per affected pipe), update the permutation
// from boundary colors (O(1)), and update the inversion count (O(n) scan).
// The reducedness check becomes crossCount == invCount, which is O(1).
//
// Combined with xoshiro256++ RNG and zero-allocation move/undo, this
// achieves ~28M steps/sec at n=36 (vs ~4.5M pre-optimization baseline).

enum class DroopDist { GEOMETRIC, UNIFORM, LOGUNIFORM, REVLOGUNIFORM };
enum class DroopAnchor { NW, SE };

class BPDMcmc {
    BPD state;
    Xoshiro256pp rng;
    MCMCStats stats;
    int n_;
    uint32_t droop_thresh_;    // precomputed threshold for smart droop move (droop_prob * 1000)
    DroopDist droop_dist_;     // rectangle size distribution for droop proposals
    DroopAnchor droop_anchor_; // which corner to pick uniformly (NW or SE)
    double geom_p_;            // side-length parameter for geometric rectangle proposals
    double geom_log1m_p_;      // cached log(1-p) for inverse-CDF sampling
    bool geom_is_half_;        // fast path for the legacy p=1/2 geometric proposal

    // === Incremental edge-color tracking ===
    // h_edges_[row*n_+col] = pipe color on top edge of cell (row,col)
    // h_edges_[n_*n_+col] = bottom boundary (pipe col+1 enters)
    // Size: (n_+1)*n_
    std::vector<int> h_edges_;
    // v_edges_[row*(n_+1)+col] = pipe color on left edge of cell (row,col)
    // v_edges_[row*(n_+1)+n_] = right boundary (pipe exits at row)
    // Size: n_*(n_+1)
    std::vector<int> v_edges_;

    // Maintained permutation and inversion count
    int perm_[MAX_N];
    int invCount_;

    // Undo buffer for edge color saves (pointers into h_edges_/v_edges_)
    // Worst case: 4 cells * 2 edges + 4 propagation chains * 2*MAX_N = 8 + 8*MAX_N
    // (each propagation can traverse up to 2n cells: n upward + n rightward)
    static constexpr int MAX_EDGE_SAVES = 8 + 8 * MAX_N;
    int* edgeSavePtrs_[MAX_EDGE_SAVES];
    int edgeSaveVals_[MAX_EDGE_SAVES];
    int nEdgeSaves_;

    // Undo buffer for perm changes
    static constexpr int MAX_PERM_SAVES = 8;
    int permSaveIdx_[MAX_PERM_SAVES];
    int permSaveVal_[MAX_PERM_SAVES];
    int nPermSaves_;
    bool permSaveOverflow_;
    int savedInvCount_;

    // --- Private helper methods ---

    void computeFullEdgeColors() {
        std::fill(h_edges_.begin(), h_edges_.end(), 0);
        std::fill(v_edges_.begin(), v_edges_.end(), 0);
        for (int c = 0; c < n_; c++) h_edges_[n_*n_+c] = c+1;
        for (int r = n_-1; r >= 0; r--) {
            for (int c = 0; c < n_; c++) {
                uint8_t tile = state.get(r, c);
                int lc = v_edges_[r*(n_+1)+c];
                int bc = h_edges_[(r+1)*n_+c];
                int tc = 0, rc = 0;
                switch (tile) {
                    case 0: break;
                    case 1: rc=lc; tc=bc; break;
                    case 2: rc=bc; break;
                    case 3: tc=lc; break;
                    case 4: tc=bc; break;
                    case 5: rc=lc; break;
                }
                h_edges_[r*n_+c] = tc;
                v_edges_[r*(n_+1)+c+1] = rc;
            }
        }
    }

    void rebuildPermAndInv() {
        std::memset(perm_, 0, sizeof(int) * n_);
        for (int r = 0; r < n_; r++) {
            int pid = v_edges_[r*(n_+1)+n_];
            if (pid > 0) perm_[pid-1] = r+1;
        }
        invCount_ = countInversions(perm_, n_);
    }

    inline void saveEdge(int* ptr, int old_val) {
        assert(nEdgeSaves_ < MAX_EDGE_SAVES);
        edgeSavePtrs_[nEdgeSaves_] = ptr;
        edgeSaveVals_[nEdgeSaves_] = old_val;
        nEdgeSaves_++;
    }

    // Recompute output edges of a single cell from current tile and input edges.
    inline void recomputeCellEdges(int r, int c) {
        uint8_t tile = state.get(r, c);
        int lc = v_edges_[r*(n_+1)+c];
        int bc = h_edges_[(r+1)*n_+c];
        int tc = 0, rc = 0;
        switch (tile) {
            case 0: break;
            case 1: rc=lc; tc=bc; break;
            case 2: rc=bc; break;
            case 3: tc=lc; break;
            case 4: tc=bc; break;
            case 5: rc=lc; break;
        }
        int* tc_ptr = &h_edges_[r*n_+c];
        int* rc_ptr = &v_edges_[r*(n_+1)+c+1];
        if (tc != *tc_ptr) { saveEdge(tc_ptr, *tc_ptr); *tc_ptr = tc; }
        if (rc != *rc_ptr) { saveEdge(rc_ptr, *rc_ptr); *rc_ptr = rc; }
    }

    // Propagate a pipe color change through the grid.
    // Starts at cell (r,c) entering from below (from_below=true) or left (false).
    // The input edge has already been set to the new value by a prior step.
    // Follows the pipe path, saving and updating each output edge until
    // the change is absorbed (output matches stored) or we hit a boundary.
    // If save=true, records old values for undo (local moves).
    // If save=false, just overwrites (rect moves, no undo needed).
    void propagatePipe(int r, int c, bool from_below, bool save = true) {
        while (r >= 0 && c < n_) {
            uint8_t tile = state.get(r, c);
            if (from_below) {
                int nv = h_edges_[(r+1)*n_+c];  // new bottom input
                switch (tile) {
                    case 0: case 3: case 5: return;  // bottom not used
                    case 1: case 4: {  // cross/vert: bottom->top
                        int* p = &h_edges_[r*n_+c];
                        if (nv == *p) return;
                        if (save) saveEdge(p, *p);
                        *p = nv;
                        r--;
                        break;
                    }
                    case 2: {  // r-elbow: bottom->right
                        int* p = &v_edges_[r*(n_+1)+c+1];
                        if (nv == *p) return;
                        if (save) saveEdge(p, *p);
                        *p = nv;
                        from_below = false; c++;
                        break;
                    }
                }
            } else {
                int nv = v_edges_[r*(n_+1)+c];  // new left input
                switch (tile) {
                    case 0: case 2: case 4: return;  // left not used
                    case 1: case 5: {  // cross/horiz: left->right
                        int* p = &v_edges_[r*(n_+1)+c+1];
                        if (nv == *p) return;
                        if (save) saveEdge(p, *p);
                        *p = nv;
                        c++;
                        break;
                    }
                    case 3: {  // j-elbow: left->top
                        int* p = &h_edges_[r*n_+c];
                        if (nv == *p) return;
                        if (save) saveEdge(p, *p);
                        *p = nv;
                        from_below = true; r--;
                        break;
                    }
                }
            }
        }
    }

    // Update edge colors incrementally after a local 2x2 move at position (i,j).
    // Block cells: (i-1,j-1), (i-1,j), (i,j-1), (i,j).
    void updateEdgesLocal(int i, int j) {
        // Save old outgoing edge values
        int ot_tl = h_edges_[(i-1)*n_+(j-1)];
        int ot_tr = h_edges_[(i-1)*n_+j];
        int or_br = v_edges_[i*(n_+1)+j+1];
        int or_tr = v_edges_[(i-1)*(n_+1)+j+1];

        // Recompute 4 cells in bottom-to-top, left-to-right order
        recomputeCellEdges(i, j-1);
        recomputeCellEdges(i, j);
        recomputeCellEdges(i-1, j-1);
        recomputeCellEdges(i-1, j);

        // Propagate changed outgoing edges beyond the block
        // Process bottom row first (higher row index = lower in grid)
        if (v_edges_[i*(n_+1)+j+1] != or_br && j+1 < n_)
            propagatePipe(i, j+1, false);
        if (v_edges_[(i-1)*(n_+1)+j+1] != or_tr && j+1 < n_)
            propagatePipe(i-1, j+1, false);
        if (h_edges_[(i-1)*n_+(j-1)] != ot_tl && i >= 2)
            propagatePipe(i-2, j-1, true);
        if (h_edges_[(i-1)*n_+j] != ot_tr && i >= 2)
            propagatePipe(i-2, j, true);

        // Update perm and invCount from right boundary edges
        updatePermFromEdges();
    }

    // Incremental edge color update after a rectangular droop/undroop move.
    // Rectangle: rows [r1..r2], cols [c1..c2] (inclusive).
    // 1. Save old top and right boundary edges of the rectangle.
    // 2. Recompute edges inside the rectangle (bottom-to-top, left-to-right).
    // 3. Propagate changed top boundary edges upward (NE).
    // 4. Propagate changed right boundary edges rightward (NE).
    // 5. Update perm from right boundary of grid.
    // Cost: O(k²) + O(k·n) instead of O(n²), where k = max(di, dj).
    void updateEdgesRect(int r1, int c1, int r2, int c2) {
        nEdgeSaves_ = 0;

        // Save old outgoing edges: top row (h_edges) and right col (v_edges)
        int top_width = c2 - c1 + 1;
        int right_height = r2 - r1 + 1;
        // Use stack arrays (max rectangle is n×n)
        int old_top[MAX_N], old_right[MAX_N];

        // Top boundary: h_edges_[r1*n_ + c] for c in [c1..c2]
        for (int c = c1; c <= c2; c++)
            old_top[c - c1] = h_edges_[r1 * n_ + c];
        // Right boundary: v_edges_[r*(n_+1) + (c2+1)] for r in [r1..r2]
        for (int r = r1; r <= r2; r++)
            old_right[r - r1] = v_edges_[r * (n_ + 1) + c2 + 1];

        // Recompute edges inside rectangle (bottom-to-top, left-to-right)
        for (int r = r2; r >= r1; r--) {
            for (int c = c1; c <= c2; c++) {
                uint8_t tile = state.get(r, c);
                int lc = v_edges_[r * (n_ + 1) + c];
                int bc = h_edges_[(r + 1) * n_ + c];
                int tc = 0, rc = 0;
                switch (tile) {
                    case 0: break;
                    case 1: rc = lc; tc = bc; break;
                    case 2: rc = bc; break;
                    case 3: tc = lc; break;
                    case 4: tc = bc; break;
                    case 5: rc = lc; break;
                }
                h_edges_[r * n_ + c] = tc;
                v_edges_[r * (n_ + 1) + c + 1] = rc;
            }
        }

        // Propagate changed top boundary edges upward (no save — rect moves are committed)
        if (r1 >= 1) {
            for (int c = c1; c <= c2; c++) {
                if (h_edges_[r1 * n_ + c] != old_top[c - c1])
                    propagatePipe(r1 - 1, c, true, false);
            }
        }

        // Propagate changed right boundary edges rightward
        if (c2 + 1 < n_) {
            for (int r = r1; r <= r2; r++) {
                if (v_edges_[r * (n_ + 1) + c2 + 1] != old_right[r - r1])
                    propagatePipe(r, c2 + 1, false, false);
            }
        }

        // Update perm and invCount from right boundary
        updatePermFromEdges();
    }

    // Scan right boundary edges, detect perm changes, update invCount.
    void updatePermFromEdges() {
        nPermSaves_ = 0;
        permSaveOverflow_ = false;
        savedInvCount_ = invCount_;
        for (int r = 0; r < n_; r++) {
            int pid = v_edges_[r*(n_+1)+n_];
            if (pid > 0 && perm_[pid-1] != r+1) {
                if (nPermSaves_ < MAX_PERM_SAVES) {
                    permSaveIdx_[nPermSaves_] = pid - 1;
                    permSaveVal_[nPermSaves_] = perm_[pid-1];
                    nPermSaves_++;
                } else {
                    permSaveOverflow_ = true;
                }
                perm_[pid-1] = r + 1;
            }
        }
        if (nPermSaves_ == 2 && !permSaveOverflow_) {
            computeInvDelta2();
        } else if (nPermSaves_ != 0) {
            invCount_ = countInversions(perm_, n_);
        }
    }

    // O(n) inversion count update when exactly 2 perm entries changed (swap).
    void computeInvDelta2() {
        int a = permSaveIdx_[0], b = permSaveIdx_[1];
        int oa = permSaveVal_[0], ob = permSaveVal_[1];
        int na = perm_[a], nb = perm_[b];
        int delta = 0;
        // Pair (a,b):
        if (a < b) { if (oa > ob) delta--; if (na > nb) delta++; }
        else       { if (ob > oa) delta--; if (nb > na) delta++; }
        // Pairs with other elements:
        for (int k = 0; k < n_; k++) {
            if (k == a || k == b) continue;
            int pk = perm_[k];
            if (k < a) { if (pk > oa) delta--; if (pk > na) delta++; }
            else       { if (oa > pk) delta--; if (na > pk) delta++; }
            if (k < b) { if (pk > ob) delta--; if (pk > nb) delta++; }
            else       { if (ob > pk) delta--; if (nb > pk) delta++; }
        }
        invCount_ += delta;
    }

    void undoEdgeChanges() {
        for (int i = nEdgeSaves_ - 1; i >= 0; i--)
            *edgeSavePtrs_[i] = edgeSaveVals_[i];
    }

    void undoPermChanges() {
        if (permSaveOverflow_) {
            // Save buffer overflowed; edges are already restored, so rebuild
            rebuildPermAndInv();
        } else {
            for (int i = nPermSaves_ - 1; i >= 0; i--)
                perm_[permSaveIdx_[i]] = permSaveVal_[i];
            invCount_ = savedInvCount_;
        }
    }

    inline int sampleGeometricOffset(int raw_max) {
        if (raw_max <= 1) return 1;
        if (geom_is_half_) {
            uint64_t bits = rng.next();
            return 1 + std::min(__builtin_ctzll(bits | (1ULL << 63)), raw_max - 1);
        }
        if (geom_p_ >= 1.0) return 1;
        double u = (rng.next() >> 11) * 0x1.0p-53;
        int k = 1 + (int)(std::log1p(-u) / geom_log1m_p_);
        return std::min(k, raw_max);
    }

public:
    BPDMcmc(int n, double droop_p = 0.25, DroopDist dd = DroopDist::GEOMETRIC,
            DroopAnchor da = DroopAnchor::SE, double geom_p = 0.5)
        : state(n), n_(n),
          droop_thresh_((uint32_t)(droop_p * 1000)),
          droop_dist_(dd), droop_anchor_(da),
          geom_p_(geom_p), geom_log1m_p_(0.0), geom_is_half_(false),
          h_edges_((n+1)*n, 0), v_edges_(n*(n+1), 0),
          invCount_(0),
          nEdgeSaves_(0), nPermSaves_(0), permSaveOverflow_(false), savedInvCount_(0) {
        std::memset(perm_, 0, sizeof(perm_));
        setGeometricP(geom_p);
    }

    void seed(uint64_t s) { rng.seed(s); }
    void setDroopDist(DroopDist dd) { droop_dist_ = dd; }
    void setGeometricP(double p) {
        geom_p_ = p;
        geom_log1m_p_ = (p < 1.0) ? std::log1p(-p) : -1e30;
        geom_is_half_ = std::fabs(p - 0.5) < 1e-15;
    }
    int n() const { return n_; }
    const BPD& current() const { return state; }
    const MCMCStats& getStats() const { return stats; }
    void resetStats() { stats = MCMCStats{}; }
    const int* getPerm() const { return perm_; }
    int getInvCount() const { return invCount_; }
    const std::vector<int>& hEdges() const { return h_edges_; }
    const std::vector<int>& vEdges() const { return v_edges_; }

    void computeHeight(std::vector<int>& height) const {
        int m = n_ + 1;
        height.resize(m * m);
        for (int r = 0; r <= n_; r++) {
            height[r * m] = 0;
            for (int c = 1; c <= n_; c++)
                height[r * m + c] = height[r * m + c - 1] +
                    (h_edges_[r * n_ + (c - 1)] != 0 ? 1 : 0);
        }
    }

    // Save full MCMC state to binary file (grid + RNG + edges + perm + invCount)
    bool saveCheckpoint(const char* filename) const {
        FILE* fp = fopen(filename, "wb");
        if (!fp) { fprintf(stderr, "Cannot open %s for writing\n", filename); return false; }
        bool ok = true;
        // Header: n
        ok = ok && fwrite(&n_, sizeof(int), 1, fp) == 1;
        // BPD grid
        ok = ok && fwrite(state.data().data(), sizeof(uint8_t), n_ * n_, fp) == (size_t)(n_ * n_);
        int cc = state.crossCount();
        ok = ok && fwrite(&cc, sizeof(int), 1, fp) == 1;
        // RNG state
        ok = ok && fwrite(&rng.s, sizeof(uint64_t), 4, fp) == 4;
        // Edge colors
        ok = ok && fwrite(h_edges_.data(), sizeof(int), h_edges_.size(), fp) == h_edges_.size();
        ok = ok && fwrite(v_edges_.data(), sizeof(int), v_edges_.size(), fp) == v_edges_.size();
        // Perm + invCount
        ok = ok && fwrite(perm_, sizeof(int), n_, fp) == (size_t)n_;
        ok = ok && fwrite(&invCount_, sizeof(int), 1, fp) == 1;
        fclose(fp);
        if (!ok) {
            fprintf(stderr, "Checkpoint write failed (disk full?): %s\n", filename);
            return false;
        }
        printf("Checkpoint saved to: %s\n", filename); runlog_file(filename);
        return true;
    }

    // Restore full MCMC state from binary checkpoint
    bool loadCheckpoint(const char* filename) {
        FILE* fp = fopen(filename, "rb");
        if (!fp) { fprintf(stderr, "Cannot open %s for reading\n", filename); return false; }
        int saved_n = 0;
        bool ok = true;
        ok = ok && fread(&saved_n, sizeof(int), 1, fp) == 1;
        if (!ok) {
            fprintf(stderr, "Failed to read checkpoint header from %s\n", filename);
            fclose(fp); return false;
        }
        if (saved_n != n_) {
            fprintf(stderr, "Checkpoint n=%d doesn't match current n=%d\n", saved_n, n_);
            fclose(fp); return false;
        }
        // BPD grid
        ok = ok && fread(state.data_mut().data(), sizeof(uint8_t), n_ * n_, fp) == (size_t)(n_ * n_);
        int cc;
        ok = ok && fread(&cc, sizeof(int), 1, fp) == 1;
        if (ok) state.setCrossCount(cc);
        // RNG state
        ok = ok && fread(&rng.s, sizeof(uint64_t), 4, fp) == 4;
        // Edge colors
        ok = ok && fread(h_edges_.data(), sizeof(int), h_edges_.size(), fp) == h_edges_.size();
        ok = ok && fread(v_edges_.data(), sizeof(int), v_edges_.size(), fp) == v_edges_.size();
        // Perm + invCount
        ok = ok && fread(perm_, sizeof(int), n_, fp) == (size_t)n_;
        ok = ok && fread(&invCount_, sizeof(int), 1, fp) == 1;
        fclose(fp);
        if (!ok) {
            fprintf(stderr, "Checkpoint file truncated or corrupted: %s\n", filename);
            return false;
        }
        printf("Checkpoint restored from: %s (ell=%d)\n", filename, state.crossCount());
        return true;
    }

    // Save multiple chains to a single checkpoint file
    static bool saveMultiCheckpoint(const char* filename,
                                    const std::vector<std::unique_ptr<BPDMcmc>>& chains) {
        FILE* fp = fopen(filename, "wb");
        if (!fp) { fprintf(stderr, "Cannot open %s for writing\n", filename); return false; }
        bool ok = true;
        int T = (int)chains.size();
        // Magic + version
        uint32_t magic = 0x4D434D43; // "MCMC"
        ok = ok && fwrite(&magic, sizeof(uint32_t), 1, fp) == 1;
        // Number of chains
        ok = ok && fwrite(&T, sizeof(int), 1, fp) == 1;
        // Each chain's state
        for (int t = 0; t < T && ok; t++) {
            const auto& mc = *chains[t];
            ok = ok && fwrite(&mc.n_, sizeof(int), 1, fp) == 1;
            ok = ok && fwrite(mc.state.data().data(), sizeof(uint8_t), mc.n_ * mc.n_, fp) == (size_t)(mc.n_ * mc.n_);
            int cc = mc.state.crossCount();
            ok = ok && fwrite(&cc, sizeof(int), 1, fp) == 1;
            ok = ok && fwrite(&mc.rng.s, sizeof(uint64_t), 4, fp) == 4;
            ok = ok && fwrite(mc.h_edges_.data(), sizeof(int), mc.h_edges_.size(), fp) == mc.h_edges_.size();
            ok = ok && fwrite(mc.v_edges_.data(), sizeof(int), mc.v_edges_.size(), fp) == mc.v_edges_.size();
            ok = ok && fwrite(mc.perm_, sizeof(int), mc.n_, fp) == (size_t)mc.n_;
            ok = ok && fwrite(&mc.invCount_, sizeof(int), 1, fp) == 1;
        }
        fclose(fp);
        if (!ok) {
            fprintf(stderr, "Multi-checkpoint write failed: %s\n", filename);
            return false;
        }
        printf("Multi-chain checkpoint saved (%d chains) to: %s\n", T, filename);
        return true;
    }

    // Load multiple chains from a multi-chain checkpoint file
    // Returns number of chains loaded (0 on failure)
    static int loadMultiCheckpoint(const char* filename, int n,
                                   double droopProb, DroopDist dd, double geomP,
                                   std::vector<std::unique_ptr<BPDMcmc>>& chains,
                                   DroopAnchor da = DroopAnchor::SE) {
        FILE* fp = fopen(filename, "rb");
        if (!fp) { fprintf(stderr, "Cannot open %s for reading\n", filename); return 0; }
        bool ok = true;

        // Check for magic header to distinguish multi vs legacy format
        uint32_t magic = 0;
        ok = ok && fread(&magic, sizeof(uint32_t), 1, fp) == 1;
        if (!ok) { fclose(fp); return 0; }

        int T = 0;
        if (magic == 0x4D434D43) {
            // Multi-chain format
            ok = ok && fread(&T, sizeof(int), 1, fp) == 1;
        } else {
            // Legacy single-chain format: magic was actually n
            // Rewind and load as single chain
            rewind(fp);
            T = 1;
        }

        if (T <= 0 || !ok) { fclose(fp); return 0; }

        chains.resize(T);
        for (int t = 0; t < T && ok; t++) {
            chains[t] = std::make_unique<BPDMcmc>(n, droopProb, dd, da, geomP);
            auto& mc = *chains[t];
            int saved_n = 0;
            ok = ok && fread(&saved_n, sizeof(int), 1, fp) == 1;
            if (ok && saved_n != n) {
                fprintf(stderr, "Checkpoint chain %d: n=%d doesn't match current n=%d\n", t, saved_n, n);
                fclose(fp); return 0;
            }
            ok = ok && fread(mc.state.data_mut().data(), sizeof(uint8_t), n * n, fp) == (size_t)(n * n);
            int cc;
            ok = ok && fread(&cc, sizeof(int), 1, fp) == 1;
            if (ok) mc.state.setCrossCount(cc);
            ok = ok && fread(&mc.rng.s, sizeof(uint64_t), 4, fp) == 4;
            ok = ok && fread(mc.h_edges_.data(), sizeof(int), mc.h_edges_.size(), fp) == mc.h_edges_.size();
            ok = ok && fread(mc.v_edges_.data(), sizeof(int), mc.v_edges_.size(), fp) == mc.v_edges_.size();
            ok = ok && fread(mc.perm_, sizeof(int), n, fp) == (size_t)n;
            ok = ok && fread(&mc.invCount_, sizeof(int), 1, fp) == 1;
        }
        fclose(fp);
        if (!ok) {
            fprintf(stderr, "Multi-checkpoint file truncated or corrupted: %s\n", filename);
            return 0;
        }
        printf("Restored %d chains from: %s (chain 0 ell=%d)\n", T, filename, chains[0]->current().crossCount());
        return T;
    }

    void init(bool startW0 = false) {
        if (startW0)
            state.initW0();
        else
            state.initIdentity();
        computeFullEdgeColors();
        rebuildPermAndInv();
    }

    // Smart LLS droop/undroop move via random rectangle proposal.
    // Pick a random cell and geometrically-distributed offset to form
    // a rectangle. Check if NW/SE corners form a valid droop
    // (r-elbow→blank) or undroop (blank→j-elbow). If valid, apply
    // and always accept — the proposal is symmetric (same NW corner
    // and same offset probability in forward and reverse directions),
    // so no MH correction is needed. Droops preserve permutation and
    // cross count, so reducedness is automatically maintained.
    // Geometric offset biases toward small rectangles where valid
    // droops are much more common, giving practical acceptance rates
    // even at large n. Cost: O(1) amortized.
    void smartDroopStep() {
        int nn = n_;
        stats.droop_proposals++;
        stats.proposals++;

        // Pick anchor corner uniformly, then draw offsets to get the other corner.
        // --anchor=nw: pick NW corner, offsets go down-right to SE
        // --anchor=se: pick SE corner, offsets go up-left to NW
        int anchor_r = rng.bounded(nn);
        int anchor_c = rng.bounded(nn);

        int raw_max_di, raw_max_dj;
        if (droop_anchor_ == DroopAnchor::NW) {
            raw_max_di = nn - 1 - anchor_r;
            raw_max_dj = nn - 1 - anchor_c;
        } else {
            raw_max_di = anchor_r;
            raw_max_dj = anchor_c;
        }
        if (raw_max_di < 1 || raw_max_dj < 1) return;

        int di, dj;
        switch (droop_dist_) {
        case DroopDist::GEOMETRIC: {
            di = sampleGeometricOffset(raw_max_di);
            dj = sampleGeometricOffset(raw_max_dj);
            break;
        }
        case DroopDist::UNIFORM: {
            di = 1 + rng.bounded(raw_max_di);
            dj = 1 + rng.bounded(raw_max_dj);
            break;
        }
        case DroopDist::LOGUNIFORM: {
            double u1 = (rng.next() >> 11) * 0x1.0p-53;
            double u2 = (rng.next() >> 11) * 0x1.0p-53;
            di = 1 + (int)(std::pow((double)raw_max_di, u1));
            dj = 1 + (int)(std::pow((double)raw_max_dj, u2));
            if (di > raw_max_di) di = raw_max_di;
            if (dj > raw_max_dj) dj = raw_max_dj;
            break;
        }
        case DroopDist::REVLOGUNIFORM: {
            double u1 = (rng.next() >> 11) * 0x1.0p-53;
            double u2 = (rng.next() >> 11) * 0x1.0p-53;
            int si = (int)(std::pow((double)raw_max_di, u1));
            int sj = (int)(std::pow((double)raw_max_dj, u2));
            di = raw_max_di - si;
            dj = raw_max_dj - sj;
            if (di < 1) di = 1;
            if (dj < 1) dj = 1;
            break;
        }
        }

        int i1, j1, i2, j2;
        if (droop_anchor_ == DroopAnchor::NW) {
            i1 = anchor_r; j1 = anchor_c;
            i2 = anchor_r + di; j2 = anchor_c + dj;
        } else {
            i2 = anchor_r; j2 = anchor_c;
            i1 = anchor_r - di; j1 = anchor_c - dj;
        }

        // Quick reject: NW must be r-elbow(2) or blank(0)
        uint8_t nw = state.get(i1, j1);
        if (nw != 2 && nw != 0) return;
        uint8_t se = state.get(i2, j2);

        bool isDroop;
        if (nw == 2 && se == 0) {
            // Candidate droop: r-elbow at NW, blank at SE
            if (!state.canRectDroop(i1, j1, i2, j2)) return;
            state.applyRectDroop(i1, j1, i2, j2);
            isDroop = true;
        } else if (nw == 0 && se == 3) {
            // Candidate undroop: blank at NW, j-elbow at SE
            if (!state.canRectUndroop(i1, j1, i2, j2)) return;
            state.applyRectUndroop(i1, j1, i2, j2);
            isDroop = false;
        } else {
            return;  // No valid move for this rectangle
        }

        // Incremental edge color update for the rectangle
        updateEdgesRect(i1, j1, i2, j2);

        stats.droop_accepts++;
        int maxside = std::max(di, dj);
        if (isDroop) { stats.droop_down_accepts++; stats.droop_size_hist[maxside]++; }
        else         { stats.droop_up_accepts++;    stats.undroop_size_hist[maxside]++; }
        stats.accepts++;
    }

    // One Metropolis step with incremental O(n) reducedness check.
    // Local moves: update edge colors incrementally, check crossCount==invCount O(1).
    // Rect moves: full recompute (rare, O(n^2) acceptable).
    void step() {
        int nn = n_;

        // Two-way split: droop / local
        uint32_t r = rng.bounded(1000);
        bool useDroop = (droop_thresh_ > 0) && (r < droop_thresh_);
        bool isUp = rng.coin();

        if (useDroop) {
            smartDroopStep();
        } else {
            stats.proposals++;
            stats.local_proposals++;
            int i = 1 + rng.bounded(nn - 1), j = 1 + rng.bounded(nn - 1);

            // Save 4 cells of 2x2 block before in-place mutation
            uint8_t s00 = state.get(i-1,j-1), s01 = state.get(i-1,j);
            uint8_t s10 = state.get(i,j-1),   s11 = state.get(i,j);

            bool moved = isUp ? state.tryUp(i, j) : state.tryDown(i, j);

            if (moved) {
                bool hasCross = (s00==1||s01==1||s10==1||s11==1||
                                 state.get(i-1,j-1)==1||state.get(i-1,j)==1||
                                 state.get(i,j-1)==1||state.get(i,j)==1);

                if (!hasCross) {
                    // Non-cross drip/undrip: perm and outgoing edges unchanged,
                    // but internal edges need recomputing for future consistency.
                    // Skip propagation, perm scan, and undo bookkeeping.
                    nEdgeSaves_ = 0;
                    recomputeCellEdges(i, j-1);
                    recomputeCellEdges(i, j);
                    recomputeCellEdges(i-1, j-1);
                    recomputeCellEdges(i-1, j);
                    stats.accepts++;
                    stats.local_accepts++;
                } else {
                    // Incremental edge color + perm update: O(n)
                    nEdgeSaves_ = 0;
                    updateEdgesLocal(i, j);

                    if (state.crossCount() == invCount_) {
                        stats.accepts++;
                        stats.local_accepts++;
                    } else {
                        // Reject: restore tiles and edge state
                        state.set(i-1,j-1,s00); state.set(i-1,j,s01);
                        state.set(i,j-1,s10);   state.set(i,j,s11);
                        undoEdgeChanges();
                        undoPermChanges();
                        stats.reducedness_rejects++;
                    }
                }
            }
        }
    }

    // Count points i with i + sigma(i) > 2n - n/4 (1-based)
    // This should go to 0 for a well-mixed chain (no mass in the bottom-right corner).
    int cornerCount() const {
        double threshold = 2.0 * n_ - n_ / 4.0;
        int count = 0;
        for (int i = 0; i < n_; i++)
            if ((i + 1) + perm_[i] > threshold) count++;
        return count;
    }

    // Run burn-in with progress reporting (quiet=true suppresses output)
    // Burn-in trace: (step, ell, corner) tuples recorded at regular intervals
    struct TracePoint { int64_t step; int ell; int corner; };
    std::vector<TracePoint> burnin_trace_;

    void burnin(int64_t steps, bool quiet = false) {
        double t0 = omp_get_wtime();
        double last_report = t0;
        const double report_interval = 2.0;  // seconds
        // Record ell every ~100K steps (or 10000 points, whichever is coarser)
        int64_t trace_interval = std::max((int64_t)100000, steps / 10000);
        burnin_trace_.clear();
        burnin_trace_.push_back({0, state.crossCount(), cornerCount()});
        for (int64_t i = 0; i < steps; i++) {
            step();
            if ((i+1) % trace_interval == 0)
                burnin_trace_.push_back({i+1, state.crossCount(), cornerCount()});
            if (!quiet && (i & 0xFFFF) == 0) {  // check every 64K steps
                double now = omp_get_wtime();
                if (now - last_report >= report_interval) {
                    double elapsed = now - t0;
                    double pct = 100.0 * i / steps;
                    double rate = i / elapsed;
                    double eta = (rate > 0) ? (steps - i) / rate : 0;
                    printf("\r  burn-in: %.1f%% (%s/%s) %.1fM/s, ETA %s, ell=%d corner=%d   ",
                           pct, fmtNum(i).c_str(), fmtNum(steps).c_str(),
                           rate / 1e6, fmtETA(eta).c_str(), state.crossCount(), cornerCount());
                    fflush(stdout);
                    last_report = now;
                }
            }
        }
        burnin_trace_.push_back({steps, state.crossCount(), cornerCount()});
        double elapsed = omp_get_wtime() - t0;
        if (!quiet)
            printf("\r  burn-in: 100%% (%s steps) in %.1fs (%.1fM/s)                    \n",
                   fmtNum(steps).c_str(), elapsed, steps / elapsed / 1e6);
    }

    const std::vector<TracePoint>& burninTrace() const { return burnin_trace_; }

    // Sample one permutation (after thinning)
    std::vector<int> sample(int64_t thin) {
        for (int64_t i = 0; i < thin; i++) step();
        return std::vector<int>(perm_, perm_ + n_);
    }

    // Verify incremental state matches full recomputation (for testing)
    bool verifyIncremental() const {
        // Full edge color computation from scratch
        std::vector<int> ch((n_+1)*n_, 0), cv(n_*(n_+1), 0);
        for (int c = 0; c < n_; c++) ch[n_*n_+c] = c+1;
        for (int r = n_-1; r >= 0; r--) {
            for (int c = 0; c < n_; c++) {
                uint8_t tile = state.get(r, c);
                int lc = cv[r*(n_+1)+c], bc = ch[(r+1)*n_+c];
                int tc = 0, rc = 0;
                switch (tile) {
                    case 0: break;
                    case 1: rc=lc; tc=bc; break;
                    case 2: rc=bc; break;
                    case 3: tc=lc; break;
                    case 4: tc=bc; break;
                    case 5: rc=lc; break;
                }
                ch[r*n_+c] = tc;
                cv[r*(n_+1)+c+1] = rc;
            }
        }
        for (size_t i = 0; i < h_edges_.size(); i++)
            if (h_edges_[i] != ch[i]) return false;
        for (size_t i = 0; i < v_edges_.size(); i++)
            if (v_edges_[i] != cv[i]) return false;
        // Check perm
        std::vector<int> cp(n_, 0);
        for (int r = 0; r < n_; r++) {
            int pid = cv[r*(n_+1)+n_];
            if (pid > 0) cp[pid-1] = r+1;
        }
        for (int i = 0; i < n_; i++)
            if (perm_[i] != cp[i]) return false;
        if (invCount_ != countInversions(cp.data(), n_)) return false;
        return true;
    }
};

// =============================================================================
// PNG Rendering
// =============================================================================

#if HAS_PNG

void hsv_to_rgb(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b) {
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rp, gp, bp;
    if (h < 60) { rp=c; gp=x; bp=0; }
    else if (h < 120) { rp=x; gp=c; bp=0; }
    else if (h < 180) { rp=0; gp=c; bp=x; }
    else if (h < 240) { rp=0; gp=x; bp=c; }
    else if (h < 300) { rp=x; gp=0; bp=c; }
    else { rp=c; gp=0; bp=x; }
    r = (uint8_t)((rp+m)*255);
    g = (uint8_t)((gp+m)*255);
    b = (uint8_t)((bp+m)*255);
}

void strandColor(int id, int n, uint8_t& r, uint8_t& g, uint8_t& b) {
    if (id <= 0) { r=g=b=0; return; }
    float hue = std::fmod((id-1) * 360.0f / n + 30.0f, 360.0f);
    hsv_to_rgb(hue, 0.85f, 0.9f, r, g, b);
}

class PNGWriter {
    int width, height;
    std::vector<uint8_t> pixels;
public:
    PNGWriter(int w, int h) : width(w), height(h), pixels(w*h*3, 255) {}
    void set(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x<0||y<0||x>=width||y>=height) return;
        int i = (y*width+x)*3; pixels[i]=r; pixels[i+1]=g; pixels[i+2]=b;
    }
    void hline(int x1, int x2, int y, uint8_t r, uint8_t g, uint8_t b, int th=2) {
        for (int t=-th/2; t<=th/2; t++)
            for (int x=x1; x<=x2; x++) set(x, y+t, r, g, b);
    }
    void vline(int x, int y1, int y2, uint8_t r, uint8_t g, uint8_t b, int th=2) {
        for (int t=-th/2; t<=th/2; t++)
            for (int y=y1; y<=y2; y++) set(x+t, y, r, g, b);
    }
    bool write(const char* fn) {
        FILE* fp = fopen(fn, "wb"); if (!fp) return false;
        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,0,0,0);
        if (!png) { fclose(fp); return false; }
        png_infop info = png_create_info_struct(png);
        if (!info) { png_destroy_write_struct(&png,0); fclose(fp); return false; }
        if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png,&info); fclose(fp); return false; }
        png_init_io(png, fp);
        png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        std::vector<png_bytep> rows(height);
        for (int y=0; y<height; y++) rows[y] = &pixels[y*width*3];
        png_write_image(png, rows.data());
        png_write_end(png, 0);
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return true;
    }
};

void renderPermutonPNG(const std::vector<int64_t>& matrix, int n, int /*B*/, const char* filename) {
    int cellSize = std::max(4, 800 / n);
    int w = n * cellSize, h = n * cellSize;
    PNGWriter png(w, h);

    // Find max for normalization
    int64_t maxVal = 0;
    for (auto v : matrix) maxVal = std::max(maxVal, v);
    if (maxVal == 0) maxVal = 1;

    double sqrtMax = sqrt((double)maxVal);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // sqrt scaling for better dynamic range (like Mathematica MatrixPlot)
            double d = sqrt((double)matrix[i*n+j]) / sqrtMax;

            // Mathematica-style "SunsetColors": white → orange → dark red
            // Interpolate: white(1,1,1) → orange(1,0.6,0) → dark red(0.4,0,0)
            uint8_t r, g, b;
            if (d < 0.5) {
                double t = d * 2.0;  // 0..1 over first half
                r = (uint8_t)(255);
                g = (uint8_t)(255 * (1.0 - 0.4*t));
                b = (uint8_t)(255 * (1.0 - t));
            } else {
                double t = (d - 0.5) * 2.0;  // 0..1 over second half
                r = (uint8_t)(255 * (1.0 - 0.6*t));
                g = (uint8_t)(255 * (0.6 - 0.6*t));
                b = 0;
            }

            int x0 = j * cellSize, y0 = i * cellSize;
            for (int dy = 0; dy < cellSize; dy++)
                for (int dx = 0; dx < cellSize; dx++)
                    png.set(x0+dx, y0+dy, r, g, b);
        }
    }

    // Grid lines only for small matrices
    if (n <= 20) {
        for (int k = 0; k <= n; k++) {
            for (int x = 0; x < w; x++) png.set(x, std::min(k*cellSize, h-1), 180,180,180);
            for (int y = 0; y < h; y++) png.set(std::min(k*cellSize, w-1), y, 180,180,180);
        }
    }

    if (!png.write(filename)) fprintf(stderr, "Failed to write %s\n", filename);
    else { printf("Permuton PNG written to: %s\n", filename); runlog_file(filename); }
}

// Render droop/undroop size histogram as a bar chart PNG
void renderDroopHistPNG(const int64_t* droop_hist, const int64_t* undroop_hist,
                        int n, const char* filename) {
    // Find max size with nonzero data
    int maxSize = 1;
    for (int i = 1; i < n; i++)
        if (droop_hist[i] > 0 || undroop_hist[i] > 0) maxSize = i;

    int W = 800, H = 500;
    int margin_left = 60, margin_bottom = 50, margin_top = 30, margin_right = 20;
    int plotW = W - margin_left - margin_right;
    int plotH = H - margin_top - margin_bottom;
    PNGWriter png(W, H);

    // Find max count for y-axis scaling
    int64_t maxCount = 1;
    for (int i = 1; i <= maxSize; i++) {
        maxCount = std::max(maxCount, droop_hist[i]);
        maxCount = std::max(maxCount, undroop_hist[i]);
    }

    // Bar width
    int numBars = maxSize;
    if (numBars < 1) numBars = 1;
    int barW = std::max(1, plotW / (numBars * 2 + numBars));  // two bars + gap per bin
    int groupW = barW * 3;  // droop bar + undroop bar + gap

    // Draw axes
    for (int x = margin_left; x < W - margin_right; x++)
        png.set(x, H - margin_bottom, 0, 0, 0);
    for (int y = margin_top; y < H - margin_bottom; y++)
        png.set(margin_left, y, 0, 0, 0);

    // Draw bars
    for (int i = 1; i <= maxSize; i++) {
        int x0 = margin_left + (i - 1) * groupW;

        // Droop bar (orange)
        int hD = (int)((double)droop_hist[i] / maxCount * plotH);
        for (int dx = 0; dx < barW; dx++)
            for (int dy = 0; dy < hD; dy++)
                png.set(x0 + dx, H - margin_bottom - 1 - dy, 230, 140, 30);

        // Undroop bar (blue)
        int hU = (int)((double)undroop_hist[i] / maxCount * plotH);
        for (int dx = 0; dx < barW; dx++)
            for (int dy = 0; dy < hU; dy++)
                png.set(x0 + barW + dx, H - margin_bottom - 1 - dy, 50, 100, 200);
    }

    if (!png.write(filename)) fprintf(stderr, "Failed to write %s\n", filename);
    else { printf("Droop histogram PNG written to: %s\n", filename); runlog_file(filename); }
}

// Render the dxdy (mixed partial derivative of height) matrix as a heatmap PNG
void renderDxDyPNG(const std::vector<double>& dxdy, int n, const char* filename) {
    int cellSize = std::max(4, 800 / n);
    int w = n * cellSize, h = n * cellSize;
    PNGWriter png(w, h);

    // Find max absolute value for normalization
    double maxVal = 0;
    for (auto v : dxdy) maxVal = std::max(maxVal, std::abs(v));
    if (maxVal < 1e-12) maxVal = 1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = dxdy[i*n+j];
            // Same SunsetColors as permuton: white → orange → dark red
            double d = std::max(0.0, sqrt(std::max(0.0, val) / maxVal));
            uint8_t r, g, b;
            if (d < 0.5) {
                double t = d * 2.0;
                r = 255; g = (uint8_t)(255*(1.0-0.4*t)); b = (uint8_t)(255*(1.0-t));
            } else {
                double t = (d - 0.5) * 2.0;
                r = (uint8_t)(255*(1.0-0.6*t)); g = (uint8_t)(255*(0.6-0.6*t)); b = 0;
            }

            int x0 = j * cellSize, y0 = i * cellSize;
            for (int dy = 0; dy < cellSize; dy++)
                for (int dx = 0; dx < cellSize; dx++)
                    png.set(x0+dx, y0+dy, r, g, b);
        }
    }

    if (n <= 20) {
        for (int k = 0; k <= n; k++) {
            for (int x = 0; x < w; x++) png.set(x, std::min(k*cellSize, h-1), 180,180,180);
            for (int y = 0; y < h; y++) png.set(std::min(k*cellSize, w-1), y, 180,180,180);
        }
    }

    if (!png.write(filename)) fprintf(stderr, "Failed to write %s\n", filename);
    else { printf("DxDy PNG written to: %s\n", filename); runlog_file(filename); }
}

// Overlay: side-by-side permuton (left) vs dxdy (right) with same colormap,
// plus a difference strip (bottom) showing red where permuton>dxdy, blue where dxdy>permuton.
void renderOverlayPNG(const std::vector<int64_t>& matrix, const std::vector<double>& dxdy,
                      int n, const char* filename) {
    int cellSize = std::max(4, 800 / n);
    int panelW = n * cellSize;
    int gap = std::max(2, cellSize / 2);
    int totalW = panelW * 3 + gap * 2;  // left + mid + right with gaps
    int totalH = panelW;
    PNGWriter png(totalW, totalH);

    // Fill gaps with light gray
    for (int y = 0; y < totalH; y++)
        for (int g = 0; g < gap; g++) {
            png.set(panelW + g, y, 220, 220, 220);
            png.set(panelW * 2 + gap + g, y, 220, 220, 220);
        }

    // Normalize both to sum to 1 (probability distributions over the grid)
    // so their scales are comparable for the difference panel.
    double sumPerm = 0;
    for (auto v : matrix) sumPerm += (double)v;
    if (sumPerm < 1e-12) sumPerm = 1;

    double sumDxDy = 0;
    for (auto v : dxdy) if (v > 0) sumDxDy += v;
    if (sumDxDy < 1e-12) sumDxDy = 1;

    // Also need max for individual panel coloring
    double maxPerm = 0;
    for (auto v : matrix) maxPerm = std::max(maxPerm, (double)v / sumPerm);
    if (maxPerm < 1e-12) maxPerm = 1;

    double maxDxDy = 0;
    for (auto v : dxdy) maxDxDy = std::max(maxDxDy, v / sumDxDy);
    if (maxDxDy < 1e-12) maxDxDy = 1;

    // SunsetColors helper
    auto sunsetColor = [](double d, uint8_t& r, uint8_t& g, uint8_t& b) {
        d = std::max(0.0, std::min(1.0, d));
        if (d < 0.5) {
            double t = d * 2.0;
            r = 255; g = (uint8_t)(255*(1.0-0.4*t)); b = (uint8_t)(255*(1.0-t));
        } else {
            double t = (d - 0.5) * 2.0;
            r = (uint8_t)(255*(1.0-0.6*t)); g = (uint8_t)(255*(0.6-0.6*t)); b = 0;
        }
    };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Probability-normalized values (both sum to 1)
            double pProb = (double)matrix[i*n+j] / sumPerm;
            double dProb = std::max(0.0, dxdy[i*n+j]) / sumDxDy;

            // For coloring: normalize each panel by its own max
            double pVis = pProb / maxPerm;
            double dVis = dProb / maxDxDy;

            // Left panel: permuton (sqrt scaling, SunsetColors)
            uint8_t r, g, b;
            sunsetColor(sqrt(std::max(0.0, pVis)), r, g, b);
            int x0 = j * cellSize, y0 = i * cellSize;
            for (int dy = 0; dy < cellSize; dy++)
                for (int dx = 0; dx < cellSize; dx++)
                    png.set(x0+dx, y0+dy, r, g, b);

            // Middle panel: dxdy (sqrt scaling, SunsetColors)
            sunsetColor(sqrt(std::max(0.0, dVis)), r, g, b);
            x0 = panelW + gap + j * cellSize;
            for (int dy = 0; dy < cellSize; dy++)
                for (int dx = 0; dx < cellSize; dx++)
                    png.set(x0+dx, y0+dy, r, g, b);

            // Right panel: difference of probability densities
            // (red=permuton larger, blue=dxdy larger, white=same)
            double diff = pProb - dProb;
            double maxProb = std::max(maxPerm, maxDxDy);
            double absDiff = std::min(1.0, std::abs(diff) / maxProb * 5.0);
            if (diff > 0) {
                // Permuton larger → red
                r = 255; g = (uint8_t)(255*(1.0-absDiff)); b = (uint8_t)(255*(1.0-absDiff));
            } else {
                // DxDy larger → blue
                r = (uint8_t)(255*(1.0-absDiff)); g = (uint8_t)(255*(1.0-absDiff)); b = 255;
            }
            x0 = panelW * 2 + gap * 2 + j * cellSize;
            for (int dy = 0; dy < cellSize; dy++)
                for (int dx = 0; dx < cellSize; dx++)
                    png.set(x0+dx, y0+dy, r, g, b);
        }
    }

    // Grid lines for small n
    if (n <= 20) {
        for (int p = 0; p < 3; p++) {
            int offX = p * (panelW + gap);
            for (int k = 0; k <= n; k++) {
                for (int x = 0; x < panelW; x++)
                    png.set(offX + x, std::min(k*cellSize, totalH-1), 180,180,180);
                for (int y = 0; y < totalH; y++)
                    png.set(offX + std::min(k*cellSize, panelW-1), y, 180,180,180);
            }
        }
    }

    if (!png.write(filename)) fprintf(stderr, "Failed to write %s\n", filename);
    else { printf("Overlay PNG written to: %s\n", filename); runlog_file(filename); }
}

// Overload that uses pre-computed edge colors (from BPDMcmc)
void renderColoredPNGWithEdges(const BPD& bpd, const std::vector<int>& h_edges,
                               const std::vector<int>& v_edges, const char* filename) {
    int n = bpd.size();
    int cell;
    if (n <= 10) cell = 40;
    else if (n <= 20) cell = 30;
    else if (n <= 30) cell = 20;
    else if (n <= 50) cell = 15;
    else if (n <= 80) cell = 10;
    else cell = 6;
    int w = n*cell, h = n*cell;

    PNGWriter png(w, h);
    for (int i=0; i<=n; i++) {
        for (int x=0; x<w; x++) png.set(x, i*cell, 220,220,220);
        for (int y=0; y<h; y++) png.set(i*cell, y, 220,220,220);
    }
    int th = std::max(1, cell/5);
    for (int row=0; row<n; row++) {
        for (int col=0; col<n; col++) {
            uint8_t tile = bpd.get(row, col);
            int cx = col*cell + cell/2, cy = row*cell + cell/2;
            int left = col*cell, right = (col+1)*cell-1;
            int top = row*cell, bottom = (row+1)*cell-1;
            int lc = v_edges[row*(n+1)+col];
            int rc = v_edges[row*(n+1)+col+1];
            int tc = h_edges[row*n+col];
            int bc = h_edges[(row+1)*n+col];
            uint8_t r,g,b;
            switch (tile) {
                case 0: break;
                case 1:
                    if (lc>0) { strandColor(lc,n,r,g,b); png.hline(left,cx,cy,r,g,b,th); }
                    if (rc>0) { strandColor(rc,n,r,g,b); png.hline(cx,right,cy,r,g,b,th); }
                    if (tc>0) { strandColor(tc,n,r,g,b); png.vline(cx,top,cy,r,g,b,th); }
                    if (bc>0) { strandColor(bc,n,r,g,b); png.vline(cx,cy,bottom,r,g,b,th); }
                    break;
                case 2:
                    if (bc>0) { strandColor(bc,n,r,g,b); png.vline(cx,cy,bottom,r,g,b,th); png.hline(cx,right,cy,r,g,b,th); }
                    break;
                case 3:
                    if (lc>0) { strandColor(lc,n,r,g,b); png.hline(left,cx,cy,r,g,b,th); png.vline(cx,top,cy,r,g,b,th); }
                    break;
                case 4:
                    if (bc>0) { strandColor(bc,n,r,g,b); png.vline(cx,top,bottom,r,g,b,th); }
                    break;
                case 5:
                    if (lc>0) { strandColor(lc,n,r,g,b); png.hline(left,right,cy,r,g,b,th); }
                    break;
            }
        }
    }
    if (!png.write(filename)) fprintf(stderr, "Failed to write %s\n", filename);
}

#endif // HAS_PNG

// =============================================================================
// TikZ output
// =============================================================================

void hsv_to_rgb_f(float h, float s, float v, float& r, float& g, float& b) {
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rp, gp, bp;
    if (h < 60) { rp=c; gp=x; bp=0; }
    else if (h < 120) { rp=x; gp=c; bp=0; }
    else if (h < 180) { rp=0; gp=c; bp=x; }
    else if (h < 240) { rp=0; gp=x; bp=c; }
    else if (h < 300) { rp=x; gp=0; bp=c; }
    else { rp=c; gp=0; bp=x; }
    r = rp+m; g = gp+m; b = bp+m;
}

void strandColorF(int id, int n, float& r, float& g, float& b) {
    if (id <= 0) { r=g=b=0; return; }
    float hue = std::fmod((id-1) * 360.0f / n + 30.0f, 360.0f);
    hsv_to_rgb_f(hue, 0.85f, 0.9f, r, g, b);
}

void writeTikZ(const BPD& bpd, const std::vector<int>& h_edges,
               const std::vector<int>& v_edges, const char* filename) {
    int n = bpd.size();
    FILE* f = fopen(filename, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return; }

    fprintf(f, "\\documentclass[tikz,border=2pt]{standalone}\n");
    fprintf(f, "\\usepackage{tikz}\n");
    fprintf(f, "\\usepackage{xcolor}\n");
    fprintf(f, "\\begin{document}\n");
    fprintf(f, "%% Auto-generated BPD TikZ (n=%d, MCMC sampler)\n", n);
    fprintf(f, "%% Pipes enter from bottom, exit right.\n");
    fprintf(f, "%% Row 0 = top of grid in TikZ (row r in data = TikZ y = n-1-r).\n\n");

    for (int id = 1; id <= n; id++) {
        float r, g, b;
        strandColorF(id, n, r, g, b);
        fprintf(f, "\\definecolor{pipe%d}{rgb}{%.3f,%.3f,%.3f}\n", id, r, g, b);
    }

    float scale = (n <= 20) ? 0.5f : (n <= 40) ? 0.3f : (n <= 60) ? 0.2f : 0.15f;
    float lw = (n > 60) ? 1.5f : 2.0f;

    fprintf(f, "\n\\begin{tikzpicture}[scale=%.2f]\n", scale);
    fprintf(f, "  \\draw[thin, gray!30] (0,0) grid (%d,%d);\n", n, n);

    for (int r = 0; r < n; r++) {
        int y = n - 1 - r;
        for (int c = 0; c < n; c++) {
            uint8_t tile = bpd.get(r, c);
            int lc = v_edges[r*(n+1)+c];
            int bc = h_edges[(r+1)*n+c];
            float cx = c + 0.5f, cy = y + 0.5f;
            switch (tile) {
                case 0: break;
                case 1:
                    if (bc > 0) fprintf(f, "  \\draw[pipe%d, line width=%.1fpt] (%.1f,%d) -- (%.1f,%d);\n", bc, lw, cx, y, cx, y+1);
                    if (lc > 0) fprintf(f, "  \\draw[pipe%d, line width=%.1fpt] (%d,%.1f) -- (%d,%.1f);\n", lc, lw, c, cy, c+1, cy);
                    break;
                case 2:
                    if (bc > 0) fprintf(f, "  \\draw[pipe%d, line width=%.1fpt] (%.1f,%d) to[out=90,in=180] (%d,%.1f);\n", bc, lw, cx, y, c+1, cy);
                    break;
                case 3:
                    if (lc > 0) fprintf(f, "  \\draw[pipe%d, line width=%.1fpt] (%d,%.1f) to[out=0,in=270] (%.1f,%d);\n", lc, lw, c, cy, cx, y+1);
                    break;
                case 4:
                    if (bc > 0) fprintf(f, "  \\draw[pipe%d, line width=%.1fpt] (%.1f,%d) -- (%.1f,%d);\n", bc, lw, cx, y, cx, y+1);
                    break;
                case 5:
                    if (lc > 0) fprintf(f, "  \\draw[pipe%d, line width=%.1fpt] (%d,%.1f) -- (%d,%.1f);\n", lc, lw, c, cy, c+1, cy);
                    break;
            }
        }
    }

    fprintf(f, "\\end{tikzpicture}\n");
    fprintf(f, "\\end{document}\n");
    fclose(f);
    printf("TikZ written to: %s\n", filename);
    runlog_file(filename);
}

// =============================================================================
// Output helpers
// =============================================================================

void writeMathematicaMatrix(const std::vector<int64_t>& matrix, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "%sperm_matrix_n%d_B%d%s.txt", g_ts, n, B, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    fprintf(fp, "{");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "{");
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%lld", (long long)matrix[i * n + j]);
            if (j < n-1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (i < n-1) fprintf(fp, ",");
    }
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Output written to: %s\n", filename); runlog_file(filename);
}

void writeMathematicaPerms(const std::vector<std::vector<int>>& perms, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "%sperms_n%d_B%d%s.txt", g_ts, n, B, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    fprintf(fp, "{");
    for (size_t p = 0; p < perms.size(); p++) {
        fprintf(fp, "{");
        for (int i = 0; i < n; i++) {
            fprintf(fp, "%d", perms[p][i]);
            if (i < n-1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (p < perms.size()-1) fprintf(fp, ",\n");
    }
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Output written to: %s\n", filename); runlog_file(filename);
}

void writeHeightSum(const std::vector<int64_t>& sumHeight, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "%sheight_sum_n%d_B%d%s.txt", g_ts, n, B, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    int m = n + 1;
    fprintf(fp, "{");
    for (int r = 0; r < m; r++) {
        fprintf(fp, "{");
        for (int c = 0; c < m; c++) {
            fprintf(fp, "%lld", (long long)sumHeight[r * m + c]);
            if (c < m-1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (r < m-1) fprintf(fp, ",\n");
    }
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Output written to: %s\n", filename); runlog_file(filename);
}

void writeHeightAvg(const std::vector<double>& avgHeight, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "%sheight_avg_n%d_B%d%s.txt", g_ts, n, B, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    int m = n + 1;
    fprintf(fp, "{");
    for (int r = 0; r < m; r++) {
        fprintf(fp, "{");
        for (int c = 0; c < m; c++) {
            fprintf(fp, "%.6f", avgHeight[r * m + c]);
            if (c < m-1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (r < m-1) fprintf(fp, ",\n");
    }
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Output written to: %s\n", filename); runlog_file(filename);
}

void writeHeightDxDy(const std::vector<double>& dxdy, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "%sheight_dxdy_n%d_B%d%s.txt", g_ts, n, B, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    fprintf(fp, "{");
    for (int r = 0; r < n; r++) {
        fprintf(fp, "{");
        for (int c = 0; c < n; c++) {
            fprintf(fp, "%.6f", dxdy[r * n + c]);
            if (c < n-1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (r < n-1) fprintf(fp, ",\n");
    }
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Output written to: %s\n", filename); runlog_file(filename);
}

void writeHeightSingle(const std::vector<int>& height, int n) {
    char filename[128];
    snprintf(filename, sizeof(filename), "%sheight_single_n%d%s.txt", g_ts, n, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    int m = n + 1;
    fprintf(fp, "{");
    for (int r = 0; r < m; r++) {
        fprintf(fp, "{");
        for (int c = 0; c < m; c++) {
            fprintf(fp, "%d", height[r * m + c]);
            if (c < m-1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (r < m-1) fprintf(fp, ",\n");
    }
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Output written to: %s\n", filename); runlog_file(filename);
}

// Write all individual height functions: one matrix per sample, separated by blank lines
void writeHeightAll(const std::vector<std::vector<int>>& allHeights, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "%sheight_all_n%d_B%d%s.txt", g_ts, n, B, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    int m = n + 1;
    for (int t = 0; t < (int)allHeights.size(); t++) {
        const auto& h = allHeights[t];
        fprintf(fp, "# sample %d\n", t);
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
                fprintf(fp, "%d", h[r * m + c]);
                if (c < m-1) fprintf(fp, " ");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Output written to: %s\n", filename); runlog_file(filename);
}

// Write burn-in trace as a three-column text file (step ell corner) for pgfplots
void writeBurninTrace(const std::vector<BPDMcmc::TracePoint>& trace,
                      int n, const char* startName, const char* distName,
                      const char* anchorName, double geomP) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%sburnin_trace_n%d%s.txt", g_ts, n, g_suffix);
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return; }
    if (strcmp(distName, "geometric") == 0) {
        fprintf(fp, "# step ell corner (n=%d, start=%s, droop_dist=%s, geom_p=%.6g, geom_mean=%.6g, anchor=%s)\n",
                n, startName, distName, geomP, 1.0 / geomP, anchorName);
    } else {
        fprintf(fp, "# step ell corner (n=%d, start=%s, droop_dist=%s, anchor=%s)\n",
                n, startName, distName, anchorName);
    }
    fprintf(fp, "# corner = #{i : i+sigma(i) > 2n - n/4}\n");
    for (auto& tp : trace)
        fprintf(fp, "%lld %d %d\n", (long long)tp.step, tp.ell, tp.corner);
    fclose(fp);
    printf("Burn-in trace written to: %s\n", filename); runlog_file(filename);
}

// =============================================================================
// Main
// =============================================================================

void printHelp(const char* prog) {
    printf("BPD MCMC Sampler - approximate uniform sampling of reduced BPDs\n\n");
    printf("Usage:\n");
    printf("  %s <n> [options]           Single sample mode\n", prog);
    printf("  %s batch:<n>:<B> [options]  Batch mode\n\n", prog);
    printf("Options:\n");
    printf("  --seed <S>       Random seed (default: random)\n");
    printf("  --burnin <K>     Burn-in steps; supports K/M/G/T suffixes (default: 10*n^5)\n");
    printf("  --thin <T>       Thinning interval; supports K/M/G/T suffixes (default: 10*n^4)\n");
    printf("  --droop <P>      Smart LLS droop probability (default: 0.25)\n");
    printf("  --droop-dist <D> Rectangle size dist: geometric, uniform, loguniform, revlog (default: geometric)\n");
    printf("  --geom-p <P>     Geometric side-length parameter when using geometric dist (default: 0.5)\n");
    printf("  --geom-mean <M>  Geometric mean side length when using geometric dist (default: 2)\n");
    printf("  --anchor <A>     Droop anchor corner: nw or se (default: nw)\n");
    printf("  --start <S>      Starting state: identity or w0 (default: identity)\n");
    printf("  --restore <F>    Restore from checkpoint file (all chains, skip burn-in)\n");
    printf("  --threads <T>    Number of independent chains in batch mode (default: 6 macOS, 8 Linux)\n");
    printf("  --no-png         Suppress PNG output\n");
    printf("  --no-tikz        Suppress TikZ output\n");
    printf("  --no-height      Suppress height function output\n");
    printf("  --checkpoint     Save checkpoint after burn-in (off by default)\n\n");
    printf("Single mode outputs: PNG, TikZ, height function of sample.\n");
    printf("Batch mode outputs: perm matrix, all perms, permuton PNG, dxdy PNG,\n");
    printf("  permuton/dxdy overlay PNG, BPD PNG+TikZ of last sample, height sum/avg/dxdy.\n\n");
    printf("Algorithm: Independent-chain Metropolis on reduced BPDs.\n");
    printf("Batch mode: T threads each run independent chains with unique seeds.\n");
    printf("Proposals: local 2x2 drip/undrip/cross moves + rectangular droops.\n");
    printf("Acceptance: always accept if move preserves reducedness, reject otherwise.\n");
    printf("Stationary distribution: uniform on RBPDs (by detailed balance).\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        printHelp(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    runlog_open(argc, argv);

    // Parse mode
    bool batchMode = strncmp(argv[1], "batch:", 6) == 0;
    bool verifyMode = strncmp(argv[1], "verify:", 7) == 0;
    int n = 0, B = 0;
    int verifySteps = 0;

    if (verifyMode) {
        if (sscanf(argv[1], "verify:%d:%d", &n, &verifySteps) != 2 || n < 2 || verifySteps < 1) {
            fprintf(stderr, "Error: Invalid verify format. Use verify:<n>:<steps>\n");
            return 1;
        }
        if (n > MAX_N) { fprintf(stderr, "Error: n=%d exceeds MAX_N=%d\n", n, MAX_N); return 1; }
        // Run verification: check incremental edge colors, perm, invCount, and
        // isReducedSlow() vs incremental reducedness on every step.
        uint32_t seed = 12345;
        for (int i = 2; i < argc; i++)
            if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
                seed = (uint32_t)atol(argv[++i]);

        printf("Verify mode: n=%d, steps=%d, seed=%u\n", n, verifySteps, seed);

        double vDroopProb = 0.25;
        DroopDist vDroopDist = DroopDist::GEOMETRIC;
        DroopAnchor vDroopAnchor = DroopAnchor::SE;
        double vGeomP = 0.5;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--droop") == 0 && i+1 < argc)
                vDroopProb = atof(argv[++i]);
            else if (strcmp(argv[i], "--droop-dist") == 0 && i+1 < argc) {
                const char* a = argv[++i];
                if (strcmp(a, "uniform") == 0) vDroopDist = DroopDist::UNIFORM;
                else if (strcmp(a, "loguniform") == 0) vDroopDist = DroopDist::LOGUNIFORM;
                else if (strcmp(a, "revlog") == 0) vDroopDist = DroopDist::REVLOGUNIFORM;
                else vDroopDist = DroopDist::GEOMETRIC;
            }
            else if (strcmp(argv[i], "--geom-p") == 0 && i+1 < argc)
                vGeomP = atof(argv[++i]);
            else if (strcmp(argv[i], "--geom-mean") == 0 && i+1 < argc)
                vGeomP = 1.0 / atof(argv[++i]);
            else if (strcmp(argv[i], "--anchor") == 0 && i+1 < argc) {
                const char* a = argv[++i];
                if (strcmp(a, "nw") == 0) vDroopAnchor = DroopAnchor::NW;
                else vDroopAnchor = DroopAnchor::SE;
            }
        }
        if (!(vGeomP > 0.0 && vGeomP <= 1.0) || !std::isfinite(vGeomP)) {
            fprintf(stderr, "Error: geometric parameter must satisfy 0 < p <= 1\n");
            return 1;
        }
        BPDMcmc mc(n, vDroopProb, vDroopDist, vDroopAnchor, vGeomP);
        mc.seed(seed);
        mc.init();
        int mismatches = 0;
        double t0 = omp_get_wtime();

        for (int s = 0; s < verifySteps; s++) {
            mc.step();
            // Check 1: incremental edge colors/perm/invCount match full recomputation
            if (!mc.verifyIncremental()) {
                mismatches++;
                printf("EDGE MISMATCH at step %d: incremental != full recompute\n", s);
                if (mismatches >= 10) { printf("Too many mismatches, aborting.\n"); return 1; }
            }
            // Check 2: isReducedSlow() agrees with incremental crossCount==invCount
            const BPD& cur = mc.current();
            bool sr = cur.isReducedSlow();
            bool fr = (cur.crossCount() == mc.getInvCount());
            if (sr != fr) {
                mismatches++;
                printf("REDUCED MISMATCH at step %d: slow=%d incr=%d cc=%d inv=%d\n",
                       s, sr, fr, cur.crossCount(), mc.getInvCount());
                if (mismatches >= 10) { printf("Too many mismatches, aborting.\n"); return 1; }
            }
        }
        double elapsed = omp_get_wtime() - t0;
        if (mismatches == 0)
            printf("PASS: %d steps, all checks agree (%.1fs, %.1fK/s)\n",
                   verifySteps, elapsed, verifySteps / elapsed / 1e3);
        else
            printf("FAIL: %d mismatches in %d steps\n", mismatches, verifySteps);
        return mismatches > 0 ? 1 : 0;
    }

    if (batchMode) {
        if (sscanf(argv[1], "batch:%d:%d", &n, &B) != 2 || n < 2 || B < 1) {
            fprintf(stderr, "Error: Invalid batch format. Use batch:<n>:<B>\n");
            return 1;
        }
    } else {
        n = atoi(argv[1]);
        if (n < 2) { fprintf(stderr, "Error: n must be >= 2\n"); return 1; }
    }
    if (n > MAX_N) { fprintf(stderr, "Error: n=%d exceeds MAX_N=%d\n", n, MAX_N); return 1; }
    if (!batchMode) {
        B = 1;
    }

    // Parse options
    uint32_t seedVal = 0;
    int64_t burnin = 10LL * n * n * n * n * n;
    int64_t thin = 10LL * n * n * n * n;
    double droopProb = 0.25;
    double geomP = 0.5;
    DroopDist droopDist = DroopDist::GEOMETRIC;
    DroopDist collectDist = DroopDist::GEOMETRIC;
    bool collectDistSet = false;
    DroopAnchor droopAnchor = DroopAnchor::SE;
    bool outputPng = true;
    bool outputTikz = true;
    bool outputHeight = true;
    bool outputCheckpoint = false;
    bool startW0 = false;
    const char* restoreFile = nullptr;
#ifdef __APPLE__
    int numThreads = 6;
#else
    int numThreads = 8;
#endif
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            seedVal = (uint32_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--burnin") == 0 && i+1 < argc)
            burnin = parseCount(argv[++i]);
        else if (strcmp(argv[i], "--thin") == 0 && i+1 < argc)
            thin = parseCount(argv[++i]);
        else if (strcmp(argv[i], "--rect") == 0 && i+1 < argc)
            { ++i; }  // accepted for backward compatibility, ignored
        else if (strcmp(argv[i], "--droop") == 0 && i+1 < argc)
            droopProb = atof(argv[++i]);
        else if (strcmp(argv[i], "--geom-p") == 0 && i+1 < argc)
            geomP = atof(argv[++i]);
        else if (strcmp(argv[i], "--geom-mean") == 0 && i+1 < argc)
            geomP = 1.0 / atof(argv[++i]);
        else if (strcmp(argv[i], "--droop-dist") == 0 && i+1 < argc) {
            const char* dd = argv[++i];
            if (strcmp(dd, "geometric") == 0) droopDist = DroopDist::GEOMETRIC;
            else if (strcmp(dd, "uniform") == 0) droopDist = DroopDist::UNIFORM;
            else if (strcmp(dd, "loguniform") == 0) droopDist = DroopDist::LOGUNIFORM;
            else if (strcmp(dd, "revlog") == 0) droopDist = DroopDist::REVLOGUNIFORM;
            else { fprintf(stderr, "Error: --droop-dist must be geometric, uniform, loguniform, or revlog\n"); return 1; }
        }
        else if (strcmp(argv[i], "--collect-droop-dist") == 0 && i+1 < argc) {
            const char* dd = argv[++i];
            collectDistSet = true;
            if (strcmp(dd, "geometric") == 0) collectDist = DroopDist::GEOMETRIC;
            else if (strcmp(dd, "uniform") == 0) collectDist = DroopDist::UNIFORM;
            else if (strcmp(dd, "loguniform") == 0) collectDist = DroopDist::LOGUNIFORM;
            else if (strcmp(dd, "revlog") == 0) collectDist = DroopDist::REVLOGUNIFORM;
            else { fprintf(stderr, "Error: --collect-droop-dist must be geometric, uniform, loguniform, or revlog\n"); return 1; }
        }
        else if (strcmp(argv[i], "--anchor") == 0 && i+1 < argc) {
            const char* a = argv[++i];
            if (strcmp(a, "nw") == 0) droopAnchor = DroopAnchor::NW;
            else if (strcmp(a, "se") == 0) droopAnchor = DroopAnchor::SE;
            else { fprintf(stderr, "Error: --anchor must be 'nw' or 'se'\n"); return 1; }
        }
        else if (strcmp(argv[i], "--start") == 0 && i+1 < argc) {
            const char* s = argv[++i];
            if (strcmp(s, "w0") == 0) startW0 = true;
            else if (strcmp(s, "identity") == 0) startW0 = false;
            else { fprintf(stderr, "Error: --start must be 'identity' or 'w0'\n"); return 1; }
        }
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc)
            numThreads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--export") == 0)
            ;  // always on, accepted for backward compatibility
        else if (strcmp(argv[i], "--png") == 0)
            ;  // always on, accepted for backward compatibility
        else if (strcmp(argv[i], "--no-png") == 0)
            outputPng = false;
        else if (strcmp(argv[i], "--no-tikz") == 0)
            outputTikz = false;
        else if (strcmp(argv[i], "--no-height") == 0)
            outputHeight = false;
        else if (strcmp(argv[i], "--checkpoint") == 0)
            outputCheckpoint = true;
        else if (strcmp(argv[i], "--restore") == 0 && i+1 < argc)
            restoreFile = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printHelp(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1;
        }
    }

    if (seedVal == 0) seedVal = std::random_device{}();
    if (!(geomP > 0.0 && geomP <= 1.0) || !std::isfinite(geomP)) {
        fprintf(stderr, "Error: geometric parameter must satisfy 0 < p <= 1\n");
        return 1;
    }

    { time_t now = time(nullptr); struct tm* t = localtime(&now);
      snprintf(g_ts, sizeof(g_ts), "%04d%02d%02d_%02d%02d%02d_",
               t->tm_year+1900, t->tm_mon+1, t->tm_mday,
               t->tm_hour, t->tm_min, t->tm_sec); }

    if (numThreads < 1) numThreads = 1;
    if (!collectDistSet) collectDist = droopDist;
    const char* ddName = droopDist == DroopDist::UNIFORM ? "uniform" :
                         droopDist == DroopDist::LOGUNIFORM ? "loguniform" :
                         droopDist == DroopDist::REVLOGUNIFORM ? "revlog" : "geometric";
    const char* cdName = collectDist == DroopDist::UNIFORM ? "uniform" :
                         collectDist == DroopDist::LOGUNIFORM ? "loguniform" :
                         collectDist == DroopDist::REVLOGUNIFORM ? "revlog" : "geometric";
    const char* daName = droopAnchor == DroopAnchor::SE ? "se" : "nw";
    const char* startName = startW0 ? "w0" : "identity";
    snprintf(g_suffix, sizeof(g_suffix), "_%s_%s_%s_s%u_t%s", ddName, daName, startName, seedVal, fmtNum(thin).c_str());
    if (collectDistSet && collectDist != droopDist) {
        printf("BPD MCMC: n=%d, B=%d, seed=%u, burnin=%s(%s), collect=%s, thin=%s, droop=%.0f%%(anchor=%s), start=%s, threads=%d\n",
               n, B, seedVal, fmtNum(burnin).c_str(), ddName, cdName, fmtNum(thin).c_str(),
               droopProb*100, daName, startName, batchMode ? numThreads : 1);
    } else {
        printf("BPD MCMC: n=%d, B=%d, seed=%u, burnin=%s, thin=%s, droop=%.0f%%(%s,anchor=%s), start=%s, threads=%d\n",
               n, B, seedVal, fmtNum(burnin).c_str(), fmtNum(thin).c_str(),
               droopProb*100, ddName, daName, startName, batchMode ? numThreads : 1);
    }
    if (droopDist == DroopDist::GEOMETRIC || collectDist == DroopDist::GEOMETRIC)
        printf("Geometric side lengths: p=%.6g (mean %.6g)\n", geomP, 1.0 / geomP);

    double t0 = omp_get_wtime();

    if (!batchMode) {
        // Single sample
        BPDMcmc mc(n, droopProb, droopDist, droopAnchor, geomP);
        if (restoreFile) {
            if (!mc.loadCheckpoint(restoreFile)) return 1;
        } else {
            mc.seed(seedVal);
            mc.init(startW0);
            printf("Burning in...\n");
            mc.burnin(burnin);
            writeBurninTrace(mc.burninTrace(), n, startName, ddName, daName, geomP);
            if (outputCheckpoint) {
                char cpfn[128];
                { time_t now = time(nullptr); struct tm* t = localtime(&now);
                  snprintf(cpfn, sizeof(cpfn), "checkpoint_n%d_%04d%02d%02d_%02d%02d%02d.bin",
                           n, t->tm_year+1900, t->tm_mon+1, t->tm_mday,
                           t->tm_hour, t->tm_min, t->tm_sec); }
                mc.saveCheckpoint(cpfn);
            }
        }

        auto p = mc.sample(0);
        printf("Permutation: (");
        for (int i = 0; i < n; i++) printf("%d%s", p[i], i<n-1?",":"");
        printf(")\n");
        printf("Crosses (ell): %d\n", mc.current().crossCount());

        auto& s = mc.getStats();
        printf("Stats: %s proposals, %s accepts (%.3f%%), %s reducedness rejects\n",
               fmtNum(s.proposals).c_str(), fmtNum(s.accepts).c_str(),
               s.proposals > 0 ? 100.0*s.accepts/s.proposals : 0,
               fmtNum(s.reducedness_rejects).c_str());
        if (s.droop_proposals > 0)
            printf("  Droop: %s proposals, %s accepts (%.3f%%, down %s, up %s)\n",
                   fmtNum(s.droop_proposals).c_str(), fmtNum(s.droop_accepts).c_str(),
                   100.0*s.droop_accepts/s.droop_proposals,
                   fmtNum(s.droop_down_accepts).c_str(), fmtNum(s.droop_up_accepts).c_str());
        if (s.local_proposals > 0)
            printf("  Local: %s proposals, %s accepts (%.3f%%)\n",
                   fmtNum(s.local_proposals).c_str(), fmtNum(s.local_accepts).c_str(),
                   100.0*s.local_accepts/s.local_proposals);

#if HAS_PNG
        if (outputPng) {
            char fn[128];
            snprintf(fn, sizeof(fn), "%sbpd_mcmc_n%d%s.png", g_ts, n, g_suffix);
            renderColoredPNGWithEdges(mc.current(), mc.hEdges(), mc.vEdges(), fn);
            printf("PNG written to: %s\n", fn); runlog_file(fn);
        }
#else
        if (outputPng) printf("PNG disabled - compile with -lpng\n");
#endif

        if (outputTikz) {
            char fn[128];
            snprintf(fn, sizeof(fn), "%sbpd_mcmc_n%d%s.tex", g_ts, n, g_suffix);
            writeTikZ(mc.current(), mc.hEdges(), mc.vEdges(), fn);
        }

        if (outputHeight) {
            std::vector<int> height;
            mc.computeHeight(height);
            writeHeightSingle(height, n);
        }

    } else {
        // Batch mode: independent parallel chains
        int T = numThreads;
        omp_set_num_threads(T);

        std::vector<int64_t> sumMatrix(n * n, 0);
        std::vector<std::vector<int>> allPerms(B);
        int m = n + 1;
        std::vector<int64_t> sumHeight(m * m, 0);

        // Distribute B samples across T threads
        std::vector<int> perThread(T);
        std::vector<int> threadOffset(T);  // starting index in allPerms for each thread
        {
            int base = B / T, rem = B % T;
            int off = 0;
            for (int t = 0; t < T; t++) {
                perThread[t] = base + (t < rem ? 1 : 0);
                threadOffset[t] = off;
                off += perThread[t];
            }
        }

        // Progress counter shared across threads
        std::atomic<int> completedAtomic{0};

        // Thread-local storage for per-thread chain state and results
        struct ThreadData {
            std::vector<int64_t> localSumMatrix;
            std::vector<std::vector<int>> localPerms;
            std::vector<int64_t> localSumHeight;
            std::vector<std::vector<int>> localAllHeights;  // per-sample height functions
            MCMCStats localStats;
            std::unique_ptr<BPDMcmc> chain;
        };
        std::vector<ThreadData> tdata(T);

        // --- Restore or burn-in ---
        double burninStart = omp_get_wtime();
        double collectStartTime = 0;

        if (restoreFile) {
            // Load all chains from multi-chain checkpoint (or legacy single-chain)
            std::vector<std::unique_ptr<BPDMcmc>> restoredChains;
            int restoredT = BPDMcmc::loadMultiCheckpoint(restoreFile, n, droopProb, droopDist, geomP, restoredChains, droopAnchor);
            if (restoredT == 0) {
                fprintf(stderr, "Failed to load checkpoint\n");
                return 1;
            }
            if (restoredT != T) {
                printf("Checkpoint has %d chains, adjusting thread count to match\n", restoredT);
                T = restoredT;
                omp_set_num_threads(T);
                perThread.resize(T);
                threadOffset.resize(T);
                int base = B / T, rem = B % T;
                int off = 0;
                for (int t = 0; t < T; t++) {
                    perThread[t] = base + (t < rem ? 1 : 0);
                    threadOffset[t] = off;
                    off += perThread[t];
                }
                tdata.resize(T);
            }
            for (int t = 0; t < T; t++)
                tdata[t].chain = std::move(restoredChains[t]);
        } else {
            // Burn in T independent chains in parallel
            printf("Burning in %d independent chains...\n", T);

            #pragma omp parallel
            {
                #pragma omp single
                {
                    int actualT = omp_get_num_threads();
                    if (actualT != T) {
                        printf("Note: OpenMP provided %d threads (requested %d)\n", actualT, T);
                        T = actualT;
                        perThread.resize(T);
                        threadOffset.resize(T);
                        int base = B / T, rem = B % T;
                        int off = 0;
                        for (int t = 0; t < T; t++) {
                            perThread[t] = base + (t < rem ? 1 : 0);
                            threadOffset[t] = off;
                            off += perThread[t];
                        }
                        tdata.resize(T);
                    }
                }

                int tid = omp_get_thread_num();
                ThreadData& td = tdata[tid];
                td.chain = std::make_unique<BPDMcmc>(n, droopProb, droopDist, droopAnchor, geomP);
                td.chain->seed(seedVal + (uint64_t)tid * 1000003ULL);
                td.chain->init(startW0);
                td.chain->burnin(burnin, /*quiet=*/ tid != 0);
            }

            double burninElapsed = omp_get_wtime() - burninStart;
            auto& bs = tdata[0].chain->getStats();
            printf("Burn-in done (%d chains, %.1fs). Thread 0 acceptance: %.3f%%",
                   T, burninElapsed,
                   bs.proposals > 0 ? 100.0*bs.accepts/bs.proposals : 0);
            if (bs.droop_proposals > 0)
                printf(", droop %.3f%% (down %s, up %s)", 100.0*bs.droop_accepts/bs.droop_proposals,
                       fmtNum(bs.droop_down_accepts).c_str(), fmtNum(bs.droop_up_accepts).c_str());
            if (bs.local_proposals > 0)
                printf(", local %.3f%%", 100.0*bs.local_accepts/bs.local_proposals);
            printf(", ell=%d\n", tdata[0].chain->current().crossCount());
            writeBurninTrace(tdata[0].chain->burninTrace(), n, startName, ddName, daName, geomP);

            // Save all chains checkpoint (off by default, enable with --checkpoint)
            if (outputCheckpoint) {
                std::vector<std::unique_ptr<BPDMcmc>> chains(T);
                for (int t = 0; t < T; t++) chains[t] = std::move(tdata[t].chain);
                char cpfn[128];
                { time_t now = time(nullptr); struct tm* tm = localtime(&now);
                  snprintf(cpfn, sizeof(cpfn), "checkpoint_n%d_s%u_%04d%02d%02d_%02d%02d%02d.bin",
                           n, seedVal, tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday,
                           tm->tm_hour, tm->tm_min, tm->tm_sec); }
                BPDMcmc::saveMultiCheckpoint(cpfn, chains);
                for (int t = 0; t < T; t++) tdata[t].chain = std::move(chains[t]);
            }
        }

        for (int t = 0; t < T; t++)
            tdata[t].chain->resetStats();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            ThreadData& td = tdata[tid];

            // Allocate thread-local result buffers
            int myB = perThread[tid];
            td.localSumMatrix.assign(n * n, 0);
            td.localPerms.resize(myB);
            if (outputHeight) {
                td.localSumHeight.assign(m * m, 0);
                td.localAllHeights.resize(myB);
            }

            // Switch droop distribution for collection if requested
            if (collectDist != droopDist) {
                td.chain->setDroopDist(collectDist);
                #pragma omp single
                printf("Switched droop distribution: %s -> %s for collection\n", ddName, cdName);
            }

            // --- Collection phase ---
            #pragma omp single
            {
                g_stop_requested = 0;
                std::signal(SIGINT, handle_sigint);
                printf("Collecting %d samples (%d chains, thin=%s, Ctrl+C to stop early)...\n",
                       B, T, fmtNum(thin).c_str());
                collectStartTime = omp_get_wtime();
            }

            double sampleStart = omp_get_wtime();
            double lastReport = sampleStart;
            std::vector<int> sampleHeightLocal;
            int actualMyB = 0;

            for (int b = 0; b < myB && !g_stop_requested; b++) {
                for (int64_t i = 0; i < thin; i++) {
                    td.chain->step();
                    if (g_stop_requested) break;
                    // Progress reporting from thread 0 only
                    if (tid == 0 && (i & 0xFFFF) == 0) {
                        double now = omp_get_wtime();
                        if (now - lastReport >= 0.3) {
                            int done = completedAtomic.load(std::memory_order_relaxed);
                            double elapsed = now - sampleStart;
                            double pct = 100.0 * done / B;
                            double sampPerSec = done / elapsed;
                            double eta = (done > 0) ? (B - done) / sampPerSec : 0;
                            printf("\r  collect: %.1f%% (%d/%d samples, %.1f samp/s, ETA %s)   ",
                                   pct, done, B, sampPerSec, fmtETA(eta).c_str());
                            fflush(stdout);
                            lastReport = now;
                        }
                    }
                }
                if (g_stop_requested) break;

                const int* perm = td.chain->getPerm();
                for (int i = 0; i < n; i++)
                    td.localSumMatrix[i * n + (perm[i]-1)] += 1;
                td.localPerms[b].assign(perm, perm + n);
                if (outputHeight) {
                    td.chain->computeHeight(sampleHeightLocal);
                    for (int k = 0; k < m * m; k++)
                        td.localSumHeight[k] += sampleHeightLocal[k];
                    td.localAllHeights[b] = sampleHeightLocal;
                }
                actualMyB = b + 1;
                completedAtomic.fetch_add(1, std::memory_order_relaxed);
            }

            // Trim localPerms to actual count (may be less if stopped early)
            td.localPerms.resize(actualMyB);
            if (outputHeight)
                td.localAllHeights.resize(actualMyB);

            td.localStats = td.chain->getStats();
        }  // end parallel

        double sampleEnd = omp_get_wtime();
        std::signal(SIGINT, SIG_DFL);  // restore default handler

        // Merge thread-local results
        int totalCompleted = completedAtomic.load();
        if (g_stop_requested)
            printf("\r  Ctrl+C: stopping early with %d/%d samples                                      \n",
                   totalCompleted, B);
        {
            double elapsed = sampleEnd - collectStartTime;
            printf("  collect: %d/%d samples in %.1fs                                        \n",
                   totalCompleted, B, elapsed);
            fflush(stdout);
        }

        // Merge sumMatrix
        for (int t = 0; t < T; t++)
            for (int k = 0; k < n * n; k++)
                sumMatrix[k] += tdata[t].localSumMatrix[k];

        // Merge allPerms — concatenate actual collected perms (may be fewer than B)
        allPerms.clear();
        allPerms.reserve(totalCompleted);
        for (int t = 0; t < T; t++)
            for (auto& p : tdata[t].localPerms)
                allPerms.push_back(std::move(p));
        B = totalCompleted;  // update B for output

        // Merge sumHeight and collect all individual height functions
        std::vector<std::vector<int>> allHeights;
        if (outputHeight) {
            for (int t = 0; t < T; t++)
                for (int k = 0; k < m * m; k++)
                    sumHeight[k] += tdata[t].localSumHeight[k];
            allHeights.reserve(totalCompleted);
            for (int t = 0; t < T; t++)
                for (auto& h : tdata[t].localAllHeights)
                    allHeights.push_back(std::move(h));
        }

        // Aggregate stats from all threads
        MCMCStats aggStats{};
        for (int t = 0; t < T; t++)
            aggStats += tdata[t].localStats;

        printf("\nSampling acceptance: %.3f%% overall",
               aggStats.proposals > 0 ? 100.0*aggStats.accepts/aggStats.proposals : 0);
        if (aggStats.droop_proposals > 0)
            printf(", droop %.3f%% (down %s, up %s)", 100.0*aggStats.droop_accepts/aggStats.droop_proposals,
                   fmtNum(aggStats.droop_down_accepts).c_str(), fmtNum(aggStats.droop_up_accepts).c_str());
        printf("\n");

        // Print droop size distribution summary
        if (aggStats.droop_accepts > 0) {
            printf("Droop size distribution (max side length):\n");
            printf("  size: droop / undroop\n");
            for (int i = 1; i < n; i++) {
                int64_t d = aggStats.droop_size_hist[i];
                int64_t u = aggStats.undroop_size_hist[i];
                if (d > 0 || u > 0)
                    printf("  %3d: %s / %s\n", i, fmtNum(d).c_str(), fmtNum(u).c_str());
            }
        }

        printf("\n");

        double elapsed = omp_get_wtime() - t0;
        printf("Completed %d samples in %.2fs (%.1f samples/sec)\n", totalCompleted, elapsed, totalCompleted/elapsed);

        writeMathematicaMatrix(sumMatrix, n, totalCompleted);
        writeMathematicaPerms(allPerms, n, totalCompleted);

#if HAS_PNG
        {
            char permutonFn[128];
            snprintf(permutonFn, sizeof(permutonFn), "%spermuton_mcmc_n%d_B%d%s.png", g_ts, n, totalCompleted, g_suffix);
            renderPermutonPNG(sumMatrix, n, totalCompleted, permutonFn);
        }
        if (aggStats.droop_accepts > 0) {
            char fn[128];
            snprintf(fn, sizeof(fn), "%sdroop_hist_n%d_B%d%s.png", g_ts, n, totalCompleted, g_suffix);
            renderDroopHistPNG(aggStats.droop_size_hist, aggStats.undroop_size_hist, n, fn);
        }
        if (outputPng && tdata[0].chain) {
            char fn[128];
            snprintf(fn, sizeof(fn), "%sbpd_mcmc_n%d%s.png", g_ts, n, g_suffix);
            renderColoredPNGWithEdges(tdata[0].chain->current(), tdata[0].chain->hEdges(), tdata[0].chain->vEdges(), fn);
            printf("PNG written to: %s\n", fn); runlog_file(fn);
        }
#endif

        if (outputTikz && tdata[0].chain) {
            char fn[128];
            snprintf(fn, sizeof(fn), "%sbpd_mcmc_n%d%s.tex", g_ts, n, g_suffix);
            writeTikZ(tdata[0].chain->current(), tdata[0].chain->hEdges(), tdata[0].chain->vEdges(), fn);
        }

        if (outputHeight) {
            writeHeightSum(sumHeight, n, totalCompleted);
            std::vector<double> avgHeight(m * m);
            for (int k = 0; k < m * m; k++)
                avgHeight[k] = (double)sumHeight[k] / totalCompleted;
            writeHeightAvg(avgHeight, n, totalCompleted);
            writeHeightAll(allHeights, n, totalCompleted);
            std::vector<double> dxdy(n * n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    dxdy[r*n+c] = avgHeight[r*m+c] - avgHeight[r*m+c+1]
                                - avgHeight[(r+1)*m+c] + avgHeight[(r+1)*m+c+1];
            writeHeightDxDy(dxdy, n, totalCompleted);
#if HAS_PNG
            {
                char fn[128];
                snprintf(fn, sizeof(fn), "%sheight_dxdy_n%d_B%d%s.png", g_ts, n, totalCompleted, g_suffix);
                renderDxDyPNG(dxdy, n, fn);
            }
            {
                char fn[128];
                snprintf(fn, sizeof(fn), "%soverlay_n%d_B%d%s.png", g_ts, n, totalCompleted, g_suffix);
                renderOverlayPNG(sumMatrix, dxdy, n, fn);
            }
#endif
        }
    }

    runlog_close();
    return 0;
}
