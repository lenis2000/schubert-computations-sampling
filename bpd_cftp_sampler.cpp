// Experimental BPD CFTP sampler for reduced bumpless pipe dreams.
// This code is retained for the paper's CFTP failure diagnostics and comparisons.
// For production sampling, use bpd_mcmc.cpp instead.
//
// Single sample mode: ./bpd_cftp_sampler <n> [seed]
// Batch mode: ./bpd_cftp_sampler batch:<n>:<B> [--export]
//
// Compilation (single-threaded, with PNG):
//   clang++ -O3 -std=c++17 bpd_cftp_sampler.cpp -o bpd_cftp_sampler -lpng
//
// Compilation (batch mode with OpenMP):
//   clang++ -O3 -std=c++17 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib \
//     -I/opt/homebrew/opt/libomp/include -lomp \
//     bpd_cftp_sampler.cpp -o bpd_cftp_sampler -lpng

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#else
// Stubs for non-OpenMP builds
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline double omp_get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

// PNG support (optional, for single sample mode visualization)
#ifdef __has_include
#if __has_include(<png.h>)
#define HAS_PNG 1
#include <png.h>
#endif
#endif

#ifndef HAS_PNG
#define HAS_PNG 0
#endif

// Tile definitions: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz

// A single random update for backward CFTP: position (i,j) and direction
struct BPDUpdate {
    int8_t i, j;  // position in [1, n-1]
    bool isUp;    // true = up-move, false = down-move
};

// HSV to RGB conversion for generating distinct strand colors
void hsv_to_rgb(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b) {
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rp, gp, bp;
    if (h < 60) { rp = c; gp = x; bp = 0; }
    else if (h < 120) { rp = x; gp = c; bp = 0; }
    else if (h < 180) { rp = 0; gp = c; bp = x; }
    else if (h < 240) { rp = 0; gp = x; bp = c; }
    else if (h < 300) { rp = x; gp = 0; bp = c; }
    else { rp = c; gp = 0; bp = x; }
    r = (uint8_t)((rp + m) * 255);
    g = (uint8_t)((gp + m) * 255);
    b = (uint8_t)((bp + m) * 255);
}

// Get distinct color for strand (1-indexed, 0 = blank/no color)
void get_strand_color(int color_id, int n, uint8_t& r, uint8_t& g, uint8_t& b) {
    if (color_id <= 0) {
        r = g = b = 0;
        return;
    }
    float hue = std::fmod((color_id - 1) * 360.0f / n + 30.0f, 360.0f);
    hsv_to_rgb(hue, 0.85f, 0.9f, r, g, b);
}

// =============================================================================
// BPDEngine - Core sampling engine using backward CFTP (Propp-Wilson)
// =============================================================================

class BPDEngine {
    int n;
    std::vector<uint8_t> b0;  // Chain starting from identity BPD (min)
    std::vector<uint8_t> b1;  // Chain starting from w0 BPD (max)
    std::mt19937 rng;
    std::vector<uint8_t> temp; // Scratch buffer

public:
    BPDEngine(int size) : n(size), b0(n * n), b1(n * n), temp(n * n) {}

    void seed(uint32_t s) { rng.seed(s); }
    int get_n() const { return n; }

    // Grid access (0-indexed)
    inline uint8_t get(const std::vector<uint8_t>& b, int r, int c) const {
        return b[r * n + c];
    }

    inline void set(std::vector<uint8_t>& b, int r, int c, uint8_t val) {
        b[r * n + c] = val;
    }

    // Generate Rothe BPD for a permutation w
    void generateRotheBPD(std::vector<uint8_t>& result, const std::vector<int>& w) {
        int minW = *std::min_element(w.begin(), w.end());

        for (int j = 1; j <= n; j++) {
            if (j + minW - 1 < w[0]) {
                set(result, 0, j - 1, 0);
            } else if (j + minW - 1 == w[0]) {
                set(result, 0, j - 1, 2);
            } else {
                set(result, 0, j - 1, 5);
            }
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

    // Reset chains to extremal states (identity and w0)
    void reset() {
        std::vector<int> identity(n);
        for (int i = 0; i < n; i++) identity[i] = i + 1;
        generateRotheBPD(b0, identity);

        std::vector<int> w0(n);
        for (int i = 0; i < n; i++) w0[i] = n - i;
        generateRotheBPD(b1, w0);
    }

    // Get simple transposition index from cross at position (i,j)
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

    // Check if BPD corresponds to a reduced word (no repeated swaps)
    bool bpd2perm_check(const std::vector<uint8_t>& grid) const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;

        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1;
                int j = n + s - t - 1;
                int k = get_sk_from_cross(grid, i, j);
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
                int k = get_sk_from_cross(grid, i, j);
                if (k != -1) {
                    if (ww[k - 1] > ww[k]) return false;
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        return true;
    }

    // Compute permutation from BPD
    std::vector<int> compute_perm(const std::vector<uint8_t>& grid) const {
        std::vector<int> ww(n);
        for (int i = 0; i < n; i++) ww[i] = i + 1;

        for (int s = 1 - n; s < 1; s++) {
            for (int t = 0; t < s + n; t++) {
                int i = n - t - 1;
                int j = n + s - t - 1;
                int k = get_sk_from_cross(grid, i, j);
                if (k != -1 && k < n) {
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }

        for (int s = 1; s < n; s++) {
            for (int t = 0; t < n - s; t++) {
                int i = n - s - t - 1;
                int j = n - t - 1;
                int k = get_sk_from_cross(grid, i, j);
                if (k != -1 && k < n) {
                    std::swap(ww[k - 1], ww[k]);
                }
            }
        }
        return ww;
    }

    // Drip move: r-elbow drips down-right to j-elbow
    inline bool try_drip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i - 1, j - 1);
        uint8_t ne = get(grid, i - 1, j);
        uint8_t sw = get(grid, i, j - 1);
        uint8_t se = get(grid, i, j);

        if (nw == 2 && se == 0) {
            set(grid, i - 1, j - 1, 0);
            set(grid, i, j, 3);
            set(grid, i - 1, j, (ne == 5) ? 2 : 4);
            set(grid, i, j - 1, (sw == 4) ? 2 : 5);
            return true;
        }
        return false;
    }

    // Undrip move: reverse of drip
    inline bool try_undrip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i - 1, j - 1);
        uint8_t ne = get(grid, i - 1, j);
        uint8_t sw = get(grid, i, j - 1);
        uint8_t se = get(grid, i, j);

        if (nw == 0 && se == 3 && (ne == 2 || ne == 4) && (sw == 2 || sw == 5)) {
            set(grid, i - 1, j - 1, 2);
            set(grid, i, j, 0);
            set(grid, i - 1, j, (ne == 2) ? 5 : 3);
            set(grid, i, j - 1, (sw == 2) ? 4 : 3);
            return true;
        }
        return false;
    }

    // Cross drip move: creates or moves crosses
    inline bool try_cross_drip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i - 1, j - 1);
        uint8_t ne = get(grid, i - 1, j);
        uint8_t sw = get(grid, i, j - 1);
        uint8_t se = get(grid, i, j);

        if (nw == 2 && se == 2) {
            set(grid, i - 1, j - 1, 0);
            set(grid, i, j, 1);
            set(grid, i - 1, j, (ne == 3) ? 4 : 2);
            set(grid, i, j - 1, (sw == 3) ? 5 : 2);
            return true;
        }
        if (nw == 1 && se == 0) {
            set(grid, i - 1, j - 1, 3);
            set(grid, i, j, 3);
            set(grid, i - 1, j, (ne == 3) ? 4 : 2);
            set(grid, i, j - 1, (sw == 3) ? 5 : 2);
            return true;
        }
        if (nw == 1 && se == 2) {
            set(grid, i - 1, j - 1, 3);
            set(grid, i, j, 1);
            set(grid, i - 1, j, (ne == 3) ? 4 : 2);
            set(grid, i, j - 1, (sw == 3) ? 5 : 2);
            return true;
        }
        return false;
    }

    // Cross undrip move: reverse of cross drip
    inline bool try_cross_undrip(std::vector<uint8_t>& grid, int i, int j) {
        uint8_t nw = get(grid, i - 1, j - 1);
        uint8_t ne = get(grid, i - 1, j);
        uint8_t sw = get(grid, i, j - 1);
        uint8_t se = get(grid, i, j);

        if (nw == 0 && se == 1) {
            set(grid, i - 1, j - 1, 2);
            set(grid, i, j, 2);
            set(grid, i - 1, j, (ne == 4) ? 3 : 5);
            set(grid, i, j - 1, (sw == 5) ? 3 : 4);
            return true;
        }
        if (nw == 3 && se == 3) {
            set(grid, i - 1, j - 1, 1);
            set(grid, i, j, 0);
            set(grid, i - 1, j, (ne == 4) ? 3 : 5);
            set(grid, i, j - 1, (sw == 5) ? 3 : 4);
            return true;
        }
        if (nw == 3 && se == 1) {
            set(grid, i - 1, j - 1, 1);
            set(grid, i, j, 2);
            set(grid, i - 1, j, (ne == 4) ? 3 : 5);
            set(grid, i, j - 1, (sw == 5) ? 3 : 4);
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

    // Rectangular droop: non-local "up" move over rectangle [i1..i2] x [j1..j2]
    inline bool can_rect_droop(const std::vector<uint8_t>& grid, int i1, int j1, int i2, int j2) const {
        if (i2 < i1 + 1 || j2 < j1 + 1) return false;
        if (i1 < 0 || j1 < 0 || i2 >= n || j2 >= n) return false;

        if (get(grid, i1, j1) != 2 || get(grid, i2, j2) != 0) return false;

        for (int j = j1 + 1; j < j2; j++) {
            uint8_t north = get(grid, i1, j);
            uint8_t south = get(grid, i2, j);
            if (north == 2 || north == 3 || south == 2 || south == 3) return false;
        }
        for (int i = i1 + 1; i < i2; i++) {
            uint8_t west = get(grid, i, j1);
            uint8_t east = get(grid, i, j2);
            if (west == 2 || west == 3 || east == 2 || east == 3) return false;
        }
        for (int i = i1 + 1; i < i2; i++) {
            for (int j = j1 + 1; j < j2; j++) {
                uint8_t t = get(grid, i, j);
                if (t == 2 || t == 3) return false;
            }
        }
        return true;
    }

    inline void apply_rect_droop(std::vector<uint8_t>& grid, int i1, int j1, int i2, int j2) {
        set(grid, i1, j1, 0);
        set(grid, i2, j2, 3);

        uint8_t ne = get(grid, i1, j2);
        uint8_t sw = get(grid, i2, j1);
        if (ne == 5) set(grid, i1, j2, 2);
        else if (ne == 3) set(grid, i1, j2, 4);
        if (sw == 4) set(grid, i2, j1, 2);
        else if (sw == 3) set(grid, i2, j1, 5);

        for (int i = i1 + 1; i < i2; i++) {
            uint8_t t = get(grid, i, j1);
            if (t == 4) set(grid, i, j1, 0);
            else if (t == 1) set(grid, i, j1, 5);
        }
        for (int j = j1 + 1; j < j2; j++) {
            uint8_t t = get(grid, i1, j);
            if (t == 5) set(grid, i1, j, 0);
            else if (t == 1) set(grid, i1, j, 4);
        }
        for (int i = i1 + 1; i < i2; i++) {
            uint8_t t = get(grid, i, j2);
            if (t == 0) set(grid, i, j2, 4);
            else if (t == 5) set(grid, i, j2, 1);
        }
        for (int j = j1 + 1; j < j2; j++) {
            uint8_t t = get(grid, i2, j);
            if (t == 0) set(grid, i2, j, 5);
            else if (t == 4) set(grid, i2, j, 1);
        }
    }

    // Rectangular undroop: non-local "down" move over rectangle [i1..i2] x [j1..j2]
    inline bool can_rect_undroop(const std::vector<uint8_t>& grid, int i1, int j1, int i2, int j2) const {
        if (i2 < i1 + 1 || j2 < j1 + 1) return false;
        if (i1 < 0 || j1 < 0 || i2 >= n || j2 >= n) return false;

        if (get(grid, i1, j1) != 0 || get(grid, i2, j2) != 3) return false;

        uint8_t ne = get(grid, i1, j2);
        uint8_t sw = get(grid, i2, j1);
        if ((ne != 2 && ne != 4) || (sw != 2 && sw != 5)) return false;

        for (int j = j1 + 1; j < j2; j++) {
            uint8_t t = get(grid, i1, j);
            if (t != 0 && t != 4) return false;
        }
        for (int j = j1 + 1; j < j2; j++) {
            uint8_t t = get(grid, i2, j);
            if (t != 5 && t != 1) return false;
        }
        for (int i = i1 + 1; i < i2; i++) {
            uint8_t t = get(grid, i, j1);
            if (t != 0 && t != 5) return false;
        }
        for (int i = i1 + 1; i < i2; i++) {
            uint8_t t = get(grid, i, j2);
            if (t != 4 && t != 1) return false;
        }
        for (int i = i1 + 1; i < i2; i++) {
            for (int j = j1 + 1; j < j2; j++) {
                uint8_t t = get(grid, i, j);
                if (t == 2 || t == 3) return false;
            }
        }
        return true;
    }

    inline void apply_rect_undroop(std::vector<uint8_t>& grid, int i1, int j1, int i2, int j2) {
        set(grid, i1, j1, 2);
        set(grid, i2, j2, 0);

        uint8_t ne = get(grid, i1, j2);
        uint8_t sw = get(grid, i2, j1);
        if (ne == 2) set(grid, i1, j2, 5);
        else if (ne == 4) set(grid, i1, j2, 3);
        if (sw == 2) set(grid, i2, j1, 4);
        else if (sw == 5) set(grid, i2, j1, 3);

        for (int i = i1 + 1; i < i2; i++) {
            uint8_t t = get(grid, i, j1);
            if (t == 0) set(grid, i, j1, 4);
            else if (t == 5) set(grid, i, j1, 1);
        }
        for (int j = j1 + 1; j < j2; j++) {
            uint8_t t = get(grid, i1, j);
            if (t == 0) set(grid, i1, j, 5);
            else if (t == 4) set(grid, i1, j, 1);
        }
        for (int i = i1 + 1; i < i2; i++) {
            uint8_t t = get(grid, i, j2);
            if (t == 4) set(grid, i, j2, 0);
            else if (t == 1) set(grid, i, j2, 5);
        }
        for (int j = j1 + 1; j < j2; j++) {
            uint8_t t = get(grid, i2, j);
            if (t == 5) set(grid, i2, j, 0);
            else if (t == 1) set(grid, i2, j, 4);
        }
    }

    inline bool try_rectangular_update(std::vector<uint8_t>& grid, int i1, int j1, int i2, int j2, bool isUp) {
        if (isUp) {
            if (!can_rect_droop(grid, i1, j1, i2, j2)) return false;
            apply_rect_droop(grid, i1, j1, i2, j2);
            return true;
        }
        if (!can_rect_undroop(grid, i1, j1, i2, j2)) return false;
        apply_rect_undroop(grid, i1, j1, i2, j2);
        return true;
    }

    // =========================================================================
    // Backward CFTP (Propp-Wilson) support
    // =========================================================================

    // Apply a single random update to one grid.
    // Each chain independently checks reducedness for cross moves.
    // Drip/undrip always preserve reducedness (no check needed).
    void apply_update(std::vector<uint8_t>& grid, const BPDUpdate& u) {
        std::copy(grid.begin(), grid.end(), temp.begin());
        bool moved = u.isUp ? try_upmove(temp, u.i, u.j)
                            : try_downmove(temp, u.i, u.j);
        if (!moved) return;

        // Check if cross involved (only cross moves can break reducedness)
        bool has_cross = (grid[(u.i-1)*n+(u.j-1)] == 1 || temp[(u.i-1)*n+(u.j-1)] == 1 ||
                          grid[u.i*n+u.j] == 1 || temp[u.i*n+u.j] == 1);
        if (has_cross) {
            if (bpd2perm_check(temp)) grid.swap(temp);
            // else: reject (non-reduced) → stay in place
        } else {
            grid.swap(temp);  // drip/undrip: always safe
        }
    }

    // Run one round of backward CFTP: replay all updates on both chains from extremes.
    // Returns true if chains coalesced.
    bool run_cftp_round(const std::vector<BPDUpdate>& updates) {
        reset();
        for (const auto& u : updates) {
            apply_update(b0, u);
            apply_update(b1, u);
        }
        return b0 == b1;
    }

    bool is_met() const { return b0 == b1; }
    const std::vector<uint8_t>& get_result() const { return b0; }
};

// =============================================================================
// PNG Rendering (only if libpng available)
// =============================================================================

#if HAS_PNG

class PNGWriter {
    int width, height;
    std::vector<uint8_t> pixels; // RGB

public:
    PNGWriter(int w, int h) : width(w), height(h), pixels(w * h * 3, 255) {}

    void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        int idx = (y * width + x) * 3;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
    }

    void draw_hline(int x1, int x2, int y, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
        for (int t = -thickness/2; t <= thickness/2; t++) {
            for (int x = x1; x <= x2; x++) {
                set_pixel(x, y + t, r, g, b);
            }
        }
    }

    void draw_vline(int x, int y1, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
        for (int t = -thickness/2; t <= thickness/2; t++) {
            for (int y = y1; y <= y2; y++) {
                set_pixel(x + t, y, r, g, b);
            }
        }
    }

    void draw_dot(int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx*dx + dy*dy <= radius*radius) {
                    set_pixel(cx + dx, cy + dy, r, g, b);
                }
            }
        }
    }

    bool write(const char* filename) {
        FILE* fp = fopen(filename, "wb");
        if (!fp) return false;

        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png) { fclose(fp); return false; }

        png_infop info = png_create_info_struct(png);
        if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); return false; }

        if (setjmp(png_jmpbuf(png))) {
            png_destroy_write_struct(&png, &info);
            fclose(fp);
            return false;
        }

        png_init_io(png, fp);
        png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);

        std::vector<png_bytep> rows(height);
        for (int y = 0; y < height; y++) {
            rows[y] = &pixels[y * width * 3];
        }
        png_write_image(png, rows.data());
        png_write_end(png, nullptr);

        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return true;
    }
};

void render_bpd_png(const std::vector<uint8_t>& grid, int n, const char* filename) {
    int cell;
    if (n <= 10) cell = 40;
    else if (n <= 20) cell = 30;
    else if (n <= 30) cell = 20;
    else if (n <= 50) cell = 15;
    else cell = 10;
    const int w = n * cell;
    const int h = n * cell;

    PNGWriter png(w, h);

    // Draw grid lines
    for (int i = 0; i <= n; i++) {
        for (int x = 0; x < w; x++) png.set_pixel(x, i * cell, 200, 200, 200);
        for (int y = 0; y < h; y++) png.set_pixel(i * cell, y, 200, 200, 200);
    }

    uint8_t pr = 33, pg = 150, pb = 243;

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            uint8_t tile = grid[row * n + col];
            int cx = col * cell + cell / 2;
            int cy = row * cell + cell / 2;
            int left = col * cell;
            int right = (col + 1) * cell - 1;
            int top = row * cell;
            int bottom = (row + 1) * cell - 1;

            switch (tile) {
                case 0: break;
                case 1: // cross
                    png.draw_hline(left, right, cy, pr, pg, pb);
                    png.draw_vline(cx, top, bottom, pr, pg, pb);
                    break;
                case 2: // r-elbow
                    png.draw_hline(cx, right, cy, pr, pg, pb);
                    png.draw_vline(cx, cy, bottom, pr, pg, pb);
                    break;
                case 3: // j-elbow
                    png.draw_hline(left, cx, cy, pr, pg, pb);
                    png.draw_vline(cx, top, cy, pr, pg, pb);
                    break;
                case 4: // vert
                    png.draw_vline(cx, top, bottom, pr, pg, pb);
                    break;
                case 5: // horiz
                    png.draw_hline(left, right, cy, pr, pg, pb);
                    break;
            }
        }
    }

    if (!png.write(filename)) {
        fprintf(stderr, "Failed to write %s\n", filename);
    }
}

// Compute edge colors by propagating from bottom boundary
void compute_edge_colors(const std::vector<uint8_t>& grid, int n,
                         std::vector<int>& h_edges, std::vector<int>& v_edges) {
    h_edges.assign((n + 1) * n, 0);
    v_edges.assign(n * (n + 1), 0);

    // Bottom boundary: colors 1..n
    for (int col = 0; col < n; col++) {
        h_edges[n * n + col] = col + 1;
    }

    // Propagate bottom-to-top, left-to-right
    for (int row = n - 1; row >= 0; row--) {
        for (int col = 0; col < n; col++) {
            uint8_t tile = grid[row * n + col];
            int left_color = v_edges[row * (n + 1) + col];
            int bottom_color = h_edges[(row + 1) * n + col];
            int right_color = 0, top_color = 0;

            switch (tile) {
                case 0: right_color = 0; top_color = 0; break;
                case 1: right_color = left_color; top_color = bottom_color; break;
                case 2: right_color = bottom_color; top_color = 0; break;
                case 3: right_color = 0; top_color = left_color; break;
                case 4: right_color = 0; top_color = bottom_color; break;
                case 5: right_color = left_color; top_color = 0; break;
            }

            v_edges[row * (n + 1) + col + 1] = right_color;
            h_edges[row * n + col] = top_color;
        }
    }
}

void render_bpd_colored_png(const std::vector<uint8_t>& grid, int n,
                            const std::vector<int>& h_edges,
                            const std::vector<int>& v_edges,
                            const char* filename) {
    int cell;
    if (n <= 10) cell = 40;
    else if (n <= 20) cell = 30;
    else if (n <= 30) cell = 20;
    else if (n <= 50) cell = 15;
    else cell = 10;
    const int w = n * cell;
    const int h = n * cell;

    PNGWriter png(w, h);

    for (int i = 0; i <= n; i++) {
        for (int x = 0; x < w; x++) png.set_pixel(x, i * cell, 200, 200, 200);
        for (int y = 0; y < h; y++) png.set_pixel(i * cell, y, 200, 200, 200);
    }

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            uint8_t tile = grid[row * n + col];
            int cx = col * cell + cell / 2;
            int cy = row * cell + cell / 2;
            int left = col * cell;
            int right = (col + 1) * cell - 1;
            int top = row * cell;
            int bottom = (row + 1) * cell - 1;

            int left_color = v_edges[row * (n + 1) + col];
            int right_color = v_edges[row * (n + 1) + col + 1];
            int top_color = h_edges[row * n + col];
            int bottom_color = h_edges[(row + 1) * n + col];

            uint8_t r, g, b;

            switch (tile) {
                case 0: break;
                case 1: // cross
                    if (left_color > 0) { get_strand_color(left_color, n, r, g, b); png.draw_hline(left, cx, cy, r, g, b); }
                    if (right_color > 0) { get_strand_color(right_color, n, r, g, b); png.draw_hline(cx, right, cy, r, g, b); }
                    if (top_color > 0) { get_strand_color(top_color, n, r, g, b); png.draw_vline(cx, top, cy, r, g, b); }
                    if (bottom_color > 0) { get_strand_color(bottom_color, n, r, g, b); png.draw_vline(cx, cy, bottom, r, g, b); }
                    break;
                case 2: // r-elbow
                    if (bottom_color > 0) {
                        get_strand_color(bottom_color, n, r, g, b);
                        png.draw_vline(cx, cy, bottom, r, g, b);
                        png.draw_hline(cx, right, cy, r, g, b);
                    }
                    break;
                case 3: // j-elbow
                    if (left_color > 0) {
                        get_strand_color(left_color, n, r, g, b);
                        png.draw_hline(left, cx, cy, r, g, b);
                        png.draw_vline(cx, top, cy, r, g, b);
                    }
                    break;
                case 4: // vert
                    if (bottom_color > 0) {
                        get_strand_color(bottom_color, n, r, g, b);
                        png.draw_vline(cx, top, bottom, r, g, b);
                    }
                    break;
                case 5: // horiz
                    if (left_color > 0) {
                        get_strand_color(left_color, n, r, g, b);
                        png.draw_hline(left, right, cy, r, g, b);
                    }
                    break;
            }
        }
    }

    if (!png.write(filename)) {
        fprintf(stderr, "Failed to write %s\n", filename);
    }
}

void render_perm_png(const std::vector<int>& perm, int n, const char* filename) {
    int cell;
    if (n <= 10) cell = 40;
    else if (n <= 20) cell = 30;
    else if (n <= 30) cell = 20;
    else if (n <= 50) cell = 15;
    else cell = 10;
    const int w = n * cell;
    const int h = n * cell;

    PNGWriter png(w, h);

    for (int i = 0; i <= n; i++) {
        for (int x = 0; x < w; x++) png.set_pixel(x, i * cell, 220, 220, 220);
        for (int y = 0; y < h; y++) png.set_pixel(i * cell, y, 220, 220, 220);
    }

    int radius = std::max(2, cell / 4);
    for (int i = 0; i < n; i++) {
        int j = perm[i] - 1;
        int cx = j * cell + cell / 2;
        int cy = i * cell + cell / 2;
        png.draw_dot(cx, cy, radius, 0, 0, 0);
    }

    if (!png.write(filename)) {
        fprintf(stderr, "Failed to write %s\n", filename);
    }
}

#endif // HAS_PNG

// =============================================================================
// Text output (always available)
// =============================================================================

void render_bpd_text(const std::vector<uint8_t>& grid, int n, const std::vector<int>& perm, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to write %s\n", filename);
        return;
    }

    fprintf(fp, "BUMPLESS PIPE DREAM (BPD)\n");
    fprintf(fp, "=========================\n\n");
    fprintf(fp, "Grid size: %d x %d\n", n, n);
    fprintf(fp, "Permutation: (");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d%s", perm[i], (i < n - 1) ? "," : "");
    }
    fprintf(fp, ")\n\n");

    fprintf(fp, "TILE TYPES:\n");
    fprintf(fp, "  0 = blank    (no pipe)\n");
    fprintf(fp, "  1 = cross    (pipes cross)\n");
    fprintf(fp, "  2 = r-elbow  (pipe turns: bottom->right)\n");
    fprintf(fp, "  3 = j-elbow  (pipe turns: left->top)\n");
    fprintf(fp, "  4 = vert     (vertical pipe)\n");
    fprintf(fp, "  5 = horiz    (horizontal pipe)\n\n");

    fprintf(fp, "TILE LIST (row, col, type) [1-indexed]:\n");
    fprintf(fp, "---------------------------------------\n");
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            uint8_t tile = grid[row * n + col];
            const char* name;
            switch (tile) {
                case 0: name = "blank"; break;
                case 1: name = "cross"; break;
                case 2: name = "r-elbow"; break;
                case 3: name = "j-elbow"; break;
                case 4: name = "vert"; break;
                case 5: name = "horiz"; break;
                default: name = "?"; break;
            }
            fprintf(fp, "(%d, %d, %d)  # %s\n", row + 1, col + 1, tile, name);
        }
    }

    fclose(fp);
}

// =============================================================================
// Sampling functions
// =============================================================================

std::string format_number(int64_t n) {
    char buf[32];
    if (n >= 1000000000) snprintf(buf, sizeof(buf), "%.2fG", n / 1e9);
    else if (n >= 1000000) snprintf(buf, sizeof(buf), "%.2fM", n / 1e6);
    else if (n >= 1000) snprintf(buf, sizeof(buf), "%.2fK", n / 1e3);
    else snprintf(buf, sizeof(buf), "%lld", (long long)n);
    return std::string(buf);
}

// Sample one BPD via backward CFTP (Propp-Wilson protocol).
// Returns permutation (empty if exceeded max rounds).
// total_updates_out: set to total number of updates replayed.
std::vector<int> sample_backward_cftp(BPDEngine& engine, std::mt19937& rng,
                                       int& rounds_out, int64_t& total_updates_out) {
    int n = engine.get_n();
    std::uniform_int_distribution<int> pos(1, n - 1);
    std::uniform_int_distribution<int> dir(0, 1);

    int initial_window = std::max(16, 2 * n * n);
    std::vector<BPDUpdate> updates;
    updates.reserve(initial_window * 8);

    // Generate initial window of random updates
    for (int t = 0; t < initial_window; t++)
        updates.push_back({(int8_t)pos(rng), (int8_t)pos(rng), dir(rng) == 0});

    total_updates_out = 0;
    for (int round = 1; round <= 40; round++) {
        total_updates_out += (int64_t)updates.size();

        if (engine.run_cftp_round(updates)) {
            rounds_out = round;
            return engine.compute_perm(engine.get_result());
        }

        // Double: prepend updates.size() new random updates (extend backward in time)
        int extend = (int)updates.size();
        std::vector<BPDUpdate> earlier;
        earlier.reserve(extend + updates.size());
        for (int t = 0; t < extend; t++)
            earlier.push_back({(int8_t)pos(rng), (int8_t)pos(rng), dir(rng) == 0});
        earlier.insert(earlier.end(), updates.begin(), updates.end());
        updates = std::move(earlier);
    }

    rounds_out = 40;
    return {};  // failed to coalesce
}

// =============================================================================
// Batch mode output
// =============================================================================

void write_mathematica_matrix(const std::vector<int64_t>& matrix, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "perm_matrix_n%d_B%d.txt", n, B);

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s for writing\n", filename);
        return;
    }

    fprintf(fp, "{");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "{");
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%lld", (long long)matrix[i * n + j]);
            if (j < n - 1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (i < n - 1) fprintf(fp, ",");
    }
    fprintf(fp, "}\n");

    fclose(fp);
    printf("Output written to: %s\n", filename);
}

void write_mathematica_permutations(const std::vector<std::vector<int>>& perms, int n, int B) {
    char filename[128];
    snprintf(filename, sizeof(filename), "perms_n%d_B%d.txt", n, B);

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s for writing\n", filename);
        return;
    }

    fprintf(fp, "{");
    for (size_t p = 0; p < perms.size(); p++) {
        fprintf(fp, "{");
        for (int i = 0; i < n; i++) {
            fprintf(fp, "%d", perms[p][i]);
            if (i < n - 1) fprintf(fp, ",");
        }
        fprintf(fp, "}");
        if (p < perms.size() - 1) fprintf(fp, ",\n");
    }
    fprintf(fp, "}\n");

    fclose(fp);
    printf("Output written to: %s\n", filename);
}

// =============================================================================
// Main
// =============================================================================

void print_help(const char* prog) {
    printf("BPD CFTP Sampler - experimental backward CFTP for reduced bumpless pipe dreams\n\n");
    printf("Usage:\n");
    printf("  %s <n> [seed]                    Single sample mode\n", prog);
    printf("  %s batch:<n>:<B> [--export]      Batch mode\n\n", prog);
    printf("Note:\n");
    printf("  This program is kept for the paper's CFTP failure diagnostics.\n");
    printf("  Use bpd_mcmc for the recommended sampling workflow.\n\n");
    printf("Arguments:\n");
    printf("  n             Grid size (>= 2)\n");
    printf("  seed          Random seed (default: random, 0 = random)\n");
    printf("  B             Number of samples to generate\n");
    printf("  --export      Export all permutations as Mathematica list\n\n");
    printf("Examples:\n");
    printf("  %s 10              Single n=10 sample\n", prog);
    printf("  %s 10 12345        Single n=10 with seed 12345\n", prog);
    printf("  %s batch:10:1000   Batch 1000 samples at n=10\n", prog);
    printf("  %s batch:25:100 --export   Export 100 permutations\n\n", prog);
    printf("Output (single mode):\n");
#if HAS_PNG
    printf("  bpd_n<N>.png         BPD visualization\n");
    printf("  bpd_n<N>_colored.png Colored by strand\n");
    printf("  perm_n<N>.png        Permutation matrix\n");
#endif
    printf("  bpd_n<N>.txt         Text representation\n\n");
    printf("Output (batch mode):\n");
    printf("  perm_matrix_n<N>_B<B>.txt  Accumulated permutation matrix\n");
    printf("  perms_n<N>_B<B>.txt        All permutations (with --export)\n\n");
    printf("Algorithm: Backward CFTP (Propp-Wilson) with monotone coupling\n");
    printf("  Included as experimental/diagnostic code for the CFTP obstruction.\n");
    printf("  Not the recommended production sampler; see bpd_mcmc.cpp.\n");
    printf("  Uses doubling protocol: window doubles until top/bottom chains coalesce.\n");
    printf("Tile types: 0=blank, 1=cross, 2=r-elbow, 3=j-elbow, 4=vert, 5=horiz\n");
}

int run_single_mode(int n, uint32_t seed_arg) {
    uint32_t seed = (seed_arg == 0) ? std::random_device{}() : seed_arg;
    printf("BPD CFTP sampler (backward CFTP): n=%d, seed=%u\n", n, seed);

    BPDEngine engine(n);
    std::mt19937 rng(seed);

    int rounds;
    int64_t total_updates;
    std::vector<int> perm = sample_backward_cftp(engine, rng, rounds, total_updates);

    if (perm.empty()) {
        printf("Failed to coalesce after 40 doubling rounds.\n");
        return 1;
    }

    printf("Coalesced after %d rounds (%s total updates)\n",
           rounds, format_number(total_updates).c_str());

    const std::vector<uint8_t>& result = engine.get_result();

    // Print permutation
    printf("Permutation: (");
    for (int i = 0; i < n; i++) {
        printf("%d%s", perm[i], (i < n - 1) ? "," : "");
    }
    printf(")\n");

    // Generate output files
    char bpd_txt_file[64];
    snprintf(bpd_txt_file, sizeof(bpd_txt_file), "bpd_n%d.txt", n);
    render_bpd_text(result, n, perm, bpd_txt_file);

#if HAS_PNG
    std::vector<int> h_edges, v_edges;
    compute_edge_colors(result, n, h_edges, v_edges);

    char bpd_file[64], perm_file[64], bpd_colored_file[64];
    snprintf(bpd_file, sizeof(bpd_file), "bpd_n%d.png", n);
    snprintf(perm_file, sizeof(perm_file), "perm_n%d.png", n);
    snprintf(bpd_colored_file, sizeof(bpd_colored_file), "bpd_n%d_colored.png", n);

    render_bpd_png(result, n, bpd_file);
    render_perm_png(perm, n, perm_file);
    render_bpd_colored_png(result, n, h_edges, v_edges, bpd_colored_file);

    printf("Output: %s, %s, %s, %s\n", bpd_file, bpd_txt_file, perm_file, bpd_colored_file);
#else
    printf("Output: %s (PNG disabled - compile with libpng for images)\n", bpd_txt_file);
#endif

    return 0;
}

int run_batch_mode(int n, int B, bool export_perms) {
    int num_threads = omp_get_max_threads();
        printf("BPD CFTP batch sampler (backward CFTP): n=%d, B=%d, threads=%d%s\n",
           n, B, num_threads, export_perms ? ", export mode" : "");

    std::vector<int64_t> sum_matrix(n * n, 0);
    std::vector<std::vector<int>> all_perms;
    if (export_perms) {
        all_perms.resize(B);
    }

    int completed = 0;
    double start_time = omp_get_wtime();
    int64_t total_updates_all = 0;
    bool verbose = (n >= 36);

    #ifdef _OPENMP
    #pragma omp parallel
    {
        BPDEngine engine(n);
        std::random_device rd;
        std::mt19937 rng(rd() ^ omp_get_thread_num());

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < B; b++) {
            int rounds;
            int64_t updates;
            std::vector<int> perm = sample_backward_cftp(engine, rng, rounds, updates);

            #pragma omp critical
            {
                total_updates_all += updates;
                for (int i = 0; i < n; i++) {
                    sum_matrix[i * n + (perm[i] - 1)] += 1;
                }
                if (export_perms) {
                    all_perms[b] = perm;
                }
                completed++;
                bool should_print = verbose || (completed % 10 == 0) || (completed == B);
                if (should_print) {
                    double avg_updates = (double)total_updates_all / completed;
                    printf("\rProgress: %d/%d (avg %.0f updates/sample)   ",
                           completed, B, avg_updates);
                    fflush(stdout);
                }
            }
        }
    }
    #else
    BPDEngine engine(n);
    std::mt19937 rng(std::random_device{}());

    for (int b = 0; b < B; b++) {
        int rounds;
        int64_t updates;
        std::vector<int> perm = sample_backward_cftp(engine, rng, rounds, updates);

        total_updates_all += updates;
        for (int i = 0; i < n; i++) {
            sum_matrix[i * n + (perm[i] - 1)] += 1;
        }
        if (export_perms) {
            all_perms[b] = perm;
        }
        completed++;
        bool should_print = verbose || (completed % 10 == 0) || (completed == B);
        if (should_print) {
            double avg_updates = (double)total_updates_all / completed;
            printf("\rProgress: %d/%d (avg %.0f updates/sample)   ",
                   completed, B, avg_updates);
            fflush(stdout);
        }
    }
    #endif

    printf("\n");

    double elapsed = omp_get_wtime() - start_time;
    printf("Completed %d samples in %.2f seconds (%.1f samples/sec)\n", B, elapsed, B / elapsed);
    printf("Average updates per sample: %.0f\n", (double)total_updates_all / B);

    if (export_perms) {
        write_mathematica_permutations(all_perms, n, B);
    } else {
        write_mathematica_matrix(sum_matrix, n, B);
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_help(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    // Check for batch mode: batch:<n>:<B>
    if (strncmp(argv[1], "batch:", 6) == 0) {
        int n, B;
        if (sscanf(argv[1], "batch:%d:%d", &n, &B) != 2) {
            fprintf(stderr, "Error: Invalid batch format. Use batch:<n>:<B>\n");
            return 1;
        }
        if (n < 2) {
            fprintf(stderr, "Error: n must be >= 2\n");
            return 1;
        }
        if (B < 1) {
            fprintf(stderr, "Error: B must be >= 1\n");
            return 1;
        }

        bool export_perms = false;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--export") == 0) {
                export_perms = true;
            } else {
                fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
                return 1;
            }
        }

        return run_batch_mode(n, B, export_perms);
    }

    // Single sample mode
    int n = atoi(argv[1]);
    if (n < 2) {
        fprintf(stderr, "Error: n must be >= 2\n");
        return 1;
    }

    uint32_t seed_arg = 0;
    if (argc >= 3 && argv[2][0] != '-') {
        seed_arg = (uint32_t)atoi(argv[2]);
    }

    return run_single_mode(n, seed_arg);
}
