// Unified Schubert polynomial evaluator
// Combines all 6 implementations: descent/cotransition/transition x double/exact
//
// CLI:
//   ./schubert [--double|--exact] [--descent|--cotrans|--transition|--best] <permutation>
//   ./schubert max:<n>            - meet-in-middle BFS (n <= 25)
//   ./schubert [--double|--exact] [--descent|--cotrans|--transition|--best] layered_test:<n>
//   ./schubert [--double|--exact] [--descent|--cotrans|--transition|--best] layered:<b1,...>
//
// Defaults: --double --best
//
// Compile:
//   clang++ -O3 -mcpu=apple-m2 -mtune=apple-m2 -flto -ffast-math \
//     -funroll-loops -fomit-frame-pointer -DNDEBUG \
//     -std=c++17 -pthread \
//     -I/opt/homebrew/include -L/opt/homebrew/lib \
//     schubert.cpp -o schubert -lgmp -lgmpxx

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <csignal>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <array>
#include <stack>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <gmpxx.h>
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;
using uint128_t = unsigned __int128;

// =============================================================================
// Configuration and constants
// =============================================================================

constexpr int MAX_N = 25;  // 25 * 5 bits = 125 bits < 128 bits
constexpr int DEFAULT_THREADS = 6;  // M2 has 6 performance cores

// Memoization hard caps - balance cache-friendliness vs recomputation
// For single-formula runs targeting ~12GB RAM:
static const size_t MEMO_HARD_CAP_DOUBLE = 3ULL << 26;  // ~192M entries, ~12GB RAM
static const size_t MEMO_HARD_CAP_EXACT = 1ULL << 27;   // ~128M entries, ~12.8GB RAM

// Global stop flag for clean termination
static std::atomic<bool> g_stop_flag{false};

// Signal handler for Ctrl+C
void signal_handler(int) {
    g_stop_flag.store(true);
}

// =============================================================================
// Utility functions
// =============================================================================

std::string format_number(size_t n) {
    char buf[32];
    if (n >= 1000000000) snprintf(buf, sizeof(buf), "%.2fG", n / 1e9);
    else if (n >= 1000000) snprintf(buf, sizeof(buf), "%.2fM", n / 1e6);
    else if (n >= 1000) snprintf(buf, sizeof(buf), "%.2fK", n / 1e3);
    else snprintf(buf, sizeof(buf), "%zu", n);
    return std::string(buf);
}

inline int compute_length(const std::vector<int>& w) {
    int len = 0;
    int n = (int)w.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (w[i] > w[j]) len++;
        }
    }
    return len;
}

std::vector<int> longest_permutation(int n) {
    std::vector<int> w(n);
    for (int i = 0; i < n; i++) w[i] = n - i;
    return w;
}

void print_permutation(const std::vector<int>& w) {
    printf("[");
    for (size_t i = 0; i < w.size(); i++) {
        printf("%d%s", w[i], i < w.size() - 1 ? "," : "");
    }
    printf("]");
}

std::vector<int> parse_permutation(const char* str) {
    std::vector<int> perm;
    const char* p = str;
    while (*p) {
        while (*p && (*p == ',' || *p == ' ' || *p == '[' || *p == ']')) p++;
        if (*p) {
            perm.push_back(atoi(p));
            while (*p && *p != ',' && *p != ' ' && *p != ']') p++;
        }
    }
    return perm;
}

// Cayley distance: minimum number of transpositions to transform u into v
// Equals n - (number of cycles in u^{-1} * v)
int cayley_distance(const std::vector<int>& u, const std::vector<int>& v) {
    int n = (int)u.size();
    if (n != (int)v.size()) return -1;

    // Compute inverse of u: inv_u[u[i]-1] = i+1
    std::vector<int> inv_u(n);
    for (int i = 0; i < n; i++) {
        inv_u[u[i] - 1] = i + 1;
    }

    // Compose: sigma = inv_u * v, where sigma[i] = inv_u[v[i]-1]
    std::vector<int> sigma(n);
    for (int i = 0; i < n; i++) {
        sigma[i] = inv_u[v[i] - 1];
    }

    // Count cycles in sigma
    std::vector<bool> visited(n, false);
    int num_cycles = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            num_cycles++;
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                j = sigma[j] - 1;  // Follow cycle
            }
        }
    }

    return n - num_cycles;
}

// =============================================================================
// 128-bit permutation key (shared by all implementations)
// =============================================================================

struct PermKey128 {
    uint128_t code;
    int n;

    PermKey128() : code(0), n(0) {}

    PermKey128(const std::vector<int>& w) : n((int)w.size()) {
        code = 0;
        for (int i = 0; i < n; i++) {
            code |= ((uint128_t)(w[i] - 1) << (5 * i));
        }
    }

    bool operator==(const PermKey128& other) const {
        return code == other.code && n == other.n;
    }

    inline int get(int i) const {
        return ((code >> (5 * i)) & 0x1F) + 1;
    }

    inline PermKey128 swap(int a, int b) const {
        uint128_t va = (code >> (5 * a)) & 0x1F;
        uint128_t vb = (code >> (5 * b)) & 0x1F;
        uint128_t new_code = code;
        new_code &= ~((uint128_t)0x1F << (5 * a));
        new_code &= ~((uint128_t)0x1F << (5 * b));
        new_code |= (vb << (5 * a));
        new_code |= (va << (5 * b));
        PermKey128 result;
        result.code = new_code;
        result.n = n;
        return result;
    }

    inline PermKey128 swap_adjacent(int pos) const {
        return swap(pos, pos + 1);
    }

    std::vector<int> decode() const {
        std::vector<int> w(n);
        for (int i = 0; i < n; i++) w[i] = get(i);
        return w;
    }

    inline int length() const {
        int len = 0;
        for (int i = 0; i < n; i++) {
            int wi = get(i);
            for (int j = i + 1; j < n; j++) {
                if (wi > get(j)) len++;
            }
        }
        return len;
    }
};

struct PermKey128Hash {
    size_t operator()(const PermKey128& k) const {
        uint64_t lo = (uint64_t)k.code;
        uint64_t hi = (uint64_t)(k.code >> 64);
        uint64_t x = lo ^ (hi * 0x9e3779b97f4a7c15ULL);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
};

// =============================================================================
// Large permutation key for beam search with n > 25 (up to 100)
// Uses uint8_t array instead of bit-packed representation
// =============================================================================

constexpr int BEAM_MAX_N = 100;

struct PermKeyLarge {
    std::array<uint8_t, BEAM_MAX_N> perm;
    int n;

    PermKeyLarge() : n(0) { perm.fill(0); }

    PermKeyLarge(const std::vector<int>& w) : n((int)w.size()) {
        perm.fill(0);
        for (int i = 0; i < n; i++) perm[i] = (uint8_t)w[i];
    }

    bool operator==(const PermKeyLarge& other) const {
        if (n != other.n) return false;
        for (int i = 0; i < n; i++) {
            if (perm[i] != other.perm[i]) return false;
        }
        return true;
    }

    inline int get(int i) const { return perm[i]; }

    inline PermKeyLarge swap(int a, int b) const {
        PermKeyLarge result = *this;
        std::swap(result.perm[a], result.perm[b]);
        return result;
    }

    inline PermKeyLarge swap_adjacent(int pos) const {
        return swap(pos, pos + 1);
    }

    std::vector<int> decode() const {
        std::vector<int> w(n);
        for (int i = 0; i < n; i++) w[i] = perm[i];
        return w;
    }

    inline int length() const {
        int len = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (perm[i] > perm[j]) len++;
            }
        }
        return len;
    }
};

struct PermKeyLargeHash {
    size_t operator()(const PermKeyLarge& k) const {
        // FNV-1a hash
        size_t h = 14695981039346656037ULL;
        for (int i = 0; i < k.n; i++) {
            h ^= k.perm[i];
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct FrontierEntryLarge {
    PermKeyLarge key;
    double val;

    bool operator<(const FrontierEntryLarge& other) const {
        if (key.n != other.key.n) return key.n < other.key.n;
        for (int i = 0; i < key.n; i++) {
            if (key.perm[i] != other.key.perm[i]) return key.perm[i] < other.key.perm[i];
        }
        return false;
    }
};

struct BeamResultLarge {
    std::unordered_map<PermKeyLarge, double, PermKeyLargeHash> frontier;
    std::vector<std::pair<PermKeyLarge, double>> level_candidates;
    PermKeyLarge best_perm;
    double best_value;
    int best_length;
    size_t total_processed;
};

// =============================================================================
// OPTIMIZATION: Compact 64-bit Packing for N <= 16
// Uses 4 bits per element (values 0-15), fits in single uint64_t
// This halves memory compared to 128-bit keys
// =============================================================================

struct CompactEntry {
    uint64_t code;
    double val;

    bool operator<(const CompactEntry& other) const {
        return code < other.code;
    }
};

// Pack permutation into 64-bit integer (4 bits per element, max N=16)
inline uint64_t pack_perm64(const std::vector<int>& w) {
    uint64_t code = 0;
    for (size_t i = 0; i < w.size(); i++) {
        code |= (uint64_t)(w[i] - 1) << (4 * i);
    }
    return code;
}

inline std::vector<int> unpack_perm64(uint64_t code, int n) {
    std::vector<int> w(n);
    for (int i = 0; i < n; i++) {
        w[i] = ((code >> (4 * i)) & 0xF) + 1;
    }
    return w;
}

inline uint64_t swap_adjacent_64(uint64_t code, int i) {
    uint64_t mask_a = (uint64_t)0xF << (4 * i);
    uint64_t mask_b = (uint64_t)0xF << (4 * (i + 1));
    uint64_t val_a = (code & mask_a) >> (4 * i);
    uint64_t val_b = (code & mask_b) >> (4 * (i + 1));

    uint64_t new_code = code & ~(mask_a | mask_b);
    new_code |= (val_b << (4 * i));
    new_code |= (val_a << (4 * (i + 1)));
    return new_code;
}

// Compute inverse of 64-bit packed permutation
// S_w = S_{w^-1} so we use canonical form min(w, w^-1) for symmetry pruning
inline uint64_t inverse_packed_64(uint64_t code, int n) {
    uint64_t inv = 0;
    for (int i = 0; i < n; i++) {
        int val = (code >> (4 * i)) & 0xF;  // 0-indexed value at position i
        // In inverse, position 'val' holds value 'i'
        inv |= ((uint64_t)i << (4 * val));
    }
    return inv;
}

// Return canonical form: min(w, w^-1) by lexicographic ordering of codes
inline uint64_t canonical_64(uint64_t code, int n) {
    uint64_t inv = inverse_packed_64(code, n);
    return (code <= inv) ? code : inv;
}

// Helper: Compute inverse of a packed permutation
// S_w = S_{w^-1} so we can use canonical form for symmetry pruning
inline PermKey128 inverse_packed(const PermKey128& k) {
    PermKey128 inv;
    inv.n = k.n;
    inv.code = 0;
    for (int i = 0; i < k.n; i++) {
        int val = k.get(i);  // Value at position i (1-based)
        // In inverse, position 'val-1' will hold value 'i+1'
        inv.code |= ((uint128_t)(i) << (5 * (val - 1)));
    }
    return inv;
}

// =============================================================================
// 128-bit helpers for n > 16 (5 bits per element, n <= 25)
// =============================================================================

// Compact entry for 128-bit sort-reduce BFS (for n > 16)
struct CompactEntry128 {
    uint128_t code;
    double val;

    bool operator<(const CompactEntry128& other) const {
        return code < other.code;
    }
};

// Check Bruhat cover going DOWN for 128-bit packed permutation (5 bits per element)
inline bool is_bruhat_cover_down_128(uint128_t code, int a, int b, int n) {
    int wa = (code >> (5 * a)) & 0x1F;
    int wb = (code >> (5 * b)) & 0x1F;
    if (wa <= wb) return false;

    // Early exit: if wa and wb are adjacent values, no intermediate values exist
    if (wa - wb == 1) return true;

    // Early exit: if positions are adjacent, no positions to check
    if (b - a == 1) return true;

    // Build bitmask of forbidden values (strictly between wb and wa)
    uint32_t forbidden = ((1u << wa) - 1) & ~((1u << (wb + 1)) - 1);

    for (int m = a + 1; m < b; m++) {
        int wm = (code >> (5 * m)) & 0x1F;
        if (forbidden & (1u << wm)) return false;
    }
    return true;
}

// Check cotrans condition for going DOWN (128-bit version)
// Values in 128-bit packed code are 0-indexed, so add 1 for cotrans formula
inline bool is_valid_cotrans_down_128(uint128_t sigma_code, int a, int b, int n) {
    for (int i = 0; i < n; i++) {
        int pi_i_raw;  // 0-indexed value from packed code
        if (i == a) pi_i_raw = (sigma_code >> (5 * b)) & 0x1F;
        else if (i == b) pi_i_raw = (sigma_code >> (5 * a)) & 0x1F;
        else pi_i_raw = (sigma_code >> (5 * i)) & 0x1F;

        int pi_i = pi_i_raw + 1;  // Convert to 1-indexed for cotrans formula
        if ((i + 1) + pi_i <= n) {
            return (i == a || i == b);
        }
    }
    return false;
}

// Swap positions a and b in 128-bit packed permutation
inline uint128_t swap_positions_128(uint128_t code, int a, int b) {
    int val_a = (code >> (5 * a)) & 0x1F;
    int val_b = (code >> (5 * b)) & 0x1F;
    code &= ~((uint128_t)0x1F << (5 * a));
    code &= ~((uint128_t)0x1F << (5 * b));
    code |= ((uint128_t)val_b << (5 * a));
    code |= ((uint128_t)val_a << (5 * b));
    return code;
}

// Swap adjacent positions in 128-bit packed permutation
inline uint128_t swap_adjacent_128(uint128_t code, int i) {
    return swap_positions_128(code, i, i + 1);
}

// Pack permutation into 128-bit integer (5 bits per element, max N=25)
inline uint128_t pack_perm128(const std::vector<int>& w) {
    uint128_t code = 0;
    for (size_t i = 0; i < w.size(); i++) {
        code |= (uint128_t)(w[i] - 1) << (5 * i);  // Store 0-indexed
    }
    return code;
}

// Unpack 128-bit integer to permutation
inline std::vector<int> unpack_perm128(uint128_t code, int n) {
    std::vector<int> w(n);
    for (int i = 0; i < n; i++) {
        w[i] = ((code >> (5 * i)) & 0x1F) + 1;  // Convert back to 1-indexed
    }
    return w;
}

// Frontier entry for vectorized BFS (sort-reduce approach)
struct FrontierEntry {
    PermKey128 key;
    double val;

    bool operator<(const FrontierEntry& other) const {
        if (key.n != other.key.n) return key.n < other.key.n;
        return key.code < other.key.code;
    }
};

// Hash for uint128_t (for frontier maps in heuristic beam search)
struct Uint128Hash {
    size_t operator()(uint128_t k) const {
        uint64_t lo = (uint64_t)k;
        uint64_t hi = (uint64_t)(k >> 64);
        uint64_t x = lo ^ (hi * 0x9e3779b97f4a7c15ULL);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
};

// Result from beam search (for heuristic max search)
struct BeamResult {
    std::unordered_map<uint128_t, double, Uint128Hash> frontier;  // key.code -> S_w value
    std::vector<std::pair<uint128_t, double>> level_candidates;   // top candidates from each level
    PermKey128 best_perm;
    double best_value;
    int best_length;
    size_t total_processed;
};

// =============================================================================
// Rational arithmetic (for descent exact)
// =============================================================================

struct Rational {
    __int128 num;
    __int128 den;

    Rational() : num(0), den(1) {}
    Rational(__int128 n) : num(n), den(1) {}
    Rational(__int128 n, __int128 d) : num(n), den(d) { reduce(); }

    static __int128 gcd(__int128 a, __int128 b) {
        if (a < 0) a = -a;
        if (b < 0) b = -b;
        while (b) { __int128 t = b; b = a % b; a = t; }
        return a;
    }

    void reduce() {
        if (den < 0) { num = -num; den = -den; }
        __int128 g = gcd(num, den);
        if (g > 1) { num /= g; den /= g; }
    }

    Rational operator+(const Rational& o) const {
        return Rational(num * o.den + o.num * den, den * o.den);
    }

    Rational operator*(const Rational& o) const {
        __int128 g1 = gcd(num, o.den);
        __int128 g2 = gcd(o.num, den);
        return Rational((num/g1) * (o.num/g2), (den/g2) * (o.den/g1));
    }

    Rational operator/(const Rational& o) const {
        return *this * Rational(o.den, o.num);
    }

    bool is_integer() const { return den == 1; }
    __int128 to_int() const { return num / den; }
    double to_double() const { return (double)num / (double)den; }

    Rational& operator+=(const Rational& o) {
        *this = *this + o;
        return *this;
    }
};

// Compact entry types for exact BFS (64-bit packing, n <= 16)
struct CompactEntryRational {
    uint64_t code;
    Rational val;

    bool operator<(const CompactEntryRational& other) const {
        return code < other.code;
    }
};

struct CompactEntryMpz {
    uint64_t code;
    mpz_class val;

    bool operator<(const CompactEntryMpz& other) const {
        return code < other.code;
    }
};

// Compact entry types for exact BFS (128-bit packing, 16 < n <= 25)
struct CompactEntry128Rational {
    uint128_t code;
    Rational val;

    bool operator<(const CompactEntry128Rational& other) const {
        return code < other.code;
    }
};

struct CompactEntry128Mpz {
    uint128_t code;
    mpz_class val;

    bool operator<(const CompactEntry128Mpz& other) const {
        return code < other.code;
    }
};

void print_int128(__int128 x) {
    if (x == 0) { printf("0"); return; }
    unsigned __int128 ux;
    if (x < 0) { printf("-"); ux = (unsigned __int128)(-(x + 1)) + 1; }
    else { ux = (unsigned __int128)x; }
    char buf[50];
    int i = 0;
    while (ux > 0) {
        buf[i++] = '0' + (int)(ux % 10);
        ux /= 10;
    }
    while (i > 0) putchar(buf[--i]);
}

static std::string int128_to_string(__int128 x) {
    char buf[50];
    int bi = 0;
    bool neg = x < 0;
    unsigned __int128 ux;
    if (neg) { ux = (unsigned __int128)(-(x + 1)) + 1; }
    else { ux = (unsigned __int128)x; }
    if (ux == 0) { buf[bi++] = '0'; }
    else { while (ux > 0) { buf[bi++] = '0' + (int)(ux % 10); ux /= 10; } }
    std::string s;
    if (neg) s += '-';
    while (bi > 0) s += buf[--bi];
    return s;
}

size_t digits_int128(__int128 x) {
    if (x == 0) return 1;
    unsigned __int128 ux;
    if (x < 0) { ux = (unsigned __int128)(-(x + 1)) + 1; }
    else { ux = (unsigned __int128)x; }
    size_t cnt = 0;
    while (ux > 0) { ux /= 10; cnt++; }
    return cnt;
}

// =============================================================================
// Prime helpers for layered product formula
// =============================================================================

constexpr int PRIME_LIMIT = 5000;

const std::vector<int> SMALL_PRIMES = []() {
    std::vector<int> primes;
    std::vector<bool> is_prime(PRIME_LIMIT + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= PRIME_LIMIT; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
            if ((int64_t)i * i <= PRIME_LIMIT) {
                for (int j = i * i; j <= PRIME_LIMIT; j += i) is_prime[j] = false;
            }
        }
    }
    return primes;
}();

const std::array<int, PRIME_LIMIT + 1> PRIME_INDEX = []() {
    std::array<int, PRIME_LIMIT + 1> idx{};
    idx.fill(-1);
    for (size_t i = 0; i < SMALL_PRIMES.size(); i++) {
        idx[SMALL_PRIMES[i]] = static_cast<int>(i);
    }
    return idx;
}();

void factor_small_int(int x, std::vector<long long>& exps, int sign) {
    int rem = x;
    for (int p : SMALL_PRIMES) {
        if (p * p > rem) break;
        if (rem % p == 0) {
            int idx = PRIME_INDEX[p];
            while (rem % p == 0) {
                exps[idx] += sign;
                rem /= p;
            }
        }
    }
    if (rem > 1) {
        int idx = PRIME_INDEX[rem];
        exps[idx] += sign;
    }
}

double log2_from_counts(const std::vector<long long>& counts) {
    double total = 0.0;
    for (size_t i = 0; i < counts.size(); i++) {
        if (counts[i] != 0) {
            total += counts[i] * std::log2(static_cast<double>(SMALL_PRIMES[i]));
        }
    }
    return total;
}

cpp_int pow_cpp_int(int base, long long exp) {
    cpp_int result = 1;
    cpp_int b = base;
    while (exp > 0) {
        if (exp & 1LL) result *= b;
        b *= b;
        exp >>= 1;
    }
    return result;
}

cpp_int product_from_counts_cpp(const std::vector<long long>& counts) {
    cpp_int num = 1;
    cpp_int den = 1;
    for (size_t i = 0; i < counts.size(); i++) {
        long long e = counts[i];
        if (e > 0) num *= pow_cpp_int(SMALL_PRIMES[i], e);
        else if (e < 0) den *= pow_cpp_int(SMALL_PRIMES[i], -e);
    }
    if (den != 1) num /= den;
    return num;
}

mpz_class product_from_counts_mpz(const std::vector<long long>& counts) {
    mpz_class num(1);
    mpz_class temp;
    for (size_t i = 0; i < counts.size(); i++) {
        long long e = counts[i];
        if (e != 0) {
            mpz_pow_ui(temp.get_mpz_t(), mpz_class(SMALL_PRIMES[i]).get_mpz_t(), std::llabs(e));
            if (e > 0) num *= temp;
            else num /= temp;
        }
    }
    return num;
}

double log2_mpz(const mpz_class& x) {
    long exp;
    double mant = mpz_get_d_2exp(&exp, x.get_mpz_t());
    return exp + std::log2(mant);
}

// =============================================================================
// Layered permutation detection and product formula
// =============================================================================

bool detect_layered(const std::vector<int>& w, std::vector<int>& layers) {
    int n = (int)w.size();
    int next_min = 1;
    int i = 0;
    layers.clear();
    while (i < n) {
        int start = i;
        while (i + 1 < n && w[i] - 1 == w[i + 1]) i++;
        int len = i - start + 1;
        int block_max = w[start];
        int block_min = w[i];
        if (block_min != next_min || block_max != next_min + len - 1) {
            layers.clear();
            return false;
        }
        layers.push_back(len);
        next_min += len;
        i++;
    }
    return next_min == n + 1;
}

std::vector<int> build_layered_permutation(const std::vector<int>& layers) {
    int n = 0;
    for (int b : layers) n += b;
    std::vector<int> w;
    int pos = 1;
    for (int b : layers) {
        for (int j = b - 1; j >= 0; j--) w.push_back(pos + j);
        pos += b;
    }
    return w;
}

void accumulate_F_product_exponents(int m, int p, std::vector<long long>& counts) {
    for (int i = 1; i < p; i++) {
        for (int j = i + 1; j <= p; j++) {
            int num = 2 * m + i + j - 1;
            int den = i + j - 1;
            factor_small_int(num, counts, +1);
            factor_small_int(den, counts, -1);
        }
    }
}

struct LayeredResult {
    cpp_int exact_value;
    double log2_value;
};

LayeredResult layered_product(const std::vector<int>& layers) {
    if (layers.empty()) return {cpp_int(1), 0.0};

    int n = 0;
    for (int b : layers) n += b;

    std::vector<long long> counts(SMALL_PRIMES.size(), 0);
    int remaining = n;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
        int b_i = layers[i];
        if (b_i > 0) {
            int m = remaining - b_i;
            accumulate_F_product_exponents(m, b_i, counts);
            remaining = m;
        }
    }

    return {product_from_counts_cpp(counts), log2_from_counts(counts)};
}


// =============================================================================
// DESCENT FORMULA - Double precision
// =============================================================================

// Forward declarations of BFS versions (defined later, after sort-reduce infrastructure)
double schubert_descent_double_bfs(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);
double schubert_cotrans_double_bfs(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);
double schubert_descent_double_bfs_128(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);
double schubert_cotrans_double_bfs_128(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);
Rational schubert_descent_rational_bfs(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);
mpz_class schubert_cotrans_exact_bfs(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);
Rational schubert_descent_rational_bfs_128(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);
mpz_class schubert_cotrans_exact_bfs_128(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet, int num_threads = 0);

double schubert_descent_double(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();

    if (target_ell == 0) return 1.0;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print_time = start_time;

    std::unordered_map<PermKey128, double, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_DOUBLE);

    // Stack-based iterative descent
    struct StackEntry {
        PermKey128 key;
        int ell;
        int state;
        int child_idx;
        double partial_sum;
        int8_t descents[32];  // positions i where w[i] > w[i+1]
        int8_t num_descents;
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.child_idx = 0;
    initial.partial_sum = 0.0;
    initial.num_descents = 0;
    stk.push(initial);

    double final_result = 0.0;
    size_t iterations = 0;
    bool memo_warning = false;

    // Ell history for averaging (time, ell)
    std::deque<std::pair<double, int>> ell_history;
    auto avg_ell = [](const std::deque<std::pair<double, int>>& hist,
                     double now, double window_sec) -> double {
        if (hist.empty()) return -1.0;
        double sum = 0.0;
        int count = 0;
        double cutoff = now - window_sec;
        for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
            if (it->first < cutoff) break;
            sum += it->second;
            count++;
        }
        return count > 0 ? sum / count : -1.0;
    };

    while (!stk.empty()) {
        // Check both stop flag and global Ctrl-C flag
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed)) break;
        iterations++;

        StackEntry& top = stk.top();

        // Time-based progress reporting every 300ms
        if (!quiet && (iterations & 0x3FFF) == 0) {
            auto now = std::chrono::steady_clock::now();
            double since_print = std::chrono::duration<double>(now - last_print_time).count();
            if (since_print >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                int current_ell = top.ell;

                // Record sample and prune old ones
                ell_history.push_back({elapsed, current_ell});
                while (!ell_history.empty() && ell_history.front().first < elapsed - 150.0) {
                    ell_history.pop_front();
                }

                double avg_5s = avg_ell(ell_history, elapsed, 5.0);
                double avg_30s = avg_ell(ell_history, elapsed, 30.0);
                double avg_2m = avg_ell(ell_history, elapsed, 120.0);

                printf("\r  [%.1fs] descent ell=%d (5s:%.1f 30s:%.1f 2m:%.1f) | Memo: %s | %.1f/s   ",
                       elapsed, current_ell, avg_5s, avg_30s, avg_2m,
                       format_number(memo.size()).c_str(), iterations / elapsed);
                fflush(stdout);
                last_print_time = now;
            }
        }

        if (top.state == 0) {
            // Entering: check base case and memo
            if (top.ell == 0) {
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            // Find all descent positions
            top.num_descents = 0;
            for (int i = 0; i < n - 1; i++) {
                if (top.key.get(i) > top.key.get(i + 1)) {
                    top.descents[top.num_descents++] = (int8_t)i;
                }
            }

            top.child_idx = 0;
            top.partial_sum = 0.0;
            top.state = 1;
        }

        if (top.state == 1) {
            // Accumulate previous child's result
            if (top.child_idx > 0) {
                int prev_i = top.descents[top.child_idx - 1];
                top.partial_sum += ((double)(prev_i + 1) / (double)top.ell) * final_result;
            }

            if (top.child_idx < top.num_descents) {
                int i = top.descents[top.child_idx++];
                PermKey128 child = top.key.swap_adjacent(i);

                auto it = memo.find(child);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child_entry;
                    child_entry.key = child;
                    child_entry.ell = top.ell - 1;
                    child_entry.state = 0;
                    child_entry.child_idx = 0;
                    child_entry.partial_sum = 0.0;
                    child_entry.num_descents = 0;
                    stk.push(child_entry);
                }
            } else {
                // All children processed
                double val = top.partial_sum;
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) {
                    memo[top.key] = val;
                } else if (!memo_warning && !quiet) {
                    printf("\n!!! MEMO LIMIT HIT !!!\n");
                    memo_warning = true;
                }
                final_result = val;
                stk.pop();
            }
        }
    }

    bool interrupted = g_stop_flag.load(std::memory_order_relaxed);
    if (!quiet && !stop.load() && !interrupted) {
        printf("\r                                                                              \r");
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        printf("Memo: %s | Iter: %s | %.1f/s | %.2fs\n",
               format_number(memo.size()).c_str(), format_number(iterations).c_str(),
               iterations / elapsed, elapsed);
    } else if (interrupted && !quiet) {
        printf("\n[Interrupted by Ctrl-C]\n");
    }

    return stop.load() || interrupted ? 0.0 : final_result;
}

// Verbose version of descent double with progress atomics for racing mode
double schubert_descent_double_verbose(
    const std::vector<int>& target_w,
    std::atomic<bool>& stop,
    std::atomic<size_t>& progress_iterations,
    std::atomic<size_t>& progress_memo_size,
    std::atomic<int>& progress_ell  // tracks max_ell_on_stack (decreases toward 0)
) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();

    progress_ell.store(target_ell, std::memory_order_relaxed);
    if (target_ell == 0) return 1.0;

    std::unordered_map<PermKey128, double, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_DOUBLE);

    // Stack-based iterative descent
    // State: 0 = entering, 1 = processing children
    struct StackEntry {
        PermKey128 key;
        int ell;
        int state;
        int child_idx;
        double partial_sum;
        int8_t descents[32];  // positions i where w[i] > w[i+1]
        int8_t num_descents;
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.child_idx = 0;
    initial.partial_sum = 0.0;
    initial.num_descents = 0;
    stk.push(initial);

    double final_result = 0.0;
    size_t iterations = 0;

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed)) break;
        iterations++;

        StackEntry& top = stk.top();

        // Update progress every 16384 iterations - show current working depth
        if ((iterations & 0x3FFF) == 0) {
            progress_iterations.store(iterations, std::memory_order_relaxed);
            progress_memo_size.store(memo.size(), std::memory_order_relaxed);
            progress_ell.store(top.ell, std::memory_order_relaxed);
        }

        if (top.state == 0) {
            // Entering: check base case and memo
            if (top.ell == 0) {
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            // Find all descent positions
            top.num_descents = 0;
            for (int i = 0; i < n - 1; i++) {
                if (top.key.get(i) > top.key.get(i + 1)) {
                    top.descents[top.num_descents++] = (int8_t)i;
                }
            }

            top.child_idx = 0;
            top.partial_sum = 0.0;
            top.state = 1;
        }

        if (top.state == 1) {
            // Accumulate previous child's result
            if (top.child_idx > 0) {
                int prev_i = top.descents[top.child_idx - 1];
                top.partial_sum += ((double)(prev_i + 1) / (double)top.ell) * final_result;
            }

            if (top.child_idx < top.num_descents) {
                int i = top.descents[top.child_idx++];
                PermKey128 child = top.key.swap_adjacent(i);

                auto it = memo.find(child);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child_entry;
                    child_entry.key = child;
                    child_entry.ell = top.ell - 1;
                    child_entry.state = 0;
                    child_entry.child_idx = 0;
                    child_entry.partial_sum = 0.0;
                    child_entry.num_descents = 0;
                    stk.push(child_entry);
                }
            } else {
                // All children processed
                double val = top.partial_sum;
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) {
                    memo[top.key] = val;
                }
                final_result = val;
                stk.pop();
            }
        }
    }

    // Final update
    progress_iterations.store(iterations, std::memory_order_relaxed);
    progress_memo_size.store(memo.size(), std::memory_order_relaxed);
    progress_ell.store(0, std::memory_order_relaxed);

    return stop.load() ? 0.0 : final_result;
}

// =============================================================================
// DESCENT FORMULA - Rational (exact)
// =============================================================================

Rational schubert_descent_rational(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet) {
    int n = (int)target_w.size();
    PermKey128 key(target_w);
    int ell = key.length();

    if (ell == 0) return Rational(1);

    std::unordered_map<PermKey128, Rational, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_EXACT);
    auto start_time = std::chrono::steady_clock::now();
    size_t iterations = 0;
    size_t last_report = 0;

    std::function<Rational(PermKey128, int)> recurse = [&](PermKey128 k, int l) -> Rational {
        if (stop.load(std::memory_order_relaxed)) return Rational(0);
        iterations++;

        if (!quiet && iterations - last_report >= 50000) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            printf("\r  [descent/exact] Iter: %s | Memo: %s | %.1f/s   ",
                   format_number(iterations).c_str(), format_number(memo.size()).c_str(),
                   iterations / elapsed);
            fflush(stdout);
            last_report = iterations;
        }

        if (l == 0) {
            if (memo.size() < MEMO_HARD_CAP_EXACT) memo[k] = Rational(1);
            return Rational(1);
        }

        auto it = memo.find(k);
        if (it != memo.end()) return it->second;

        Rational total(0);
        for (int i = 0; i < n - 1; i++) {
            if (k.get(i) > k.get(i + 1)) {
                PermKey128 child = k.swap_adjacent(i);
                Rational child_val = recurse(child, l - 1);
                total = total + (Rational(i + 1) / Rational(l)) * child_val;
            }
        }
        if (memo.size() < MEMO_HARD_CAP_EXACT) memo[k] = total;
        return total;
    };

    Rational result = recurse(key, ell);

    if (!quiet && !stop.load()) {
        printf("\r                                                                    \r");
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        printf("Iterations: %s | Memo: %s | %.1f iter/sec\n",
               format_number(iterations).c_str(), format_number(memo.size()).c_str(),
               iterations / elapsed);
    }

    return result;
}

// Verbose version of descent rational with progress atomics for racing mode
Rational schubert_descent_rational_verbose(
    const std::vector<int>& target_w,
    std::atomic<bool>& stop,
    std::atomic<size_t>& progress_iterations,
    std::atomic<size_t>& progress_memo_size,
    std::atomic<int>& progress_ell  // tracks max_ell_on_stack (decreases toward 0)
) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();

    progress_ell.store(target_ell, std::memory_order_relaxed);
    if (target_ell == 0) return Rational(1);

    std::unordered_map<PermKey128, Rational, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_EXACT);

    // Stack-based iterative descent for progress tracking
    struct StackEntry {
        PermKey128 key;
        int ell;
        int state;
        int child_idx;
        Rational partial_sum;
        int8_t descents[32];
        int8_t num_descents;
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.child_idx = 0;
    initial.partial_sum = Rational(0);
    initial.num_descents = 0;
    stk.push(initial);

    Rational final_result(0);
    size_t iterations = 0;

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed)) break;
        iterations++;

        StackEntry& top = stk.top();

        // Update progress every 16384 iterations - show current working depth
        if ((iterations & 0x3FFF) == 0) {
            progress_iterations.store(iterations, std::memory_order_relaxed);
            progress_memo_size.store(memo.size(), std::memory_order_relaxed);
            progress_ell.store(top.ell, std::memory_order_relaxed);
        }

        if (top.state == 0) {
            // Entering: check base case and memo
            if (top.ell == 0) {
                if (memo.size() < MEMO_HARD_CAP_EXACT) memo[top.key] = Rational(1);
                final_result = Rational(1);
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            // Find all descent positions
            top.num_descents = 0;
            for (int i = 0; i < n - 1; i++) {
                if (top.key.get(i) > top.key.get(i + 1)) {
                    top.descents[top.num_descents++] = (int8_t)i;
                }
            }

            top.child_idx = 0;
            top.partial_sum = Rational(0);
            top.state = 1;
        }

        if (top.state == 1) {
            // Accumulate previous child's result
            if (top.child_idx > 0) {
                int prev_i = top.descents[top.child_idx - 1];
                top.partial_sum = top.partial_sum + (Rational(prev_i + 1) / Rational(top.ell)) * final_result;
            }

            if (top.child_idx < top.num_descents) {
                int i = top.descents[top.child_idx++];
                PermKey128 child = top.key.swap_adjacent(i);

                auto it = memo.find(child);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child_entry;
                    child_entry.key = child;
                    child_entry.ell = top.ell - 1;
                    child_entry.state = 0;
                    child_entry.child_idx = 0;
                    child_entry.partial_sum = Rational(0);
                    child_entry.num_descents = 0;
                    stk.push(child_entry);
                }
            } else {
                // All children processed
                Rational val = top.partial_sum;
                if (memo.size() < MEMO_HARD_CAP_EXACT) {
                    memo[top.key] = val;
                }
                final_result = val;
                stk.pop();
            }
        }
    }

    // Final update
    progress_iterations.store(iterations, std::memory_order_relaxed);
    progress_memo_size.store(memo.size(), std::memory_order_relaxed);
    progress_ell.store(0, std::memory_order_relaxed);

    return stop.load() ? Rational(0) : final_result;
}

// =============================================================================
// COTRANSITION FORMULA - Helpers
// =============================================================================

inline bool is_bruhat_cover(const PermKey128& k, int a, int b) {
    int wa = k.get(a);
    int wb = k.get(b);
    if (wa >= wb) return false;
    for (int m = a + 1; m < b; m++) {
        int wm = k.get(m);
        if (wa < wm && wm < wb) return false;
    }
    return true;
}

inline int find_cotrans_index(const PermKey128& k) {
    int n = k.n;
    for (int i = 0; i < n; i++) {
        int wi = k.get(i);
        if ((i + 1) + wi <= n) return i;
    }
    return -1;
}

inline bool is_w0(const PermKey128& k) {
    for (int i = 0; i < k.n; i++) {
        if (k.get(i) != k.n - i) return false;
    }
    return true;
}

// =============================================================================
// COTRANSITION FORMULA - Double precision
// =============================================================================

double schubert_cotrans_double(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();

    if (is_w0(target_key)) return 1.0;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print_time = start_time;

    std::unordered_map<PermKey128, double, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_DOUBLE);

    struct StackEntry {
        PermKey128 key;
        int ell;  // track length for progress
        int state;
        int cotrans_idx;
        int cover_idx;
        int num_covers;
        double partial_sum;
        std::pair<int8_t, int8_t> covers[64];
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.cotrans_idx = -1;
    initial.cover_idx = 0;
    initial.num_covers = 0;
    initial.partial_sum = 0.0;
    stk.push(initial);

    double final_result = 0.0;
    size_t iterations = 0;
    bool memo_warning = false;

    // Ell history for averaging (time, ell)
    std::deque<std::pair<double, int>> ell_history;
    auto avg_ell = [](const std::deque<std::pair<double, int>>& hist,
                     double now, double window_sec) -> double {
        if (hist.empty()) return -1.0;
        double sum = 0.0;
        int count = 0;
        double cutoff = now - window_sec;
        for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
            if (it->first < cutoff) break;
            sum += it->second;
            count++;
        }
        return count > 0 ? sum / count : -1.0;
    };

    while (!stk.empty()) {
        // Check both stop flag and global Ctrl-C flag
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed)) break;
        iterations++;

        // Time-based progress reporting every 300ms
        if (!quiet && (iterations & 0x3FFF) == 0) {
            auto now = std::chrono::steady_clock::now();
            double since_print = std::chrono::duration<double>(now - last_print_time).count();
            if (since_print >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                int current_ell = stk.top().ell;

                // Record sample and prune old ones
                ell_history.push_back({elapsed, current_ell});
                while (!ell_history.empty() && ell_history.front().first < elapsed - 150.0) {
                    ell_history.pop_front();
                }

                double avg_5s = avg_ell(ell_history, elapsed, 5.0);
                double avg_30s = avg_ell(ell_history, elapsed, 30.0);
                double avg_2m = avg_ell(ell_history, elapsed, 120.0);

                printf("\r  [%.1fs] cotrans ell=%d (5s:%.1f 30s:%.1f 2m:%.1f) | Memo: %s | %.1f/s   ",
                       elapsed, current_ell, avg_5s, avg_30s, avg_2m,
                       format_number(memo.size()).c_str(), iterations / elapsed);
                fflush(stdout);
                last_print_time = now;
            }
        }

        StackEntry& top = stk.top();

        if (top.state == 0) {
            if (is_w0(top.key)) {
                memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            top.cotrans_idx = find_cotrans_index(top.key);
            if (top.cotrans_idx < 0) {
                memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            int ci = top.cotrans_idx;
            int pi_ci = top.key.get(ci);
            top.num_covers = 0;

            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    if (is_bruhat_cover(top.key, a, b)) {
                        int sigma_ci;
                        if (ci == a) sigma_ci = top.key.get(b);
                        else if (ci == b) sigma_ci = top.key.get(a);
                        else sigma_ci = pi_ci;
                        if (sigma_ci != pi_ci) {
                            top.covers[top.num_covers++] = {(int8_t)a, (int8_t)b};
                        }
                    }
                }
            }

            top.cover_idx = 0;
            top.partial_sum = 0.0;
            top.state = 1;
        }

        if (top.state == 1) {
            if (top.cover_idx > 0) top.partial_sum += final_result;

            if (top.cover_idx < top.num_covers) {
                auto [a, b] = top.covers[top.cover_idx];
                PermKey128 cover_key = top.key.swap(a, b);
                top.cover_idx++;

                auto it = memo.find(cover_key);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child;
                    child.key = cover_key;
                    child.ell = top.ell + 1;  // cotrans goes UP in length
                    child.state = 0;
                    child.cotrans_idx = -1;
                    child.cover_idx = 0;
                    child.num_covers = 0;
                    child.partial_sum = 0.0;
                    stk.push(child);
                }
            } else {
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) {
                    memo[top.key] = top.partial_sum;
                } else if (!memo_warning && !quiet) {
                    printf("\n!!! MEMO LIMIT HIT !!!\n");
                    memo_warning = true;
                }
                final_result = top.partial_sum;
                stk.pop();
            }
        }
    }

    bool interrupted = g_stop_flag.load(std::memory_order_relaxed);
    if (!quiet && !stop.load() && !interrupted) {
        printf("\r                                                                              \r");
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        printf("Memo: %s | Iter: %s | %.1f/s | %.2fs\n",
               format_number(memo.size()).c_str(), format_number(iterations).c_str(),
               iterations / elapsed, elapsed);
    } else if (interrupted && !quiet) {
        printf("\n[Interrupted by Ctrl-C]\n");
    }

    return final_result;
}

// =============================================================================
// COTRANSITION FORMULA - Double precision (verbose with progress atomics)
// =============================================================================

double schubert_cotrans_double_verbose(
    const std::vector<int>& target_w,
    std::atomic<bool>& stop,
    std::atomic<size_t>& progress_iterations,
    std::atomic<size_t>& progress_memo_size,
    std::atomic<int>& progress_ell  // tracks min_ell_on_stack (increases toward binom(n,2))
) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();
    int max_possible_ell = n * (n - 1) / 2;

    progress_ell.store(target_ell, std::memory_order_relaxed);
    if (is_w0(target_key)) return 1.0;

    std::unordered_map<PermKey128, double, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_DOUBLE);

    struct StackEntry {
        PermKey128 key;
        int ell;  // track length for progress
        int state;
        int cotrans_idx;
        int cover_idx;
        int num_covers;
        double partial_sum;
        std::pair<int8_t, int8_t> covers[64];
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.cotrans_idx = -1;
    initial.cover_idx = 0;
    initial.num_covers = 0;
    initial.partial_sum = 0.0;
    stk.push(initial);

    double final_result = 0.0;
    size_t iterations = 0;

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed)) break;
        iterations++;

        StackEntry& top = stk.top();

        // Update progress atomics every 16384 iterations - show current working depth
        if ((iterations & 0x3FFF) == 0) {
            progress_iterations.store(iterations, std::memory_order_relaxed);
            progress_memo_size.store(memo.size(), std::memory_order_relaxed);
            progress_ell.store(top.ell, std::memory_order_relaxed);
        }

        if (top.state == 0) {
            if (is_w0(top.key)) {
                memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            top.cotrans_idx = find_cotrans_index(top.key);
            if (top.cotrans_idx < 0) {
                memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            int ci = top.cotrans_idx;
            int pi_ci = top.key.get(ci);
            top.num_covers = 0;

            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    if (is_bruhat_cover(top.key, a, b)) {
                        int sigma_ci;
                        if (ci == a) sigma_ci = top.key.get(b);
                        else if (ci == b) sigma_ci = top.key.get(a);
                        else sigma_ci = pi_ci;
                        if (sigma_ci != pi_ci) {
                            top.covers[top.num_covers++] = {(int8_t)a, (int8_t)b};
                        }
                    }
                }
            }

            top.cover_idx = 0;
            top.partial_sum = 0.0;
            top.state = 1;
        }

        if (top.state == 1) {
            if (top.cover_idx > 0) top.partial_sum += final_result;

            if (top.cover_idx < top.num_covers) {
                auto [a, b] = top.covers[top.cover_idx++];
                PermKey128 child = top.key.swap(a, b);

                auto it = memo.find(child);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child_entry;
                    child_entry.key = child;
                    child_entry.ell = top.ell + 1;  // cotrans goes UP in length
                    child_entry.state = 0;
                    child_entry.cotrans_idx = -1;
                    child_entry.cover_idx = 0;
                    child_entry.num_covers = 0;
                    child_entry.partial_sum = 0.0;
                    stk.push(child_entry);
                }
            } else {
                double val = top.partial_sum;
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) {
                    memo[top.key] = val;
                }
                final_result = val;
                stk.pop();
            }
        }
    }

    // Final update
    progress_iterations.store(iterations, std::memory_order_relaxed);
    progress_memo_size.store(memo.size(), std::memory_order_relaxed);
    progress_ell.store(max_possible_ell, std::memory_order_relaxed);

    return final_result;
}

// =============================================================================
// COTRANSITION FORMULA - Double precision (with external memo for re-use)
// =============================================================================

double schubert_cotrans_double_with_memo(
    const std::vector<int>& target_w,
    std::unordered_map<PermKey128, double, PermKey128Hash>& memo,
    std::atomic<bool>& stop,
    bool quiet
) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);

    if (is_w0(target_key)) return 1.0;

    // Check if already computed
    auto cached_it = memo.find(target_key);
    if (cached_it != memo.end()) return cached_it->second;

    struct StackEntry {
        PermKey128 key;
        int state;
        int cotrans_idx;
        int cover_idx;
        int num_covers;
        double partial_sum;
        std::pair<int8_t, int8_t> covers[64];
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.state = 0;
    initial.cotrans_idx = -1;
    initial.cover_idx = 0;
    initial.num_covers = 0;
    initial.partial_sum = 0.0;
    stk.push(initial);

    double final_result = 0.0;
    bool memo_warning = false;

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed)) break;

        StackEntry& top = stk.top();

        if (top.state == 0) {
            if (is_w0(top.key)) {
                memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            top.cotrans_idx = find_cotrans_index(top.key);
            if (top.cotrans_idx < 0) {
                memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            int ci = top.cotrans_idx;
            int pi_ci = top.key.get(ci);
            top.num_covers = 0;

            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    if (is_bruhat_cover(top.key, a, b)) {
                        int sigma_ci;
                        if (ci == a) sigma_ci = top.key.get(b);
                        else if (ci == b) sigma_ci = top.key.get(a);
                        else sigma_ci = pi_ci;
                        if (sigma_ci != pi_ci) {
                            top.covers[top.num_covers++] = {(int8_t)a, (int8_t)b};
                        }
                    }
                }
            }

            top.cover_idx = 0;
            top.partial_sum = 0.0;
            top.state = 1;
        }

        if (top.state == 1) {
            if (top.cover_idx > 0) top.partial_sum += final_result;

            if (top.cover_idx < top.num_covers) {
                auto [a, b] = top.covers[top.cover_idx];
                PermKey128 cover_key = top.key.swap(a, b);
                top.cover_idx++;

                auto it = memo.find(cover_key);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child;
                    child.key = cover_key;
                    child.state = 0;
                    child.cotrans_idx = -1;
                    child.cover_idx = 0;
                    child.num_covers = 0;
                    child.partial_sum = 0.0;
                    stk.push(child);
                }
            } else {
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) {
                    memo[top.key] = top.partial_sum;
                } else if (!memo_warning && !quiet) {
                    printf("\n!!! SHARED MEMO LIMIT HIT !!!\n");
                    memo_warning = true;
                }
                final_result = top.partial_sum;
                stk.pop();
            }
        }
    }

    return final_result;
}

// =============================================================================
// COTRANSITION FORMULA - GMP exact
// =============================================================================

mpz_class schubert_cotrans_exact(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);

    if (is_w0(target_key)) return 1;

    auto start_time = std::chrono::steady_clock::now();

    std::unordered_map<PermKey128, mpz_class, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_EXACT);

    struct StackEntry {
        PermKey128 key;
        int state;
        int cotrans_idx;
        int cover_idx;
        int num_covers;
        mpz_class partial_sum;
        std::pair<int8_t, int8_t> covers[64];
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.state = 0;
    initial.cotrans_idx = -1;
    initial.cover_idx = 0;
    initial.num_covers = 0;
    initial.partial_sum = 0;
    stk.push(initial);

    mpz_class final_result = 0;
    size_t iterations = 0;
    size_t last_report = 0;
    bool memo_warning = false;

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed)) break;
        iterations++;

        if (!quiet && iterations - last_report >= 100000) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            printf("\r  [cotrans/exact] Iter: %s | Memo: %s | Stack: %s | %.1f/s   ",
                   format_number(iterations).c_str(), format_number(memo.size()).c_str(),
                   format_number(stk.size()).c_str(), iterations / elapsed);
            fflush(stdout);
            last_report = iterations;
        }

        StackEntry& top = stk.top();

        if (top.state == 0) {
            if (is_w0(top.key)) {
                memo[top.key] = 1;
                final_result = 1;
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            top.cotrans_idx = find_cotrans_index(top.key);
            if (top.cotrans_idx < 0) {
                memo[top.key] = 1;
                final_result = 1;
                stk.pop();
                continue;
            }

            int ci = top.cotrans_idx;
            int pi_ci = top.key.get(ci);
            top.num_covers = 0;

            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    if (is_bruhat_cover(top.key, a, b)) {
                        int sigma_ci;
                        if (ci == a) sigma_ci = top.key.get(b);
                        else if (ci == b) sigma_ci = top.key.get(a);
                        else sigma_ci = pi_ci;
                        if (sigma_ci != pi_ci) {
                            top.covers[top.num_covers++] = {(int8_t)a, (int8_t)b};
                        }
                    }
                }
            }

            top.cover_idx = 0;
            top.partial_sum = 0;
            top.state = 1;
        }

        if (top.state == 1) {
            if (top.cover_idx > 0) top.partial_sum += final_result;

            if (top.cover_idx < top.num_covers) {
                auto [a, b] = top.covers[top.cover_idx];
                PermKey128 cover_key = top.key.swap(a, b);
                top.cover_idx++;

                auto it = memo.find(cover_key);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child;
                    child.key = cover_key;
                    child.state = 0;
                    child.cotrans_idx = -1;
                    child.cover_idx = 0;
                    child.num_covers = 0;
                    child.partial_sum = 0;
                    stk.push(child);
                }
            } else {
                if (memo.size() < MEMO_HARD_CAP_EXACT) {
                    memo[top.key] = top.partial_sum;
                } else if (!memo_warning && !quiet) {
                    printf("\n!!! MEMO LIMIT HIT !!!\n");
                    memo_warning = true;
                }
                final_result = top.partial_sum;
                stk.pop();
            }
        }
    }

    if (!quiet && !stop.load()) {
        printf("\r                                                                    \r");
        printf("Memo size: %s entries | Iterations: %s\n",
               format_number(memo.size()).c_str(), format_number(iterations).c_str());
    }

    return final_result;
}

// Verbose version of cotrans exact with progress atomics for racing mode
mpz_class schubert_cotrans_exact_verbose(
    const std::vector<int>& target_w,
    std::atomic<bool>& stop,
    std::atomic<size_t>& progress_iterations,
    std::atomic<size_t>& progress_memo_size,
    std::atomic<int>& progress_ell  // tracks min_ell_on_stack (increases toward binom(n,2))
) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();
    int max_possible_ell = n * (n - 1) / 2;

    progress_ell.store(target_ell, std::memory_order_relaxed);
    if (is_w0(target_key)) return 1;

    std::unordered_map<PermKey128, mpz_class, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_EXACT);

    struct StackEntry {
        PermKey128 key;
        int ell;  // track length for progress
        int state;
        int cotrans_idx;
        int cover_idx;
        int num_covers;
        mpz_class partial_sum;
        std::pair<int8_t, int8_t> covers[64];
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.cotrans_idx = -1;
    initial.cover_idx = 0;
    initial.num_covers = 0;
    initial.partial_sum = 0;
    stk.push(initial);

    mpz_class final_result = 0;
    size_t iterations = 0;

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed)) break;
        iterations++;

        StackEntry& top = stk.top();

        // Update progress every 16384 iterations - show current working depth
        if ((iterations & 0x3FFF) == 0) {
            progress_iterations.store(iterations, std::memory_order_relaxed);
            progress_memo_size.store(memo.size(), std::memory_order_relaxed);
            progress_ell.store(top.ell, std::memory_order_relaxed);
        }

        if (top.state == 0) {
            if (is_w0(top.key)) {
                memo[top.key] = 1;
                final_result = 1;
                stk.pop();
                continue;
            }

            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            top.cotrans_idx = find_cotrans_index(top.key);
            if (top.cotrans_idx < 0) {
                memo[top.key] = 1;
                final_result = 1;
                stk.pop();
                continue;
            }

            int ci = top.cotrans_idx;
            int pi_ci = top.key.get(ci);
            top.num_covers = 0;

            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    if (is_bruhat_cover(top.key, a, b)) {
                        int sigma_ci;
                        if (ci == a) sigma_ci = top.key.get(b);
                        else if (ci == b) sigma_ci = top.key.get(a);
                        else sigma_ci = pi_ci;
                        if (sigma_ci != pi_ci) {
                            top.covers[top.num_covers++] = {(int8_t)a, (int8_t)b};
                        }
                    }
                }
            }

            top.cover_idx = 0;
            top.partial_sum = 0;
            top.state = 1;
        }

        if (top.state == 1) {
            if (top.cover_idx > 0) top.partial_sum += final_result;

            if (top.cover_idx < top.num_covers) {
                auto [a, b] = top.covers[top.cover_idx];
                PermKey128 cover_key = top.key.swap(a, b);
                top.cover_idx++;

                auto it = memo.find(cover_key);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child;
                    child.key = cover_key;
                    child.ell = top.ell + 1;  // cotrans goes UP in Bruhat order
                    child.state = 0;
                    child.cotrans_idx = -1;
                    child.cover_idx = 0;
                    child.num_covers = 0;
                    child.partial_sum = 0;
                    stk.push(child);
                }
            } else {
                if (memo.size() < MEMO_HARD_CAP_EXACT) {
                    memo[top.key] = top.partial_sum;
                }
                final_result = top.partial_sum;
                stk.pop();
            }
        }
    }

    // Final update
    progress_iterations.store(iterations, std::memory_order_relaxed);
    progress_memo_size.store(memo.size(), std::memory_order_relaxed);
    progress_ell.store(max_possible_ell, std::memory_order_relaxed);

    return (stop.load() || g_stop_flag.load()) ? mpz_class(0) : final_result;
}

// =============================================================================
// TRANSITION FORMULA helpers
// =============================================================================

// Find (r, s) for the transition formula.
// r = largest index such that w_r is the "3" in a 132 pattern
//     (i.e., exist i < r < s with w_i < w_s < w_r).
// s = largest index > r such that w_s < w_r and exist i < r with w_i < w_s.
// Returns {-1, -1} if w is dominant (no 132 pattern).
inline std::pair<int, int> find_transition_rs(const PermKey128& key) {
    int n = key.n;
    // Scan r from right to left
    for (int r = n - 2; r >= 1; r--) {
        int wr = key.get(r);
        // Find min value to the left of r
        int min_left = wr + 1;
        for (int i = 0; i < r; i++) {
            int wi = key.get(i);
            if (wi < min_left) min_left = wi;
        }
        // Find the LARGEST s > r with min_left < w_s < w_r
        int best_s = -1;
        for (int s = r + 1; s < n; s++) {
            int ws = key.get(s);
            if (ws < wr && min_left < ws) {
                best_s = s;
            }
        }
        if (best_s >= 0) return {r, best_s};
    }
    return {-1, -1};
}

// =============================================================================
// TRANSITION FORMULA - Double precision
// =============================================================================

double schubert_transition_double(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();

    // Base case: identity or dominant checked below via find_transition_rs
    if (target_ell == 0) return 1.0;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print_time = start_time;

    std::unordered_map<PermKey128, double, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_DOUBLE);

    // Stack-based iterative DFS
    // Transition produces: v (ell-1) and w' terms (same ell as w).
    // All children accumulated into partial_sum with coefficient 1.
    struct StackEntry {
        PermKey128 key;
        int ell;
        int state;
        int child_idx;
        int num_children;
        double partial_sum;
        PermKey128 children[64];  // v plus w' terms
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.child_idx = 0;
    initial.num_children = 0;
    initial.partial_sum = 0.0;
    stk.push(initial);

    double final_result = 0.0;
    size_t iterations = 0;
    bool memo_warning = false;

    // Ell history for averaging (time, ell)
    std::deque<std::pair<double, int>> ell_history;
    auto avg_ell = [](const std::deque<std::pair<double, int>>& hist,
                     double now, double window_sec) -> double {
        if (hist.empty()) return -1.0;
        double sum = 0.0;
        int count = 0;
        double cutoff = now - window_sec;
        for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
            if (it->first < cutoff) break;
            sum += it->second;
            count++;
        }
        return count > 0 ? sum / count : -1.0;
    };

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed)) break;
        iterations++;

        // Time-based progress reporting every 300ms
        if (!quiet && (iterations & 0x3FFF) == 0) {
            auto now = std::chrono::steady_clock::now();
            double since_print = std::chrono::duration<double>(now - last_print_time).count();
            if (since_print >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                int current_ell = stk.top().ell;

                ell_history.push_back({elapsed, current_ell});
                while (!ell_history.empty() && ell_history.front().first < elapsed - 150.0) {
                    ell_history.pop_front();
                }

                double avg_5s = avg_ell(ell_history, elapsed, 5.0);
                double avg_30s = avg_ell(ell_history, elapsed, 30.0);
                double avg_2m = avg_ell(ell_history, elapsed, 120.0);

                printf("\r  [%.1fs] transition ell=%d (5s:%.1f 30s:%.1f 2m:%.1f) | Memo: %s | %.1f/s   ",
                       elapsed, current_ell, avg_5s, avg_30s, avg_2m,
                       format_number(memo.size()).c_str(), iterations / elapsed);
                fflush(stdout);
                last_print_time = now;
            }
        }

        StackEntry& top = stk.top();

        if (top.state == 0) {
            // Check memo first (before expensive find_transition_rs)
            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            // Check dominant (base case): find_transition_rs returns {-1,-1}
            auto [r, s] = find_transition_rs(top.key);
            if (r < 0) {
                // Dominant permutation: Upsilon_w = 1
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) memo[top.key] = 1.0;
                final_result = 1.0;
                stk.pop();
                continue;
            }

            // Compute v = w * t_{rs} (swap positions r and s)
            PermKey128 v_key = top.key.swap(r, s);

            // Collect children: first child is v, then w' = v * t_{ir} for valid i
            top.num_children = 0;
            top.children[top.num_children++] = v_key;

            // Find w' = v * t_{ir} where i < r, v(i) < v(r), and is_bruhat_cover(v, i, r)
            for (int i = 0; i < r; i++) {
                if (v_key.get(i) < v_key.get(r) && is_bruhat_cover(v_key, i, r)) {
                    PermKey128 wprime = v_key.swap(i, r);
                    top.children[top.num_children++] = wprime;
                }
            }

            top.child_idx = 0;
            top.partial_sum = 0.0;
            top.state = 1;
        }

        if (top.state == 1) {
            // Accumulate previous child's result
            if (top.child_idx > 0) {
                top.partial_sum += final_result;
            }

            if (top.child_idx < top.num_children) {
                PermKey128 child_key = top.children[top.child_idx];
                int child_ell = (top.child_idx == 0) ? top.ell - 1 : top.ell;  // v has ell-1, w' has same ell
                top.child_idx++;

                // Inline memo check
                auto it = memo.find(child_key);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child;
                    child.key = child_key;
                    child.ell = child_ell;
                    child.state = 0;
                    child.child_idx = 0;
                    child.num_children = 0;
                    child.partial_sum = 0.0;
                    stk.push(child);
                }
            } else {
                // All children processed
                if (memo.size() < MEMO_HARD_CAP_DOUBLE) {
                    memo[top.key] = top.partial_sum;
                } else if (!memo_warning && !quiet) {
                    printf("\n!!! MEMO LIMIT HIT !!!\n");
                    memo_warning = true;
                }
                final_result = top.partial_sum;
                stk.pop();
            }
        }
    }

    bool interrupted = g_stop_flag.load(std::memory_order_relaxed);
    if (!quiet && !stop.load() && !interrupted) {
        printf("\r                                                                              \r");
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        printf("Memo: %s | Iter: %s | %.1f/s | %.2fs\n",
               format_number(memo.size()).c_str(), format_number(iterations).c_str(),
               iterations / elapsed, elapsed);
    } else if (interrupted && !quiet) {
        printf("\n[Interrupted by Ctrl-C]\n");
    }

    return (stop.load() || g_stop_flag.load()) ? 0.0 : final_result;
}

// =============================================================================
// TRANSITION FORMULA - Exact precision (mpz_class)
// =============================================================================

mpz_class schubert_transition_exact(const std::vector<int>& target_w, std::atomic<bool>& stop, bool quiet) {
    int n = (int)target_w.size();
    PermKey128 target_key(target_w);
    int target_ell = target_key.length();

    // Base case: identity
    if (target_ell == 0) return mpz_class(1);

    auto start_time = std::chrono::steady_clock::now();
    auto last_print_time = start_time;

    std::unordered_map<PermKey128, mpz_class, PermKey128Hash> memo;
    memo.reserve(MEMO_HARD_CAP_EXACT);

    // Stack-based iterative DFS
    struct StackEntry {
        PermKey128 key;
        int ell;
        int state;
        int child_idx;
        int num_children;
        mpz_class partial_sum;
        PermKey128 children[64];  // v plus w' terms
    };

    std::stack<StackEntry> stk;
    StackEntry initial;
    initial.key = target_key;
    initial.ell = target_ell;
    initial.state = 0;
    initial.child_idx = 0;
    initial.num_children = 0;
    initial.partial_sum = 0;
    stk.push(initial);

    mpz_class final_result = 0;
    size_t iterations = 0;
    bool memo_warning = false;

    // Ell history for averaging (time, ell)
    std::deque<std::pair<double, int>> ell_history;
    auto avg_ell = [](const std::deque<std::pair<double, int>>& hist,
                     double now, double window_sec) -> double {
        if (hist.empty()) return -1.0;
        double sum = 0.0;
        int count = 0;
        double cutoff = now - window_sec;
        for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
            if (it->first < cutoff) break;
            sum += it->second;
            count++;
        }
        return count > 0 ? sum / count : -1.0;
    };

    while (!stk.empty()) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed)) break;
        iterations++;

        // Time-based progress reporting every 300ms
        if (!quiet && (iterations & 0x3FFF) == 0) {
            auto now = std::chrono::steady_clock::now();
            double since_print = std::chrono::duration<double>(now - last_print_time).count();
            if (since_print >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                int current_ell = stk.top().ell;

                ell_history.push_back({elapsed, current_ell});
                while (!ell_history.empty() && ell_history.front().first < elapsed - 150.0) {
                    ell_history.pop_front();
                }

                double avg_5s = avg_ell(ell_history, elapsed, 5.0);
                double avg_30s = avg_ell(ell_history, elapsed, 30.0);
                double avg_2m = avg_ell(ell_history, elapsed, 120.0);

                printf("\r  [%.1fs] transition/exact ell=%d (5s:%.1f 30s:%.1f 2m:%.1f) | Memo: %s | %.1f/s   ",
                       elapsed, current_ell, avg_5s, avg_30s, avg_2m,
                       format_number(memo.size()).c_str(), iterations / elapsed);
                fflush(stdout);
                last_print_time = now;
            }
        }

        StackEntry& top = stk.top();

        if (top.state == 0) {
            // Check memo first (before expensive find_transition_rs)
            auto it = memo.find(top.key);
            if (it != memo.end()) {
                final_result = it->second;
                stk.pop();
                continue;
            }

            // Check dominant (base case): find_transition_rs returns {-1,-1}
            auto [r, s] = find_transition_rs(top.key);
            if (r < 0) {
                // Dominant permutation: Upsilon_w = 1
                if (memo.size() < MEMO_HARD_CAP_EXACT) memo[top.key] = mpz_class(1);
                final_result = mpz_class(1);
                stk.pop();
                continue;
            }

            // Compute v = w * t_{rs} (swap positions r and s)
            PermKey128 v_key = top.key.swap(r, s);

            // Collect children: first child is v, then w' = v * t_{ir} for valid i
            top.num_children = 0;
            top.children[top.num_children++] = v_key;

            // Find w' = v * t_{ir} where i < r, v(i) < v(r), and is_bruhat_cover(v, i, r)
            for (int i = 0; i < r; i++) {
                if (v_key.get(i) < v_key.get(r) && is_bruhat_cover(v_key, i, r)) {
                    PermKey128 wprime = v_key.swap(i, r);
                    top.children[top.num_children++] = wprime;
                }
            }

            top.child_idx = 0;
            top.partial_sum = 0;
            top.state = 1;
        }

        if (top.state == 1) {
            // Accumulate previous child's result
            if (top.child_idx > 0) {
                top.partial_sum += final_result;
            }

            if (top.child_idx < top.num_children) {
                PermKey128 child_key = top.children[top.child_idx];
                int child_ell = (top.child_idx == 0) ? top.ell - 1 : top.ell;
                top.child_idx++;

                // Inline memo check
                auto it = memo.find(child_key);
                if (it != memo.end()) {
                    final_result = it->second;
                } else {
                    StackEntry child;
                    child.key = child_key;
                    child.ell = child_ell;
                    child.state = 0;
                    child.child_idx = 0;
                    child.num_children = 0;
                    child.partial_sum = 0;
                    stk.push(child);
                }
            } else {
                // All children processed
                if (memo.size() < MEMO_HARD_CAP_EXACT) {
                    memo[top.key] = top.partial_sum;
                } else if (!memo_warning && !quiet) {
                    printf("\n!!! MEMO LIMIT HIT !!!\n");
                    memo_warning = true;
                }
                final_result = top.partial_sum;
                stk.pop();
            }
        }
    }

    bool interrupted = g_stop_flag.load(std::memory_order_relaxed);
    if (!quiet && !stop.load() && !interrupted) {
        printf("\r                                                                              \r");
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        printf("Memo: %s | Iter: %s | %.1f/s | %.2fs\n",
               format_number(memo.size()).c_str(), format_number(iterations).c_str(),
               iterations / elapsed, elapsed);
    } else if (interrupted && !quiet) {
        printf("\n[Interrupted by Ctrl-C]\n");
    }

    return (stop.load() || g_stop_flag.load()) ? mpz_class(0) : final_result;
}

// =============================================================================
// Enums and racing infrastructure
// =============================================================================

enum class Precision { DOUBLE, EXACT };
enum class Formula { DESCENT, COTRANS, TRANSITION, BEST };
enum class BfsMode { AUTO, BFS, DFS };  // AUTO = BFS for n<=25, DFS otherwise

struct ComputeResult {
    double double_val;
    Rational rational_val;
    mpz_class mpz_val;
    bool is_exact;
    std::string winner;  // "descent" or "cotrans"
    double time_seconds;
};

// Helper: should we use BFS for this n?
static bool use_bfs(int n, BfsMode mode) {
    if (mode == BfsMode::DFS) return false;
    if (mode == BfsMode::BFS) return (n <= 25);  // BFS only works for n<=25
    return (n <= 25);  // AUTO: use BFS when available
}

// Dispatch helpers: select DFS or BFS evaluator based on n and bfs_mode
static double eval_descent_double(const std::vector<int>& w, std::atomic<bool>& stop, bool quiet, BfsMode bfs_mode) {
    int n = (int)w.size();
    if (use_bfs(n, bfs_mode) && n <= 16)
        return schubert_descent_double_bfs(w, stop, quiet);
    if (use_bfs(n, bfs_mode) && n <= 25)
        return schubert_descent_double_bfs_128(w, stop, quiet);
    return schubert_descent_double(w, stop, quiet);
}

static double eval_cotrans_double(const std::vector<int>& w, std::atomic<bool>& stop, bool quiet, BfsMode bfs_mode) {
    int n = (int)w.size();
    if (use_bfs(n, bfs_mode) && n <= 16)
        return schubert_cotrans_double_bfs(w, stop, quiet);
    if (use_bfs(n, bfs_mode) && n <= 25)
        return schubert_cotrans_double_bfs_128(w, stop, quiet);
    return schubert_cotrans_double(w, stop, quiet);
}

static Rational eval_descent_exact(const std::vector<int>& w, std::atomic<bool>& stop, bool quiet, BfsMode bfs_mode) {
    int n = (int)w.size();
    if (use_bfs(n, bfs_mode) && n <= 16)
        return schubert_descent_rational_bfs(w, stop, quiet);
    if (use_bfs(n, bfs_mode) && n <= 25)
        return schubert_descent_rational_bfs_128(w, stop, quiet);
    return schubert_descent_rational(w, stop, quiet);
}

static mpz_class eval_cotrans_exact(const std::vector<int>& w, std::atomic<bool>& stop, bool quiet, BfsMode bfs_mode) {
    int n = (int)w.size();
    if (use_bfs(n, bfs_mode) && n <= 16)
        return schubert_cotrans_exact_bfs(w, stop, quiet);
    if (use_bfs(n, bfs_mode) && n <= 25)
        return schubert_cotrans_exact_bfs_128(w, stop, quiet);
    return schubert_cotrans_exact(w, stop, quiet);
}

// Transition dispatch helpers: DFS only (no BFS variant)
static double eval_transition_double(const std::vector<int>& w, std::atomic<bool>& stop, bool quiet, BfsMode /*bfs_mode*/) {
    return schubert_transition_double(w, stop, quiet);
}

static mpz_class eval_transition_exact(const std::vector<int>& w, std::atomic<bool>& stop, bool quiet, BfsMode /*bfs_mode*/) {
    return schubert_transition_exact(w, stop, quiet);
}

// Race two methods, return first result (with progress monitoring)
ComputeResult race_methods(const std::vector<int>& w, Precision prec, bool quiet, BfsMode bfs_mode = BfsMode::AUTO) {
    ComputeResult result;
    result.is_exact = (prec == Precision::EXACT);

    // Compute length for progress display
    int ell = compute_length(w);
    int n = (int)w.size();
    int max_ell = n * (n - 1) / 2;  // binom(n,2)

    std::atomic<bool> stop1{false}, stop2{false};
    std::atomic<bool> done{false};
    std::mutex mtx;

    // Progress atomics for both threads
    std::atomic<size_t> descent_iter{0}, descent_memo{0};
    std::atomic<size_t> cotrans_iter{0}, cotrans_memo{0};
    std::atomic<int> descent_ell{ell}, cotrans_ell{ell};  // progress: descent goes DOWN, cotrans goes UP
    std::atomic<bool> descent_done{false}, cotrans_done{false};

    auto start_time = std::chrono::steady_clock::now();

    // Progress monitor thread (prints every second when not quiet, handles Ctrl-C)
    std::atomic<bool> monitor_stop{false};
    std::atomic<bool> interrupted{false};
    bool bfs_active = use_bfs(n, bfs_mode);
    std::thread monitor_thread;
    if (!quiet && !bfs_active) {
        // DFS mode: show detailed ell/iter/memo progress
        monitor_thread = std::thread([&, ell, max_ell]() {
            int print_counter = 0;

            // Time-windowed ell tracking for descent and cotrans
            // Store (elapsed_time, ell) samples
            std::deque<std::pair<double, int>> d_ell_history, c_ell_history;

            // Helper to compute average ell over window [now - window_sec, now]
            auto avg_ell = [](const std::deque<std::pair<double, int>>& hist,
                             double now, double window_sec) -> double {
                if (hist.empty()) return -1.0;
                double sum = 0.0;
                int count = 0;
                double cutoff = now - window_sec;
                for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
                    if (it->first < cutoff) break;
                    sum += it->second;
                    count++;
                }
                return count > 0 ? sum / count : -1.0;
            };

            while (!monitor_stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (monitor_stop.load()) break;

                // Check for Ctrl-C
                if (g_stop_flag.load()) {
                    interrupted.store(true);
                    stop1.store(true);
                    stop2.store(true);
                    break;
                }

                // Print progress every 300ms (3 iterations of 100ms)
                if (++print_counter < 3) continue;
                print_counter = 0;

                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();

                size_t d_iter = descent_iter.load();
                size_t d_memo = descent_memo.load();
                int d_ell = descent_ell.load();
                size_t c_iter = cotrans_iter.load();
                size_t c_memo = cotrans_memo.load();
                int c_ell = cotrans_ell.load();
                bool d_done = descent_done.load();
                bool c_done = cotrans_done.load();

                // Record samples for averaging
                d_ell_history.push_back({elapsed, d_ell});
                c_ell_history.push_back({elapsed, c_ell});

                // Prune old samples (keep only last 2.5 minutes = 150 seconds)
                while (!d_ell_history.empty() && d_ell_history.front().first < elapsed - 150.0) {
                    d_ell_history.pop_front();
                }
                while (!c_ell_history.empty() && c_ell_history.front().first < elapsed - 150.0) {
                    c_ell_history.pop_front();
                }

                // Compute averages over 5s, 30s, 2min windows
                double d_avg_5s = avg_ell(d_ell_history, elapsed, 5.0);
                double d_avg_30s = avg_ell(d_ell_history, elapsed, 30.0);
                double d_avg_2m = avg_ell(d_ell_history, elapsed, 120.0);
                double c_avg_5s = avg_ell(c_ell_history, elapsed, 5.0);
                double c_avg_30s = avg_ell(c_ell_history, elapsed, 30.0);
                double c_avg_2m = avg_ell(c_ell_history, elapsed, 120.0);

                // descent: current working ell (jumps around, trending toward 0)
                // cotrans: current working ell (jumps around, trending toward max_ell)
                printf("\r  [%.1fs] D: ell=%d (5s:%.1f 30s:%.1f 2m:%.1f)%s | C: ell=%d (5s:%.1f 30s:%.1f 2m:%.1f)%s   ",
                       elapsed,
                       d_ell, d_avg_5s, d_avg_30s, d_avg_2m,
                       d_done ? " DONE" : "",
                       c_ell, c_avg_5s, c_avg_30s, c_avg_2m,
                       c_done ? " DONE" : "");
                fflush(stdout);
            }
        });
    } else {
        // Quiet mode or BFS mode: only check for Ctrl-C (BFS functions print their own progress)
        monitor_thread = std::thread([&]() {
            while (!monitor_stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (g_stop_flag.load()) {
                    interrupted.store(true);
                    stop1.store(true);
                    stop2.store(true);
                    break;
                }
            }
        });
    }

    if (prec == Precision::DOUBLE) {
        double val1 = 0.0, val2 = 0.0;

        // Halve thread count per BFS task to avoid over-subscription (both run concurrently).
        int race_threads = std::max(1, (DEFAULT_THREADS > 0 ? DEFAULT_THREADS : 4) / 2);
        auto descent_task = [&]() {
            if (use_bfs(n, bfs_mode) && n <= 16) {
                val1 = schubert_descent_double_bfs(w, stop1, quiet, race_threads);
            } else if (use_bfs(n, bfs_mode) && n <= 25) {
                val1 = schubert_descent_double_bfs_128(w, stop1, quiet, race_threads);
            } else {
                val1 = schubert_descent_double_verbose(w, stop1, descent_iter, descent_memo, descent_ell);
            }
            descent_done.store(true);
            std::lock_guard<std::mutex> lock(mtx);
            if (!done.load()) {
                done.store(true);
                stop2.store(true);
            }
        };

        auto cotrans_task = [&]() {
            if (use_bfs(n, bfs_mode) && n <= 16) {
                val2 = schubert_cotrans_double_bfs(w, stop2, quiet, race_threads);
            } else if (use_bfs(n, bfs_mode) && n <= 25) {
                val2 = schubert_cotrans_double_bfs_128(w, stop2, quiet, race_threads);
            } else {
                val2 = schubert_cotrans_double_verbose(w, stop2, cotrans_iter, cotrans_memo, cotrans_ell);
            }
            cotrans_done.store(true);
            std::lock_guard<std::mutex> lock(mtx);
            if (!done.load()) {
                done.store(true);
                stop1.store(true);
            }
        };

        std::thread t1(descent_task);
        std::thread t2(cotrans_task);
        t1.join();
        t2.join();

        auto end_time = std::chrono::steady_clock::now();
        result.time_seconds = std::chrono::duration<double>(end_time - start_time).count();

        // Determine winner based on which wasn't stopped
        if (!stop1.load()) {
            result.winner = "descent";
            result.double_val = val1;
        } else {
            result.winner = "cotrans";
            result.double_val = val2;
        }
    } else {
        // Exact mode: race descent (Rational) vs cotrans (mpz)
        Rational val1;
        mpz_class val2;

        int race_threads = std::max(1, (DEFAULT_THREADS > 0 ? DEFAULT_THREADS : 4) / 2);
        auto descent_task = [&]() {
            if (use_bfs(n, bfs_mode) && n <= 16) {
                val1 = schubert_descent_rational_bfs(w, stop1, quiet, race_threads);
            } else if (use_bfs(n, bfs_mode) && n <= 25) {
                val1 = schubert_descent_rational_bfs_128(w, stop1, quiet, race_threads);
            } else {
                val1 = schubert_descent_rational_verbose(w, stop1, descent_iter, descent_memo, descent_ell);
            }
            descent_done.store(true);
            std::lock_guard<std::mutex> lock(mtx);
            if (!done.load()) {
                done.store(true);
                stop2.store(true);
            }
        };

        auto cotrans_task = [&]() {
            if (use_bfs(n, bfs_mode) && n <= 16) {
                val2 = schubert_cotrans_exact_bfs(w, stop2, quiet, race_threads);
            } else if (use_bfs(n, bfs_mode) && n <= 25) {
                val2 = schubert_cotrans_exact_bfs_128(w, stop2, quiet, race_threads);
            } else {
                val2 = schubert_cotrans_exact_verbose(w, stop2, cotrans_iter, cotrans_memo, cotrans_ell);
            }
            cotrans_done.store(true);
            std::lock_guard<std::mutex> lock(mtx);
            if (!done.load()) {
                done.store(true);
                stop1.store(true);
            }
        };

        std::thread t1(descent_task);
        std::thread t2(cotrans_task);
        t1.join();
        t2.join();

        auto end_time = std::chrono::steady_clock::now();
        result.time_seconds = std::chrono::duration<double>(end_time - start_time).count();

        if (!stop1.load()) {
            result.winner = "descent";
            result.rational_val = val1;
            // Convert to mpz for unified output (__int128 may exceed long long range)
            result.mpz_val = mpz_class(int128_to_string(val1.to_int()));
        } else {
            result.winner = "cotrans";
            result.mpz_val = val2;
        }
    }

    // Stop monitor and print final status
    monitor_stop.store(true);
    monitor_thread.join();

    if (interrupted.load()) {
        printf("\r                                                                                              \r");
        printf("  Interrupted by Ctrl-C\n");
        if (!bfs_active) {
            printf("  Final: descent %s iter/%s memo | cotrans %s iter/%s memo\n",
                   format_number(descent_iter.load()).c_str(), format_number(descent_memo.load()).c_str(),
                   format_number(cotrans_iter.load()).c_str(), format_number(cotrans_memo.load()).c_str());
        }
        result.winner = "interrupted";
    } else if (!quiet && !bfs_active) {
        printf("\r                                                                                              \r");
        printf("  Final: descent %s iter/%s memo | cotrans %s iter/%s memo\n",
               format_number(descent_iter.load()).c_str(), format_number(descent_memo.load()).c_str(),
               format_number(cotrans_iter.load()).c_str(), format_number(cotrans_memo.load()).c_str());
    }

    return result;
}

// =============================================================================
// MPPY layered permutation data
// =============================================================================

struct LayeredData {
    int n;
    std::vector<int> layers;
    double f_n;
};

const LayeredData MPPY_DATA[] = {
    {1, {1}, 0.000000}, {2, {1, 1}, 0.000000}, {3, {1, 2}, 0.111111},
    {4, {1, 3}, 0.145121}, {5, {1, 1, 3}, 0.152294}, {6, {1, 1, 4}, 0.177564},
    {7, {1, 2, 4}, 0.191149}, {8, {1, 2, 5}, 0.206317}, {9, {1, 2, 6}, 0.213824},
    {10, {1, 3, 6}, 0.220771}, {11, {1, 3, 7}, 0.227005}, {12, {1, 3, 8}, 0.229879},
    {13, {1, 1, 3, 8}, 0.233769}, {14, {1, 1, 4, 8}, 0.237048}, {15, {1, 1, 4, 9}, 0.241677},
    {16, {1, 1, 4, 10}, 0.244446}, {17, {1, 2, 4, 10}, 0.246954}, {18, {1, 2, 4, 11}, 0.249509},
    {19, {1, 2, 5, 11}, 0.251966}, {20, {1, 2, 5, 12}, 0.254240}, {21, {1, 2, 5, 13}, 0.255575},
    {22, {1, 2, 6, 13}, 0.257354}, {23, {1, 2, 6, 14}, 0.258685}, {24, {1, 3, 6, 14}, 0.260063},
    {25, {1, 3, 6, 15}, 0.261360},
};

// =============================================================================
// Command functions
// =============================================================================

void cmd_eval(const std::vector<int>& w_orig, Precision prec, Formula formula, bool no_product = false, BfsMode bfs_mode = BfsMode::AUTO) {
    int n_orig = (int)w_orig.size();
    int ell = compute_length(w_orig);

    printf("Computing S_w(1^%d) for w = ", n_orig);
    print_permutation(w_orig);
    printf("\nLength ell(w) = %d\n", ell);
    printf("Precision: %s | Formula: %s\n\n",
           prec == Precision::DOUBLE ? "double" : "exact",
           formula == Formula::DESCENT ? "descent" :
           formula == Formula::COTRANS ? "cotrans" :
           formula == Formula::TRANSITION ? "transition" : "best (race)");

    // Strip trailing fixed points (stability: S_w doesn't change)
    std::vector<int> w = w_orig;
    while (w.size() >= 2 && w.back() == (int)w.size()) {
        w.pop_back();
    }
    int n = (int)w.size();
    if (n < n_orig) {
        printf("Stripped %d trailing fixed point(s): n reduced to %d\n", n_orig - n, n);
    }

    // Check for layered permutation (skip if --no-product)
    std::vector<int> layers;
    if (!no_product && detect_layered(w, layers)) {
        printf("Detected layered permutation; using product formula.\n");
        auto start = std::chrono::high_resolution_clock::now();
        LayeredResult lr = layered_product(layers);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        std::string val_str = lr.exact_value.convert_to<std::string>();
        printf("\nResult: S_w(1^%d) = %s\n", n_orig, val_str.c_str());
        printf("Digits: %zu\n", val_str.size());
        printf("log2(S_w) / n^2 = %.8f\n", lr.log2_value / ((double)n_orig * n_orig));
        printf("Computation time: %.4f seconds\n", elapsed);
        return;
    }

    std::atomic<bool> stop{false};
    auto start = std::chrono::high_resolution_clock::now();

    if (formula == Formula::BEST) {
        ComputeResult res = race_methods(w, prec, false, bfs_mode);
        printf("\nWinner: %s (finished first)\n", res.winner.c_str());
        if (prec == Precision::DOUBLE) {
            printf("Result: S_w(1^%d) = %.0f\n", n_orig, res.double_val);
            printf("log2(S_w) / n^2 = %.8f\n", log2(res.double_val) / ((double)n_orig * n_orig));
        } else {
            gmp_printf("Result: S_w(1^%d) = %Zd\n", n_orig, res.mpz_val.get_mpz_t());
            printf("log2(S_w) / n^2 = %.8f\n", log2_mpz(res.mpz_val) / ((double)n_orig * n_orig));
        }
        printf("Computation time: %.4f seconds\n", res.time_seconds);
    } else if (formula == Formula::DESCENT) {
        if (prec == Precision::DOUBLE) {
            double result = eval_descent_double(w, stop, false, bfs_mode);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            printf("\nResult: S_w(1^%d) = %.0f\n", n_orig, result);
            printf("log2(S_w) / n^2 = %.8f\n", log2(result) / ((double)n_orig * n_orig));
            printf("Computation time: %.4f seconds\n", elapsed);
        } else {
            Rational result = eval_descent_exact(w, stop, false, bfs_mode);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            printf("\nResult: S_w(1^%d) = ", n_orig);
            print_int128(result.to_int());
            printf("\n");
            printf("log2(S_w) / n^2 = %.8f\n", log2(result.to_double()) / ((double)n_orig * n_orig));
            printf("Computation time: %.4f seconds\n", elapsed);
        }
    } else if (formula == Formula::TRANSITION) {
        if (prec == Precision::DOUBLE) {
            double result = eval_transition_double(w, stop, false, bfs_mode);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            printf("\nResult: S_w(1^%d) = %.0f\n", n_orig, result);
            printf("log2(S_w) / n^2 = %.8f\n", log2(result) / ((double)n_orig * n_orig));
            printf("Computation time: %.4f seconds\n", elapsed);
        } else {
            mpz_class result = eval_transition_exact(w, stop, false, bfs_mode);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            gmp_printf("\nResult: S_w(1^%d) = %Zd\n", n_orig, result.get_mpz_t());
            printf("log2(S_w) / n^2 = %.8f\n", log2_mpz(result) / ((double)n_orig * n_orig));
            printf("Computation time: %.4f seconds\n", elapsed);
        }
    } else {  // COTRANS
        if (prec == Precision::DOUBLE) {
            double result = eval_cotrans_double(w, stop, false, bfs_mode);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            printf("\nResult: S_w(1^%d) = %.0f\n", n_orig, result);
            printf("log2(S_w) / n^2 = %.8f\n", log2(result) / ((double)n_orig * n_orig));
            printf("Computation time: %.4f seconds\n", elapsed);
        } else {
            mpz_class result = eval_cotrans_exact(w, stop, false, bfs_mode);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            gmp_printf("\nResult: S_w(1^%d) = %Zd\n", n_orig, result.get_mpz_t());
            printf("log2(S_w) / n^2 = %.8f\n", log2_mpz(result) / ((double)n_orig * n_orig));
            printf("Computation time: %.4f seconds\n", elapsed);
        }
    }
}

uint64_t perm_to_index(const std::vector<int>& perm) {
    int n = (int)perm.size();
    uint64_t index = 0;
    std::vector<uint64_t> factorials(n);
    factorials[0] = 1;
    for (int i = 1; i < n; i++) factorials[i] = factorials[i-1] * i;

    std::vector<bool> used(n + 1, false);
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 1; j < perm[i]; j++) {
            if (!used[j]) count++;
        }
        index += count * factorials[n - 1 - i];
        used[perm[i]] = true;
    }
    return index;
}

std::vector<int> index_to_perm(uint64_t index, int n) {
    std::vector<int> perm(n);
    std::vector<bool> used(n + 1, false);
    std::vector<uint64_t> factorials(n);
    factorials[0] = 1;
    for (int i = 1; i < n; i++) factorials[i] = factorials[i-1] * i;

    for (int i = 0; i < n; i++) {
        uint64_t fact = factorials[n - 1 - i];
        int count = index / fact;
        index %= fact;
        int elem = 0;
        for (int j = 1; j <= n; j++) {
            if (!used[j]) {
                if (count == 0) { elem = j; break; }
                count--;
            }
        }
        perm[i] = elem;
        used[elem] = true;
    }
    return perm;
}

// Helper: find Bruhat covers of a permutation (positions (a,b) where swapping increases length)
// Returns pairs where perm[a] < perm[b] and no intermediate value exists
inline void find_bruhat_covers_up(const std::vector<int>& perm, int n,
                                   std::vector<std::pair<int,int>>& covers) {
    covers.clear();
    for (int a = 0; a < n; a++) {
        for (int b = a + 1; b < n; b++) {
            if (perm[a] < perm[b]) {
                // Check if this is a cover (no intermediate values)
                bool is_cover = true;
                for (int m = a + 1; m < b && is_cover; m++) {
                    if (perm[a] < perm[m] && perm[m] < perm[b]) {
                        is_cover = false;
                    }
                }
                if (is_cover) {
                    covers.push_back({a, b});
                }
            }
        }
    }
}

// Helper: find cotrans index for a permutation (minimum i where (i+1) + perm[i] <= n)
inline int find_cotrans_index_vec(const std::vector<int>& perm, int n) {
    for (int i = 0; i < n; i++) {
        if ((i + 1) + perm[i] <= n) return i;
    }
    return -1;
}

// =============================================================================
// OPTIMIZED MAX SEARCH - Meet-in-the-Middle with Vectorized BFS
// Strategies:
// 1. Descent BFS from identity (up) + Cotrans BFS from w0 (down)
// 2. Bit-packed operations only (no std::vector allocs in hot loop)
// 3. Vectorized Frontier (Sort + Dedup) instead of HashMap
// 4. Inner-loop parallelism for each BFS direction
// =============================================================================

// Helper: Check if swapping (a,b) in sigma is a valid cotrans transition to pi
// Returns true if the cotrans index of pi is either a or b
inline bool is_valid_cotrans_down(const PermKey128& sigma, int a, int b, int n) {
    // After swap, pi[a] = sigma[b], pi[b] = sigma[a]
    // Check cotrans index of the resulting pi
    for (int i = 0; i < n; i++) {
        int pi_i;
        if (i == a) pi_i = sigma.get(b);
        else if (i == b) pi_i = sigma.get(a);
        else pi_i = sigma.get(i);

        if ((i + 1) + pi_i <= n) {
            // i is the cotrans index of pi
            return (i == a || i == b);
        }
    }
    return false;  // pi is w0, always valid
}

// Helper: Check if (a,b) is a Bruhat cover going DOWN (sigma[a] > sigma[b], no intermediate)
inline bool is_bruhat_cover_down(const PermKey128& k, int a, int b) {
    int wa = k.get(a);
    int wb = k.get(b);
    if (wa <= wb) return false;
    for (int m = a + 1; m < b; m++) {
        int wm = k.get(m);
        if (wb < wm && wm < wa) return false;
    }
    return true;
}


// =============================================================================
// OPTIMIZED MAX SEARCH - Meet-in-the-middle with 64-bit packing (N <= 16)
// Descent UP from identity, Cotrans DOWN from w0, meet when levels cross
// Uses sort-reduce BFS for memory efficiency
// =============================================================================

// Helper: expand descent layer UP (ascent positions)
// Uses dynamic load balancing with atomic counter for irregular workloads
static void expand_descent_layer(const std::vector<CompactEntry>& current,
                                  std::vector<CompactEntry>& next, int n, int num_threads) {
    std::vector<std::vector<CompactEntry>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 64;  // Process in batches to reduce atomic contention

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / num_threads);
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint64_t p_code = current[k].code;
                    double p_val = current[k].val;
                    for (int i = 0; i < n - 1; i++) {
                        int val_i = (p_code >> (4 * i)) & 0xF;
                        int val_next = (p_code >> (4 * (i + 1))) & 0xF;
                        if (val_i < val_next) {  // Ascent: can go UP
                            uint64_t child_code = swap_adjacent_64(p_code, i);
                            buffers[t].push_back({child_code, p_val * (i + 1)});
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry>().swap(b);
    }
}

// Helper: swap positions a and b in 64-bit packed permutation
inline uint64_t swap_positions_64(uint64_t code, int a, int b) {
    int val_a = (code >> (4 * a)) & 0xF;
    int val_b = (code >> (4 * b)) & 0xF;
    // Clear both positions and set swapped values
    code &= ~((uint64_t)0xF << (4 * a));
    code &= ~((uint64_t)0xF << (4 * b));
    code |= ((uint64_t)val_b << (4 * a));
    code |= ((uint64_t)val_a << (4 * b));
    return code;
}

// Helper: check Bruhat cover going DOWN (sigma covers child where sigma[a] > sigma[b])
// Optimized with early-outs and bitmask approach for intermediate values
inline bool is_bruhat_cover_down_64(uint64_t code, int a, int b, int n) {
    int wa = (code >> (4 * a)) & 0xF;
    int wb = (code >> (4 * b)) & 0xF;
    if (wa <= wb) return false;

    // Early exit: if wa and wb are adjacent values, no intermediate values exist
    if (wa - wb == 1) return true;

    // Early exit: if positions are adjacent, no positions to check
    if (b - a == 1) return true;

    // Build bitmask of forbidden values (strictly between wb and wa)
    // For n<=16, values are 0-15, so 16-bit mask suffices
    uint16_t forbidden = ((1u << wa) - 1) & ~((1u << (wb + 1)) - 1);
    // forbidden has bits set for values in (wb, wa)

    // Check each intermediate position
    for (int m = a + 1; m < b; m++) {
        int wm = (code >> (4 * m)) & 0xF;
        if (forbidden & (1u << wm)) return false;
    }
    return true;
}

// Helper: check cotrans condition for going DOWN
// After swapping (a,b) in sigma to get child pi, check if cotrans_index of pi is a or b
// NOTE: Values in packed code are 0-indexed (stored as w[i]-1), so add 1 for cotrans formula
inline bool is_valid_cotrans_down_64(uint64_t sigma_code, int a, int b, int n) {
    // Compute cotrans index of the child (after swap)
    for (int i = 0; i < n; i++) {
        int pi_i_raw;  // 0-indexed value from packed code
        if (i == a) pi_i_raw = (sigma_code >> (4 * b)) & 0xF;
        else if (i == b) pi_i_raw = (sigma_code >> (4 * a)) & 0xF;
        else pi_i_raw = (sigma_code >> (4 * i)) & 0xF;

        int pi_i = pi_i_raw + 1;  // Convert to 1-indexed for cotrans formula
        if ((i + 1) + pi_i <= n) {
            // i is the cotrans index of pi
            return (i == a || i == b);
        }
    }
    return false;  // pi is w0, always valid (shouldn't happen going DOWN)
}

// Helper: expand cotrans layer DOWN using proper Bruhat covers and cotrans condition
// Uses dynamic load balancing with atomic counter for irregular workloads
static void expand_cotrans_layer(const std::vector<CompactEntry>& current,
                                  std::vector<CompactEntry>& next, int n, int num_threads) {
    std::vector<std::vector<CompactEntry>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 64;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n * n / (2 * num_threads));
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint64_t sigma_code = current[k].code;
                    double sigma_val = current[k].val;

                    // Check all pairs (a,b) for Bruhat covers going DOWN
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover_down_64(sigma_code, a, b, n)) {
                                if (is_valid_cotrans_down_64(sigma_code, a, b, n)) {
                                    uint64_t child_code = swap_positions_64(sigma_code, a, b);
                                    // Cotrans: S_child gets contribution S_sigma (no coefficient)
                                    buffers[t].push_back({child_code, sigma_val});
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry>().swap(b);
    }
}

// Radix sort for CompactEntry by 64-bit code (LSD, 8-bit digits)
// O(n) vs O(n log n) for std::sort - significant speedup for large frontiers
static void radix_sort_layer(std::vector<CompactEntry>& layer) {
    if (layer.size() <= 1) return;

    std::vector<CompactEntry> temp(layer.size());
    constexpr int RADIX_BITS = 8;
    constexpr int RADIX_SIZE = 1 << RADIX_BITS;
    constexpr uint64_t RADIX_MASK = RADIX_SIZE - 1;

    // Process 8 passes (64 bits / 8 bits per pass)
    for (int shift = 0; shift < 64; shift += RADIX_BITS) {
        size_t count[RADIX_SIZE] = {0};

        // Count occurrences
        for (const auto& e : layer) {
            size_t digit = (e.code >> shift) & RADIX_MASK;
            count[digit]++;
        }

        // Compute prefix sums
        size_t total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            size_t c = count[i];
            count[i] = total;
            total += c;
        }

        // Scatter to temp
        for (const auto& e : layer) {
            size_t digit = (e.code >> shift) & RADIX_MASK;
            temp[count[digit]++] = e;
        }

        std::swap(layer, temp);
    }
}

// Helper: sort-reduce a layer (sort by code, sum duplicates)
static void sort_reduce_layer(std::vector<CompactEntry>& layer) {
    if (layer.empty()) return;

    std::sort(layer.begin(), layer.end());

    size_t write = 0;
    for (size_t read = 1; read < layer.size(); read++) {
        if (layer[read].code == layer[write].code) {
            layer[write].val += layer[read].val;
        } else {
            write++;
            layer[write] = layer[read];
        }
    }
    layer.resize(write + 1);
}

// Sort-reduce for exact Rational BFS layers
static void sort_reduce_layer_rational(std::vector<CompactEntryRational>& layer) {
    if (layer.empty()) return;

    std::sort(layer.begin(), layer.end());

    size_t write = 0;
    for (size_t read = 1; read < layer.size(); read++) {
        if (layer[read].code == layer[write].code) {
            layer[write].val += layer[read].val;
        } else {
            write++;
            layer[write] = std::move(layer[read]);
        }
    }
    layer.resize(write + 1);
}

// Sort-reduce for exact mpz_class BFS layers
static void sort_reduce_layer_mpz(std::vector<CompactEntryMpz>& layer) {
    if (layer.empty()) return;

    std::sort(layer.begin(), layer.end());

    size_t write = 0;
    for (size_t read = 1; read < layer.size(); read++) {
        if (layer[read].code == layer[write].code) {
            layer[write].val += layer[read].val;
        } else {
            write++;
            layer[write] = std::move(layer[read]);
        }
    }
    layer.resize(write + 1);
}

// =============================================================================
// BFS SORT-REDUCE DESCENT EVALUATOR (N <= 16)
// Evaluates S_w(1^n) by BFS from target DOWN to identity
// Uses 64-bit packing, multi-threaded expansion, and sort-reduce dedup
//
// Algorithm: Track unnormalized values V_w = ell! * S_w
// Recurrence: V_w = sum_{descents i} (i+1) * V_{w*s_i}
// Final: S_w = V_identity / ell!
// =============================================================================

// Helper: expand descent layer DOWN (single-threaded for small frontiers)
static void expand_descent_layer_down_st(const std::vector<CompactEntry>& current,
                                          std::vector<CompactEntry>& next, int n) {
    next.reserve(current.size() * (n / 2));
    for (size_t k = 0; k < current.size(); k++) {
        uint64_t p_code = current[k].code;
        double p_val = current[k].val;
        for (int i = 0; i < n - 1; i++) {
            int val_i = (p_code >> (4 * i)) & 0xF;
            int val_next = (p_code >> (4 * (i + 1))) & 0xF;
            if (val_i > val_next) {
                next.push_back({swap_adjacent_64(p_code, i), p_val * (i + 1)});
            }
        }
    }
}

// Helper: expand descent layer DOWN (multi-threaded for large frontiers)
static void expand_descent_layer_down_mt(const std::vector<CompactEntry>& current,
                                          std::vector<CompactEntry>& next, int n,
                                          int num_threads) {
    std::vector<std::vector<CompactEntry>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 64;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / (2 * num_threads));
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint64_t p_code = current[k].code;
                    double p_val = current[k].val;
                    for (int i = 0; i < n - 1; i++) {
                        int val_i = (p_code >> (4 * i)) & 0xF;
                        int val_next = (p_code >> (4 * (i + 1))) & 0xF;
                        if (val_i > val_next) {
                            buffers[t].push_back({swap_adjacent_64(p_code, i), p_val * (i + 1)});
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry>().swap(b);
    }
}

// Radix sort + reduce in one pass
static void radix_sort_reduce_layer(std::vector<CompactEntry>& layer) {
    if (layer.size() <= 1) return;
    radix_sort_layer(layer);
    size_t write = 0;
    for (size_t read = 1; read < layer.size(); read++) {
        if (layer[read].code == layer[write].code) {
            layer[write].val += layer[read].val;
        } else {
            write++;
            layer[write] = layer[read];
        }
    }
    layer.resize(write + 1);
}

// BFS sort-reduce descent evaluator for n <= 16
// Starts from target, descends level by level to identity
double schubert_descent_double_bfs(const std::vector<int>& target_w,
                                    std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    if (ell == 0) return 1.0;

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntry> layer;
    layer.push_back({pack_perm64(target_w), 1.0});

    int level = ell;

    while (level > 0) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return 0.0;

        // Progress reporting (throttled to every 300ms)
        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS descent level %d/%d | frontier: %s   ",
                       elapsed, level, ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        // Expand down one level
        std::vector<CompactEntry> next;
        if (layer.size() < 1000) {
            expand_descent_layer_down_st(layer, next, n);
        } else {
            expand_descent_layer_down_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntry>().swap(layer);

        // Sort-reduce: radix sort for large layers, std::sort for small
        if (next.size() >= 50000) {
            radix_sort_reduce_layer(next);
        } else {
            sort_reduce_layer(next);
        }

        // Normalize incrementally: divide by current level to avoid ell! overflow
        // Total effect across all levels: divide by ell * (ell-1) * ... * 1 = ell!
        double inv_level = 1.0 / level;
        for (auto& e : next) e.val *= inv_level;

        layer = std::move(next);
        level--;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS descent: %d levels in %.2fs                                    \n", ell, elapsed);
    }

    // At level 0, identity holds the normalized value (already divided by ell!)
    double val = 0.0;
    for (const auto& e : layer) val += e.val;
    return val;
}

// =============================================================================
// BFS SORT-REDUCE DESCENT EVALUATOR — EXACT / RATIONAL (N <= 16)
// Same BFS pattern as double version but with Rational arithmetic
// =============================================================================

static void expand_descent_layer_down_rational_st(const std::vector<CompactEntryRational>& current,
                                                   std::vector<CompactEntryRational>& next, int n) {
    next.reserve(current.size() * (n / 2));
    for (size_t k = 0; k < current.size(); k++) {
        uint64_t p_code = current[k].code;
        const Rational& p_val = current[k].val;
        for (int i = 0; i < n - 1; i++) {
            int val_i = (p_code >> (4 * i)) & 0xF;
            int val_next = (p_code >> (4 * (i + 1))) & 0xF;
            if (val_i > val_next) {
                next.push_back({swap_adjacent_64(p_code, i), p_val * Rational(i + 1)});
            }
        }
    }
}

static void expand_descent_layer_down_rational_mt(const std::vector<CompactEntryRational>& current,
                                                   std::vector<CompactEntryRational>& next, int n,
                                                   int num_threads) {
    std::vector<std::vector<CompactEntryRational>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 64;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / (2 * num_threads));
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint64_t p_code = current[k].code;
                    const Rational& p_val = current[k].val;
                    for (int i = 0; i < n - 1; i++) {
                        int val_i = (p_code >> (4 * i)) & 0xF;
                        int val_next = (p_code >> (4 * (i + 1))) & 0xF;
                        if (val_i > val_next) {
                            buffers[t].push_back({swap_adjacent_64(p_code, i), p_val * Rational(i + 1)});
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
        std::vector<CompactEntryRational>().swap(b);
    }
}

Rational schubert_descent_rational_bfs(const std::vector<int>& target_w,
                                        std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    if (ell == 0) return Rational(1);

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntryRational> layer;
    layer.push_back({pack_perm64(target_w), Rational(1)});

    int level = ell;

    while (level > 0) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return Rational(0);

        // Progress reporting (throttled to every 300ms)
        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS descent/exact level %d/%d | frontier: %s   ",
                       elapsed, level, ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        // Expand down one level
        std::vector<CompactEntryRational> next;
        if (layer.size() < 1000) {
            expand_descent_layer_down_rational_st(layer, next, n);
        } else {
            expand_descent_layer_down_rational_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntryRational>().swap(layer);

        // Sort-reduce (std::sort only for exact types)
        sort_reduce_layer_rational(next);

        // Normalize: divide by current level
        Rational inv_level = Rational(1, level);
        for (auto& e : next) e.val = e.val * inv_level;

        layer = std::move(next);
        level--;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS descent/exact: %d levels in %.2fs                                    \n", ell, elapsed);
    }

    // At level 0, identity holds the normalized value
    Rational val(0);
    for (const auto& e : layer) val = val + e.val;
    return val;
}

// =============================================================================
// BFS SORT-REDUCE COTRANS EVALUATOR (N <= 16)
// Evaluates S_w(1^n) by BFS from target UP to w0
// Uses 64-bit packing, multi-threaded expansion, and sort-reduce dedup
//
// Cotrans recurrence: S_pi = sum_{valid covers sigma} S_sigma
// No coefficients (unlike descent), base case S_w0 = 1
// =============================================================================

// Helper: check Bruhat cover going UP (pi covers sigma where pi[a] < pi[b])
inline bool is_bruhat_cover_up_64(uint64_t code, int a, int b, int n) {
    int wa = (code >> (4 * a)) & 0xF;
    int wb = (code >> (4 * b)) & 0xF;
    if (wa >= wb) return false;  // Need pi(a) < pi(b) for going UP

    if (wb - wa == 1) return true;
    if (b - a == 1) return true;

    uint16_t forbidden = ((1u << wb) - 1) & ~((1u << (wa + 1)) - 1);
    for (int m = a + 1; m < b; m++) {
        int wm = (code >> (4 * m)) & 0xF;
        if (forbidden & (1u << wm)) return false;
    }
    return true;
}

// Helper: check cotrans condition for going UP
// pi's cotrans index ci must be a or b (the swapped positions)
inline bool is_valid_cotrans_up_64(uint64_t pi_code, int a, int b, int n) {
    // Find cotrans index of pi
    for (int i = 0; i < n; i++) {
        int pi_i = ((pi_code >> (4 * i)) & 0xF) + 1;  // 1-indexed
        if ((i + 1) + pi_i <= n) {
            return (i == a || i == b);
        }
    }
    return false;  // pi is w0 (shouldn't happen, we check before expanding)
}

// Helper: expand cotrans layer UP (single-threaded)
static void expand_cotrans_layer_up_st(const std::vector<CompactEntry>& current,
                                        std::vector<CompactEntry>& next, int n) {
    next.reserve(current.size() * n);
    for (size_t k = 0; k < current.size(); k++) {
        uint64_t pi_code = current[k].code;
        double pi_val = current[k].val;
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (is_bruhat_cover_up_64(pi_code, a, b, n)) {
                    if (is_valid_cotrans_up_64(pi_code, a, b, n)) {
                        uint64_t sigma_code = swap_positions_64(pi_code, a, b);
                        next.push_back({sigma_code, pi_val});
                    }
                }
            }
        }
    }
}

// Helper: expand cotrans layer UP (multi-threaded)
static void expand_cotrans_layer_up_mt(const std::vector<CompactEntry>& current,
                                        std::vector<CompactEntry>& next, int n,
                                        int num_threads) {
    std::vector<std::vector<CompactEntry>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 64;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / num_threads);
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint64_t pi_code = current[k].code;
                    double pi_val = current[k].val;
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover_up_64(pi_code, a, b, n)) {
                                if (is_valid_cotrans_up_64(pi_code, a, b, n)) {
                                    uint64_t sigma_code = swap_positions_64(pi_code, a, b);
                                    buffers[t].push_back({sigma_code, pi_val});
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry>().swap(b);
    }
}

// BFS sort-reduce cotrans evaluator for n <= 16
// Starts from target, goes UP level by level to w0
double schubert_cotrans_double_bfs(const std::vector<int>& target_w,
                                    std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    int max_ell = n * (n - 1) / 2;

    if (ell == max_ell) return 1.0;  // w0

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntry> layer;
    layer.push_back({pack_perm64(target_w), 1.0});

    int level = ell;

    while (level < max_ell) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return 0.0;

        // Progress reporting (throttled to every 300ms)
        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS cotrans level %d/%d | frontier: %s   ",
                       elapsed, level, max_ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        // Expand up one level
        std::vector<CompactEntry> next;
        if (layer.size() < 1000) {
            expand_cotrans_layer_up_st(layer, next, n);
        } else {
            expand_cotrans_layer_up_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntry>().swap(layer);

        // Sort-reduce
        if (next.size() >= 50000) {
            radix_sort_reduce_layer(next);
        } else {
            sort_reduce_layer(next);
        }

        layer = std::move(next);
        level++;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS cotrans: %d levels in %.2fs                                    \n",
               max_ell - ell, elapsed);
    }

    // At level max_ell (w0), the single entry holds the answer
    double val = 0.0;
    for (const auto& e : layer) val += e.val;
    return val;
}

// =============================================================================
// BFS SORT-REDUCE COTRANS EXACT EVALUATOR (N <= 16)
// Evaluates S_w(1^n) exactly by BFS from target UP to w0
// Uses mpz_class (GMP integers) for exact arithmetic
// =============================================================================

// Helper: expand cotrans layer UP with mpz_class values (single-threaded)
static void expand_cotrans_layer_up_mpz_st(const std::vector<CompactEntryMpz>& current,
                                            std::vector<CompactEntryMpz>& next, int n) {
    next.reserve(current.size() * n);
    for (size_t k = 0; k < current.size(); k++) {
        uint64_t pi_code = current[k].code;
        const mpz_class& pi_val = current[k].val;
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (is_bruhat_cover_up_64(pi_code, a, b, n)) {
                    if (is_valid_cotrans_up_64(pi_code, a, b, n)) {
                        uint64_t sigma_code = swap_positions_64(pi_code, a, b);
                        next.push_back({sigma_code, pi_val});
                    }
                }
            }
        }
    }
}

// Helper: expand cotrans layer UP with mpz_class values (multi-threaded)
static void expand_cotrans_layer_up_mpz_mt(const std::vector<CompactEntryMpz>& current,
                                            std::vector<CompactEntryMpz>& next, int n,
                                            int num_threads) {
    std::vector<std::vector<CompactEntryMpz>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 64;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / num_threads);
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint64_t pi_code = current[k].code;
                    const mpz_class& pi_val = current[k].val;
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover_up_64(pi_code, a, b, n)) {
                                if (is_valid_cotrans_up_64(pi_code, a, b, n)) {
                                    uint64_t sigma_code = swap_positions_64(pi_code, a, b);
                                    buffers[t].push_back({sigma_code, pi_val});
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
        std::vector<CompactEntryMpz>().swap(b);
    }
}

// BFS sort-reduce cotrans exact evaluator for n <= 16
// Starts from target, goes UP level by level to w0
mpz_class schubert_cotrans_exact_bfs(const std::vector<int>& target_w,
                                      std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    int max_ell = n * (n - 1) / 2;

    if (ell == max_ell) return mpz_class(1);  // w0

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntryMpz> layer;
    layer.push_back({pack_perm64(target_w), mpz_class(1)});

    int level = ell;

    while (level < max_ell) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return mpz_class(0);

        // Progress reporting (throttled to every 300ms)
        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS cotrans/exact level %d/%d | frontier: %s   ",
                       elapsed, level, max_ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        // Expand up one level
        std::vector<CompactEntryMpz> next;
        if (layer.size() < 1000) {
            expand_cotrans_layer_up_mpz_st(layer, next, n);
        } else {
            expand_cotrans_layer_up_mpz_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntryMpz>().swap(layer);

        // Sort-reduce (std::sort only for exact types)
        sort_reduce_layer_mpz(next);

        layer = std::move(next);
        level++;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS cotrans/exact: %d levels in %.2fs                                    \n",
               max_ell - ell, elapsed);
    }

    // At level max_ell (w0), the single entry holds the answer
    mpz_class val(0);
    for (const auto& e : layer) val += e.val;
    return val;
}

// =============================================================================
// 128-BIT OPTIMIZED SOLVER (n <= 25)
// Uses 5 bits per element, radix sort, dynamic load balancing
// =============================================================================

// Radix sort for CompactEntry128 by 128-bit code (LSD, 16-bit digits)
static void radix_sort_layer_128(std::vector<CompactEntry128>& layer) {
    if (layer.size() <= 1) return;

    std::vector<CompactEntry128> temp(layer.size());
    constexpr int RADIX_BITS = 16;
    constexpr int RADIX_SIZE = 1 << RADIX_BITS;
    constexpr uint64_t RADIX_MASK = RADIX_SIZE - 1;

    // Process up to 128 bits (8 passes of 16 bits)
    // For n=20: 100 bits needed, so 7 passes suffice
    for (int shift = 0; shift < 128; shift += RADIX_BITS) {
        size_t count[RADIX_SIZE] = {0};

        // Count occurrences
        for (const auto& e : layer) {
            size_t digit = (uint64_t)(e.code >> shift) & RADIX_MASK;
            count[digit]++;
        }

        // Compute prefix sums
        size_t total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            size_t c = count[i];
            count[i] = total;
            total += c;
        }

        // Scatter to temp
        for (const auto& e : layer) {
            size_t digit = (uint64_t)(e.code >> shift) & RADIX_MASK;
            temp[count[digit]++] = e;
        }

        std::swap(layer, temp);
    }
}

static void sort_reduce_layer_128(std::vector<CompactEntry128>& layer) {
    if (layer.empty()) return;

    // HYBRID SORT STRATEGY:
    // std::sort (IntroSort) is faster for small arrays due to low overhead.
    // Radix Sort is faster for massive arrays due to O(K) complexity and better cache locality.
    // Threshold: ~150k elements is the crossover point.
    constexpr size_t RADIX_THRESHOLD = 150000;

    if (layer.size() >= RADIX_THRESHOLD) {
        radix_sort_layer_128(layer);
    } else {
        std::sort(layer.begin(), layer.end());
    }

    // Deduplication (summing values for identical permutations)
    size_t write = 0;
    for (size_t read = 1; read < layer.size(); read++) {
        if (layer[read].code == layer[write].code) {
            layer[write].val += layer[read].val;
        } else {
            write++;
            layer[write] = layer[read];
        }
    }
    layer.resize(write + 1);
}

// Expand descent UP for 128-bit
static void expand_descent_layer_128(const std::vector<CompactEntry128>& current,
                                      std::vector<CompactEntry128>& next, int n, int num_threads) {
    std::vector<std::vector<CompactEntry128>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 256;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / num_threads);
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint128_t p_code = current[k].code;
                    double p_val = current[k].val;
                    for (int i = 0; i < n - 1; i++) {
                        int val_i = (p_code >> (5 * i)) & 0x1F;
                        int val_next = (p_code >> (5 * (i + 1))) & 0x1F;
                        if (val_i < val_next) {
                            uint128_t child_code = swap_adjacent_128(p_code, i);
                            buffers[t].push_back({child_code, p_val * (i + 1)});
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry128>().swap(b);
    }
}

// Expand cotrans DOWN for 128-bit
static void expand_cotrans_layer_128(const std::vector<CompactEntry128>& current,
                                      std::vector<CompactEntry128>& next, int n, int num_threads) {
    std::vector<std::vector<CompactEntry128>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 256;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n * n / (2 * num_threads));
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint128_t sigma_code = current[k].code;
                    double sigma_val = current[k].val;

                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover_down_128(sigma_code, a, b, n)) {
                                if (is_valid_cotrans_down_128(sigma_code, a, b, n)) {
                                    uint128_t child_code = swap_positions_128(sigma_code, a, b);
                                    buffers[t].push_back({child_code, sigma_val});
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry128>().swap(b);
    }
}

// =============================================================================
// BFS SORT-REDUCE EVALUATORS FOR 17 <= N <= 25 (128-bit)
// =============================================================================

// Helper: check Bruhat cover going UP (pi[a] < pi[b], 128-bit)
inline bool is_bruhat_cover_up_128(uint128_t code, int a, int b, int n) {
    int wa = (code >> (5 * a)) & 0x1F;
    int wb = (code >> (5 * b)) & 0x1F;
    if (wa >= wb) return false;  // Need pi(a) < pi(b) for going UP

    if (wb - wa == 1) return true;
    if (b - a == 1) return true;

    uint32_t forbidden = ((1u << wb) - 1) & ~((1u << (wa + 1)) - 1);
    for (int m = a + 1; m < b; m++) {
        int wm = (code >> (5 * m)) & 0x1F;
        if (forbidden & (1u << wm)) return false;
    }
    return true;
}

// Helper: check cotrans condition for going UP (128-bit)
inline bool is_valid_cotrans_up_128(uint128_t pi_code, int a, int b, int n) {
    for (int i = 0; i < n; i++) {
        int pi_i = ((pi_code >> (5 * i)) & 0x1F) + 1;
        if ((i + 1) + pi_i <= n) {
            return (i == a || i == b);
        }
    }
    return false;
}

// Descent DOWN expansion (128-bit, single-threaded)
static void expand_descent_layer_down_128_st(const std::vector<CompactEntry128>& current,
                                              std::vector<CompactEntry128>& next, int n) {
    next.reserve(current.size() * (n / 2));
    for (size_t k = 0; k < current.size(); k++) {
        uint128_t p_code = current[k].code;
        double p_val = current[k].val;
        for (int i = 0; i < n - 1; i++) {
            int val_i = (p_code >> (5 * i)) & 0x1F;
            int val_next = (p_code >> (5 * (i + 1))) & 0x1F;
            if (val_i > val_next) {
                next.push_back({swap_adjacent_128(p_code, i), p_val * (i + 1)});
            }
        }
    }
}

// Descent DOWN expansion (128-bit, multi-threaded)
static void expand_descent_layer_down_128_mt(const std::vector<CompactEntry128>& current,
                                              std::vector<CompactEntry128>& next, int n,
                                              int num_threads) {
    std::vector<std::vector<CompactEntry128>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 256;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / (2 * num_threads));
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint128_t p_code = current[k].code;
                    double p_val = current[k].val;
                    for (int i = 0; i < n - 1; i++) {
                        int val_i = (p_code >> (5 * i)) & 0x1F;
                        int val_next = (p_code >> (5 * (i + 1))) & 0x1F;
                        if (val_i > val_next) {
                            buffers[t].push_back({swap_adjacent_128(p_code, i), p_val * (i + 1)});
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry128>().swap(b);
    }
}

// Cotrans UP expansion (128-bit, single-threaded)
static void expand_cotrans_layer_up_128_st(const std::vector<CompactEntry128>& current,
                                            std::vector<CompactEntry128>& next, int n) {
    next.reserve(current.size() * n);
    for (size_t k = 0; k < current.size(); k++) {
        uint128_t pi_code = current[k].code;
        double pi_val = current[k].val;
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (is_bruhat_cover_up_128(pi_code, a, b, n)) {
                    if (is_valid_cotrans_up_128(pi_code, a, b, n)) {
                        next.push_back({swap_positions_128(pi_code, a, b), pi_val});
                    }
                }
            }
        }
    }
}

// Cotrans UP expansion (128-bit, multi-threaded)
static void expand_cotrans_layer_up_128_mt(const std::vector<CompactEntry128>& current,
                                            std::vector<CompactEntry128>& next, int n,
                                            int num_threads) {
    std::vector<std::vector<CompactEntry128>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 256;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / num_threads);
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint128_t pi_code = current[k].code;
                    double pi_val = current[k].val;
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover_up_128(pi_code, a, b, n)) {
                                if (is_valid_cotrans_up_128(pi_code, a, b, n)) {
                                    buffers[t].push_back({swap_positions_128(pi_code, a, b), pi_val});
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), b.begin(), b.end());
        std::vector<CompactEntry128>().swap(b);
    }
}

// BFS descent evaluator for 17 <= n <= 25 (128-bit)
double schubert_descent_double_bfs_128(const std::vector<int>& target_w,
                                        std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    if (ell == 0) return 1.0;

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntry128> layer;
    layer.push_back({pack_perm128(target_w), 1.0});

    int level = ell;

    while (level > 0) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return 0.0;

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS128 descent level %d/%d | frontier: %s   ",
                       elapsed, level, ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        std::vector<CompactEntry128> next;
        if (layer.size() < 1000) {
            expand_descent_layer_down_128_st(layer, next, n);
        } else {
            expand_descent_layer_down_128_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntry128>().swap(layer);
        sort_reduce_layer_128(next);

        // Normalize incrementally: divide by current level to avoid ell! overflow
        double inv_level = 1.0 / level;
        for (auto& e : next) e.val *= inv_level;

        layer = std::move(next);
        level--;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS128 descent: %d levels in %.2fs                                 \n", ell, elapsed);
    }

    // Values are already normalized (divided by ell! incrementally)
    double val = 0.0;
    for (const auto& e : layer) val += e.val;
    return val;
}

// BFS cotrans evaluator for 17 <= n <= 25 (128-bit)
double schubert_cotrans_double_bfs_128(const std::vector<int>& target_w,
                                        std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    int max_ell = n * (n - 1) / 2;

    if (ell == max_ell) return 1.0;

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntry128> layer;
    layer.push_back({pack_perm128(target_w), 1.0});

    int level = ell;

    while (level < max_ell) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return 0.0;

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS128 cotrans level %d/%d | frontier: %s   ",
                       elapsed, level, max_ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        std::vector<CompactEntry128> next;
        if (layer.size() < 1000) {
            expand_cotrans_layer_up_128_st(layer, next, n);
        } else {
            expand_cotrans_layer_up_128_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntry128>().swap(layer);
        sort_reduce_layer_128(next);
        layer = std::move(next);
        level++;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS128 cotrans: %d levels in %.2fs                                 \n",
               max_ell - ell, elapsed);
    }

    double val = 0.0;
    for (const auto& e : layer) val += e.val;
    return val;
}

// =============================================================================
// 128-BIT EXACT BFS SORT-REDUCE (N <= 25)
// =============================================================================

static void sort_reduce_layer_128_rational(std::vector<CompactEntry128Rational>& layer) {
    if (layer.empty()) return;

    std::sort(layer.begin(), layer.end());

    size_t write = 0;
    for (size_t read = 1; read < layer.size(); read++) {
        if (layer[read].code == layer[write].code) {
            layer[write].val += layer[read].val;
        } else {
            write++;
            layer[write] = std::move(layer[read]);
        }
    }
    layer.resize(write + 1);
}

static void sort_reduce_layer_128_mpz(std::vector<CompactEntry128Mpz>& layer) {
    if (layer.empty()) return;

    std::sort(layer.begin(), layer.end());

    size_t write = 0;
    for (size_t read = 1; read < layer.size(); read++) {
        if (layer[read].code == layer[write].code) {
            layer[write].val += layer[read].val;
        } else {
            write++;
            layer[write] = std::move(layer[read]);
        }
    }
    layer.resize(write + 1);
}

// =============================================================================
// 128-BIT BFS DESCENT EXACT (Rational, N <= 25)
// =============================================================================

static void expand_descent_layer_down_128_rational_st(const std::vector<CompactEntry128Rational>& current,
                                                       std::vector<CompactEntry128Rational>& next, int n) {
    next.reserve(current.size() * (n / 2));
    for (size_t k = 0; k < current.size(); k++) {
        uint128_t p_code = current[k].code;
        const Rational& p_val = current[k].val;
        for (int i = 0; i < n - 1; i++) {
            int val_i = (p_code >> (5 * i)) & 0x1F;
            int val_next = (p_code >> (5 * (i + 1))) & 0x1F;
            if (val_i > val_next) {
                next.push_back({swap_adjacent_128(p_code, i), p_val * Rational(i + 1)});
            }
        }
    }
}

static void expand_descent_layer_down_128_rational_mt(const std::vector<CompactEntry128Rational>& current,
                                                       std::vector<CompactEntry128Rational>& next, int n,
                                                       int num_threads) {
    std::vector<std::vector<CompactEntry128Rational>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 256;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / (2 * num_threads));
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint128_t p_code = current[k].code;
                    const Rational& p_val = current[k].val;
                    for (int i = 0; i < n - 1; i++) {
                        int val_i = (p_code >> (5 * i)) & 0x1F;
                        int val_next = (p_code >> (5 * (i + 1))) & 0x1F;
                        if (val_i > val_next) {
                            buffers[t].push_back({swap_adjacent_128(p_code, i), p_val * Rational(i + 1)});
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
        std::vector<CompactEntry128Rational>().swap(b);
    }
}

Rational schubert_descent_rational_bfs_128(const std::vector<int>& target_w,
                                            std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    if (ell == 0) return Rational(1);

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntry128Rational> layer;
    layer.push_back({pack_perm128(target_w), Rational(1)});

    int level = ell;

    while (level > 0) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return Rational(0);

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS128 descent/exact level %d/%d | frontier: %s   ",
                       elapsed, level, ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        std::vector<CompactEntry128Rational> next;
        if (layer.size() < 1000) {
            expand_descent_layer_down_128_rational_st(layer, next, n);
        } else {
            expand_descent_layer_down_128_rational_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntry128Rational>().swap(layer);
        sort_reduce_layer_128_rational(next);

        Rational inv_level = Rational(1, level);
        for (auto& e : next) e.val = e.val * inv_level;

        layer = std::move(next);
        level--;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS128 descent/exact: %d levels in %.2fs                                    \n", ell, elapsed);
    }

    Rational val(0);
    for (const auto& e : layer) val = val + e.val;
    return val;
}

// =============================================================================
// 128-BIT BFS COTRANS EXACT (mpz_class, N <= 25)
// =============================================================================

static void expand_cotrans_layer_up_128_mpz_st(const std::vector<CompactEntry128Mpz>& current,
                                                std::vector<CompactEntry128Mpz>& next, int n) {
    next.reserve(current.size() * n);
    for (size_t k = 0; k < current.size(); k++) {
        uint128_t pi_code = current[k].code;
        const mpz_class& pi_val = current[k].val;
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (is_bruhat_cover_up_128(pi_code, a, b, n)) {
                    if (is_valid_cotrans_up_128(pi_code, a, b, n)) {
                        next.push_back({swap_positions_128(pi_code, a, b), pi_val});
                    }
                }
            }
        }
    }
}

static void expand_cotrans_layer_up_128_mpz_mt(const std::vector<CompactEntry128Mpz>& current,
                                                std::vector<CompactEntry128Mpz>& next, int n,
                                                int num_threads) {
    std::vector<std::vector<CompactEntry128Mpz>> buffers(num_threads);
    std::atomic<size_t> global_idx{0};
    constexpr size_t BATCH_SIZE = 256;

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t, n]() {
            buffers[t].reserve(current.size() * n / num_threads);
            while (true) {
                size_t batch_start = global_idx.fetch_add(BATCH_SIZE);
                if (batch_start >= current.size()) break;
                size_t batch_end = std::min(batch_start + BATCH_SIZE, current.size());

                for (size_t k = batch_start; k < batch_end; k++) {
                    uint128_t pi_code = current[k].code;
                    const mpz_class& pi_val = current[k].val;
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover_up_128(pi_code, a, b, n)) {
                                if (is_valid_cotrans_up_128(pi_code, a, b, n)) {
                                    buffers[t].push_back({swap_positions_128(pi_code, a, b), pi_val});
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    size_t total_size = 0;
    for (const auto& b : buffers) total_size += b.size();
    next.reserve(total_size);
    for (auto& b : buffers) {
        next.insert(next.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
        std::vector<CompactEntry128Mpz>().swap(b);
    }
}

mpz_class schubert_cotrans_exact_bfs_128(const std::vector<int>& target_w,
                                          std::atomic<bool>& stop, bool quiet, int num_threads) {
    int n = (int)target_w.size();
    int ell = compute_length(target_w);
    int max_ell = n * (n - 1) / 2;

    if (ell == max_ell) return mpz_class(1);

    if (num_threads <= 0) num_threads = DEFAULT_THREADS;
    if (num_threads <= 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();
    auto last_print = start_time;

    std::vector<CompactEntry128Mpz> layer;
    layer.push_back({pack_perm128(target_w), mpz_class(1)});

    int level = ell;

    while (level < max_ell) {
        if (stop.load(std::memory_order_relaxed) || g_stop_flag.load(std::memory_order_relaxed))
            return mpz_class(0);

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 0.3) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  [%.1fs] BFS128 cotrans/exact level %d/%d | frontier: %s   ",
                       elapsed, level, max_ell, format_number(layer.size()).c_str());
                fflush(stdout);
                last_print = now;
            }
        }

        std::vector<CompactEntry128Mpz> next;
        if (layer.size() < 1000) {
            expand_cotrans_layer_up_128_mpz_st(layer, next, n);
        } else {
            expand_cotrans_layer_up_128_mpz_mt(layer, next, n, num_threads);
        }
        std::vector<CompactEntry128Mpz>().swap(layer);
        sort_reduce_layer_128_mpz(next);
        layer = std::move(next);
        level++;
    }

    if (!quiet) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  BFS128 cotrans/exact: %d levels in %.2fs                                    \n",
               max_ell - ell, elapsed);
    }

    mpz_class val(0);
    for (const auto& e : layer) val += e.val;
    return val;
}

void cmd_max_128(int n, int requested_threads = 0) {
    if (n > 25) {
        printf("Error: 128-bit solver only supports N <= 25.\n");
        return;
    }

    int num_threads = requested_threads > 0 ? requested_threads : DEFAULT_THREADS;
    if (num_threads == 0) num_threads = 4;
    int threads_per_side = std::max(1, num_threads / 2);

    printf("Finding max S_w(1^%d) using Meet-in-the-Middle (128-bit packed)\n", n);
    printf("Descent UP from identity, Cotrans DOWN from w0\n");
    printf("Using %d threads (%d per side)\n\n", num_threads, threads_per_side);

    auto start_time = std::chrono::steady_clock::now();
    int max_len = n * (n - 1) / 2;

    std::vector<double> fact(max_len + 2);
    fact[0] = 1.0;
    for (int i = 1; i <= max_len + 1; i++) fact[i] = fact[i - 1] * i;

    // Shared state
    std::atomic<int> descent_level_atomic(0);
    std::atomic<int> cotrans_level_atomic(max_len);
    std::atomic<bool> done(false);
    std::atomic<size_t> total_processed(2);

    std::mutex max_mutex;
    double global_max = 1.0;
    uint128_t global_max_perm = 0;
    int global_max_level = 0;

    std::vector<CompactEntry128> descent_frontier, cotrans_frontier;
    int descent_final_level = 0, cotrans_final_level = max_len;
    std::mutex frontier_mutex;

    printf("Descent starts at level 0, Cotrans starts at level %d\n\n", max_len);

    // Descent thread
    std::thread descent_thread([&]() {
        std::vector<int> id(n);
        for (int i = 0; i < n; i++) id[i] = i + 1;
        std::vector<CompactEntry128> layer;
        layer.push_back({pack_perm128(id), 1.0});
        int level = 0;

        while (!done.load() && !g_stop_flag.load()) {
            int cotrans_lvl = cotrans_level_atomic.load();
            if (level >= cotrans_lvl) break;

            std::vector<CompactEntry128> next;
            expand_descent_layer_128(layer, next, n, threads_per_side);
            std::vector<CompactEntry128>().swap(layer);
            sort_reduce_layer_128(next);
            layer = std::move(next);
            level++;
            descent_level_atomic.store(level);
            total_processed.fetch_add(layer.size());

            double norm = fact[level];
            for (const auto& e : layer) {
                double actual = e.val / norm;
                std::lock_guard<std::mutex> lock(max_mutex);
                if (actual > global_max) {
                    global_max = actual;
                    global_max_perm = e.code;
                    global_max_level = level;
                }
            }
        }

        std::lock_guard<std::mutex> lock(frontier_mutex);
        descent_frontier = std::move(layer);
        descent_final_level = level;
    });

    // Cotrans thread
    std::thread cotrans_thread([&]() {
        std::vector<int> w0(n);
        for (int i = 0; i < n; i++) w0[i] = n - i;
        std::vector<CompactEntry128> layer;
        layer.push_back({pack_perm128(w0), 1.0});
        int level = max_len;

        while (!done.load() && !g_stop_flag.load()) {
            int descent_lvl = descent_level_atomic.load();
            if (level <= descent_lvl) break;

            std::vector<CompactEntry128> next;
            expand_cotrans_layer_128(layer, next, n, threads_per_side);
            std::vector<CompactEntry128>().swap(layer);
            sort_reduce_layer_128(next);
            layer = std::move(next);
            level--;
            cotrans_level_atomic.store(level);
            total_processed.fetch_add(layer.size());

            for (const auto& e : layer) {
                std::lock_guard<std::mutex> lock(max_mutex);
                if (e.val > global_max) {
                    global_max = e.val;
                    global_max_perm = e.code;
                    global_max_level = level;
                }
            }
        }

        std::lock_guard<std::mutex> lock(frontier_mutex);
        cotrans_frontier = std::move(layer);
        cotrans_final_level = level;
    });

    // Progress reporting
    while (!done.load() && !g_stop_flag.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        int d_lvl = descent_level_atomic.load();
        int c_lvl = cotrans_level_atomic.load();

        if (d_lvl >= c_lvl) {
            done.store(true);
            break;
        }

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  Descent@%d | Cotrans@%d | Max: %.2e | %.1fs   ",
               d_lvl, c_lvl, global_max, elapsed);
        fflush(stdout);
    }

    descent_thread.join();
    cotrans_thread.join();

    auto end_time = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::vector<int> w = unpack_perm128(global_max_perm, n);

    // Verify with exact computation
    printf("\n\nVerifying with exact CPU computation...\n");
    std::atomic<bool> stop_flag{false};
    double exact_val = schubert_cotrans_double(w, stop_flag, true);

    printf("\n--- Result ---\n");
    printf("Max S_w(1^%d) = %.0f\n", n, exact_val);
    printf("Achieved by w = ");
    print_permutation(w);
    printf("\n");
    printf("Length ell(w) = %d\n", global_max_level);
    printf("Processed: %s permutations\n", format_number(total_processed.load()).c_str());
    printf("Total time: %.4f seconds\n", total_elapsed);

    if (n >= 1 && n <= 25) {
        const auto& mppy = MPPY_DATA[n - 1];
        LayeredResult lr = layered_product(mppy.layers);
        double layered_val = lr.exact_value.convert_to<double>();

        printf("\n--- Comparison to Layered ---\n");
        printf("Layered w(");
        for (size_t i = 0; i < mppy.layers.size(); i++) {
            printf("%d%s", mppy.layers[i], i + 1 < mppy.layers.size() ? "," : "");
        }
        printf("): %.0f\n", layered_val);
        if (exact_val > layered_val + 0.5) {
            printf("*** BEAT layered! %.0f (%.4fx) ***\n", exact_val, exact_val / layered_val);
        } else if (fabs(exact_val - layered_val) < 0.5) {
            printf("Matched layered exactly.\n");
        } else {
            printf("Found: %.0f (%.4fx layered)\n", exact_val, exact_val / layered_val);
        }
    }
}

void cmd_max_optimized(int n, int requested_threads = 0) {
    if (n > 16) {
        // Dispatch to 128-bit solver for n > 16
        cmd_max_128(n, requested_threads);
        return;
    }

    int num_threads = requested_threads > 0 ? requested_threads : DEFAULT_THREADS;
    if (num_threads == 0) num_threads = 4;
    int threads_per_side = std::max(1, num_threads / 2);

    printf("Finding max S_w(1^%d) using Meet-in-the-Middle (64-bit packed)\n", n);
    printf("Descent UP from identity, Cotrans DOWN from w0\n");
    printf("Using %d threads (%d per side)\n\n", num_threads, threads_per_side);

    auto start_time = std::chrono::steady_clock::now();

    int max_len = n * (n - 1) / 2;

    // Precompute factorials for descent normalization
    std::vector<double> fact(max_len + 2);
    fact[0] = 1.0;
    for (int i = 1; i <= max_len + 1; i++) fact[i] = fact[i - 1] * i;

    // Shared state for synchronization
    std::atomic<int> descent_level_atomic(0);
    std::atomic<int> cotrans_level_atomic(max_len);
    std::atomic<bool> done(false);
    std::atomic<size_t> total_processed(2);  // identity + w0

    // Shared max tracking
    std::mutex max_mutex;
    double global_max = 1.0;
    uint64_t global_max_perm = 0;
    int global_max_level = 0;

    // Final frontiers (set when threads complete)
    std::vector<CompactEntry> descent_frontier, cotrans_frontier;
    int descent_final_level = 0, cotrans_final_level = max_len;
    std::mutex frontier_mutex;

    printf("Descent starts at level 0, Cotrans starts at level %d\n\n", max_len);

    // Descent thread
    std::thread descent_thread([&]() {
        std::vector<int> id(n);
        for (int i = 0; i < n; i++) id[i] = i + 1;
        std::vector<CompactEntry> layer;
        layer.push_back({pack_perm64(id), 1.0});
        int level = 0;

        while (!done.load() && !g_stop_flag.load()) {
            int cotrans_lvl = cotrans_level_atomic.load();
            if (level >= cotrans_lvl) {
                // We've met or passed cotrans
                break;
            }

            // Expand UP one level
            std::vector<CompactEntry> next;
            expand_descent_layer(layer, next, n, threads_per_side);
            std::vector<CompactEntry>().swap(layer);
            sort_reduce_layer(next);
            layer = std::move(next);
            level++;
            descent_level_atomic.store(level);
            total_processed.fetch_add(layer.size());

            // Track max
            double norm = fact[level];
            for (const auto& e : layer) {
                double actual = e.val / norm;
                std::lock_guard<std::mutex> lock(max_mutex);
                if (actual > global_max) {
                    global_max = actual;
                    global_max_perm = e.code;
                    global_max_level = level;
                }
            }
        }

        // Save final frontier
        {
            std::lock_guard<std::mutex> lock(frontier_mutex);
            descent_frontier = std::move(layer);
            descent_final_level = level;
        }
    });

    // Cotrans thread
    std::thread cotrans_thread([&]() {
        std::vector<int> w0(n);
        for (int i = 0; i < n; i++) w0[i] = n - i;
        std::vector<CompactEntry> layer;
        layer.push_back({pack_perm64(w0), 1.0});
        int level = max_len;

        while (!done.load() && !g_stop_flag.load()) {
            int descent_lvl = descent_level_atomic.load();
            if (level <= descent_lvl) {
                // Descent has met or passed us
                break;
            }

            // Expand DOWN one level
            std::vector<CompactEntry> next;
            expand_cotrans_layer(layer, next, n, threads_per_side);
            std::vector<CompactEntry>().swap(layer);
            sort_reduce_layer(next);
            layer = std::move(next);
            level--;
            cotrans_level_atomic.store(level);
            total_processed.fetch_add(layer.size());

            // Track max (cotrans values are actual S_w)
            for (const auto& e : layer) {
                std::lock_guard<std::mutex> lock(max_mutex);
                if (e.val > global_max) {
                    global_max = e.val;
                    global_max_perm = e.code;
                    global_max_level = level;
                }
            }
        }

        // Save final frontier
        {
            std::lock_guard<std::mutex> lock(frontier_mutex);
            cotrans_frontier = std::move(layer);
            cotrans_final_level = level;
        }
    });

    // Progress reporting in main thread
    while (!done.load() && !g_stop_flag.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        int d_lvl = descent_level_atomic.load();
        int c_lvl = cotrans_level_atomic.load();

        if (d_lvl >= c_lvl) {
            done.store(true);
            break;
        }

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        printf("\r  Descent@%d | Cotrans@%d | Max: %.2e | %.1fs   ",
               d_lvl, c_lvl, global_max, elapsed);
        fflush(stdout);
    }

    descent_thread.join();
    cotrans_thread.join();

    // Report meeting
    printf("\n\nDescent reached level %d, Cotrans reached level %d\n",
           descent_final_level, cotrans_final_level);
    printf("Descent frontier: %s permutations\n", format_number(descent_frontier.size()).c_str());
    printf("Cotrans frontier: %s permutations\n", format_number(cotrans_frontier.size()).c_str());

    // Match frontiers if at same level
    size_t matches = 0;
    if (descent_final_level == cotrans_final_level) {
        size_t di = 0, ci = 0;
        while (di < descent_frontier.size() && ci < cotrans_frontier.size()) {
            if (descent_frontier[di].code < cotrans_frontier[ci].code) {
                di++;
            } else if (descent_frontier[di].code > cotrans_frontier[ci].code) {
                ci++;
            } else {
                matches++;
                double val = cotrans_frontier[ci].val;
                if (val > global_max) {
                    global_max = val;
                    global_max_perm = cotrans_frontier[ci].code;
                    global_max_level = descent_final_level;
                }
                di++;
                ci++;
            }
        }
    }
    printf("Matched permutations: %s\n", format_number(matches).c_str());

    auto end_time = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::vector<int> w = unpack_perm64(global_max_perm, n);

    printf("\n--- Result ---\n");
    printf("Max S_w(1^%d) = %.0f\n", n, global_max);
    printf("Achieved by w = ");
    print_permutation(w);
    printf("\n");
    printf("Length ell(w) = %d\n", global_max_level);
    printf("Processed: %s permutations\n", format_number(total_processed.load()).c_str());
    printf("Total time: %.4f seconds\n", total_elapsed);

    // Compare to MPP3 layered
    if (n >= 1 && n <= 25) {
        const auto& mppy = MPPY_DATA[n - 1];
        LayeredResult lr = layered_product(mppy.layers);
        double layered_val = lr.exact_value.convert_to<double>();

        printf("\n--- Comparison to Layered ---\n");
        printf("Layered w(");
        for (size_t i = 0; i < mppy.layers.size(); i++) {
            printf("%d%s", mppy.layers[i], i + 1 < mppy.layers.size() ? "," : "");
        }
        printf("): %.0f\n", layered_val);
        if (global_max > layered_val) {
            printf("*** BEAT layered! %.0f (%.2fx) ***\n", global_max, global_max / layered_val);
        } else if (global_max == layered_val) {
            printf("Matched layered exactly.\n");
        } else {
            printf("Found: %.0f (%.2fx layered)\n", global_max, global_max / layered_val);
        }
    }
}

// =============================================================================
// =============================================================================
// HEURISTIC BEAM SEARCH
// Runs descent-UP and cotrans-DOWN beam searches independently to completion,
// then collects top candidates from both frontiers for re-evaluation
// =============================================================================

// Compute "layeredness" score: measures how layered-like a permutation is
// Returns sum of squared descending run lengths (higher = more layered-like)
inline double layered_score(const PermKey128& key) {
    int n = key.n;
    double score = 0.0;
    int run_length = 1;
    for (int i = 0; i < n - 1; i++) {
        if (key.get(i) - 1 == key.get(i + 1)) {
            // Continuing a descending run
            run_length++;
        } else {
            // End of run
            score += (double)run_length * run_length;
            run_length = 1;
        }
    }
    score += (double)run_length * run_length;  // Last run
    return score;
}

// Check if a PermKey128 represents a layered permutation
// Layered permutations consist of descending blocks: [b1, b1-1, ..., 1], [b1+b2, ..., b1+1], etc.
inline bool is_layered_key(const PermKey128& key) {
    int n = key.n;
    int next_min = 1;
    int i = 0;
    while (i < n) {
        int start = i;
        // Find descending run
        while (i + 1 < n && key.get(i) - 1 == key.get(i + 1)) i++;
        int len = i - start + 1;
        int block_max = key.get(start);
        int block_min = key.get(i);
        if (block_min != next_min || block_max != next_min + len - 1) {
            return false;
        }
        next_min += len;
        i++;
    }
    return next_min == n + 1;
}

// Count number of valid ascents (positions where w[i] < w[i+1], so swap increases length)
// More ascents = more "upward potential" in Bruhat graph
inline int count_ascents(const PermKey128& key) {
    int n = key.n;
    int ascents = 0;
    for (int i = 0; i < n - 1; i++) {
        if (key.get(i) < key.get(i + 1)) {
            ascents++;
        }
    }
    return ascents;
}

// Compute "partial layered" score: high for perms that look like [layered prefix][identity suffix]
// Returns a score indicating how well the permutation matches the partial layered pattern
inline double partial_layered_score(const PermKey128& key) {
    int n = key.n;

    // Find where the "active" part ends (first position where w[i] = i+1 and stays identity)
    int active_end = n;
    for (int i = n - 1; i >= 0; i--) {
        if (key.get(i) == i + 1) {
            active_end = i;
        } else {
            break;
        }
    }

    if (active_end == 0) return 0.0;  // Identity permutation

    // Check if the active prefix forms valid layered blocks
    int next_min = 1;
    int i = 0;
    double score = 0.0;
    int num_blocks = 0;

    while (i < active_end) {
        int start = i;
        // Find descending run
        while (i + 1 < active_end && key.get(i) - 1 == key.get(i + 1)) i++;
        int len = i - start + 1;
        int block_max = key.get(start);
        int block_min = key.get(i);

        // Check if this forms a valid layered block
        if (block_min == next_min && block_max == next_min + len - 1) {
            // Valid block! Add to score
            score += (double)len * len;
            num_blocks++;
            next_min += len;
        } else {
            // Invalid - not a proper layered structure
            return layered_score(key);  // Fall back to simple score
        }
        i++;
    }

    // Bonus for having identity suffix (room for growth)
    int suffix_len = n - active_end;
    if (suffix_len > 0 && num_blocks >= 1) {
        // Good partial layered structure with room to grow
        score *= (1.0 + 0.5 * suffix_len / n);
    }

    return score;
}

// Run descent beam search with DUAL OBJECTIVE selection:
// - Half of beam selected by pure estimate
// - Half of beam selected by length-weighted estimate (favors longer perms)
BeamResult run_descent_beam_v2(int n, size_t beam_width, bool quiet, int target_level, bool exclude_layered = false) {
    BeamResult result;
    result.best_value = 1.0;
    result.best_length = 0;
    result.total_processed = 1;

    int max_length = n * (n - 1) / 2;

    // Precompute factorials for normalization
    std::vector<double> fact(max_length + 2);
    fact[0] = 1.0;
    for (int i = 1; i <= max_length + 1; i++) fact[i] = fact[i-1] * i;

    // Start from identity
    std::vector<int> identity(n);
    for (int i = 0; i < n; i++) identity[i] = i + 1;
    PermKey128 id_key(identity);
    result.best_perm = id_key;

    std::vector<FrontierEntry> current;
    current.push_back({id_key, 1.0});

    int num_threads = DEFAULT_THREADS;
    if (num_threads == 0) num_threads = 4;

    int level = 0;
    for (; level < max_length && level <= target_level; level++) {
        if (g_stop_flag.load()) break;

        // Track best at current level
        for (const auto& entry : current) {
            double actual_val = entry.val / fact[level];
            if (actual_val > result.best_value) {
                result.best_value = actual_val;
                result.best_perm = entry.key;
                result.best_length = level;
            }
        }

        // Parallel expansion UP
        std::vector<std::vector<FrontierEntry>> buffers(num_threads);
        std::vector<std::thread> threads;
        size_t chunk = (current.size() + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, current.size());
            if (start >= end) continue;

            threads.emplace_back([&, t, start, end]() {
                buffers[t].reserve((end - start) * (n - 1));
                for (size_t k = start; k < end; k++) {
                    const auto& parent = current[k];
                    for (int i = 0; i < n - 1; i++) {
                        if (parent.key.get(i) < parent.key.get(i + 1)) {
                            PermKey128 child = parent.key.swap_adjacent(i);
                            buffers[t].push_back({child, parent.val * (i + 1)});
                        }
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        // Merge and dedup
        size_t total = 0;
        for (const auto& b : buffers) total += b.size();
        if (total == 0) break;

        std::vector<FrontierEntry> next;
        next.reserve(total);
        for (const auto& b : buffers) next.insert(next.end(), b.begin(), b.end());

        std::sort(next.begin(), next.end());

        std::vector<FrontierEntry> deduped;
        deduped.reserve(next.size());
        if (!next.empty()) {
            deduped.push_back(next[0]);
            for (size_t k = 1; k < next.size(); k++) {
                if (next[k].key.code == deduped.back().key.code) {
                    deduped.back().val += next[k].val;
                } else {
                    deduped.push_back(next[k]);
                }
            }
        }

        result.total_processed += deduped.size();

        // Normalize values
        double next_fact = fact[level + 1];
        for (auto& e : deduped) {
            e.val = e.val / next_fact;
        }

        // If exclude_layered mode, filter out layered permutations (only near target level)
        // Early levels need layered structure to reach interesting non-layered later
        if (exclude_layered && level >= target_level / 2) {
            size_t before = deduped.size();
            // Count non-layered
            size_t non_layered_count = 0;
            for (const auto& e : deduped) {
                if (!is_layered_key(e.key)) non_layered_count++;
            }
            // Only filter if we have enough non-layered to continue (at least beam_width/4)
            if (non_layered_count >= beam_width / 4) {
                deduped.erase(std::remove_if(deduped.begin(), deduped.end(),
                    [](const FrontierEntry& e) { return is_layered_key(e.key); }),
                    deduped.end());
                size_t removed = before - deduped.size();
                if (!quiet && removed > 0) {
                    printf("    [excluded %zu layered from %zu, kept %zu non-layered]\n",
                           removed, before, deduped.size());
                }
            }
        }

        // DIVERSITY-FOCUSED SELECTION: 50% by estimate, 50% by branching potential
        // Goal: explore diverse structures, not just layered-like ones
        if (deduped.size() > beam_width) {
            size_t by_estimate = beam_width / 2;
            size_t by_branching = beam_width - by_estimate;

            // 1. Select top 50% by pure estimate (exploitation)
            std::partial_sort(deduped.begin(), deduped.begin() + by_estimate, deduped.end(),
                [](const FrontierEntry& a, const FrontierEntry& b) {
                    return a.val > b.val;
                });

            // Mark selected entries
            std::unordered_set<uint128_t, Uint128Hash> selected;
            for (size_t i = 0; i < by_estimate; i++) {
                selected.insert(deduped[i].key.code);
            }

            // 2. For remaining entries, score by branching potential (exploration)
            // High ascent count = more paths to explore = more chance to find anomalies
            std::vector<std::pair<double, size_t>> scored;
            for (size_t i = by_estimate; i < deduped.size(); i++) {
                int ascents = count_ascents(deduped[i].key);
                // Score = estimate × (1 + ascents/(n-1)) to favor high-branching candidates
                double score = deduped[i].val * (1.0 + (double)ascents / (n - 1));
                scored.push_back({score, i});
            }

            // Select top 50% by branching-weighted score
            std::partial_sort(scored.begin(), scored.begin() + std::min(by_branching, scored.size()), scored.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            // Build final beam
            std::vector<FrontierEntry> new_beam;
            new_beam.reserve(beam_width);
            for (size_t i = 0; i < by_estimate; i++) {
                new_beam.push_back(deduped[i]);
            }
            for (size_t i = 0; i < std::min(by_branching, scored.size()); i++) {
                new_beam.push_back(deduped[scored[i].second]);
            }
            deduped = std::move(new_beam);
        }

        // Save top candidates from this level
        size_t save_count = std::min((size_t)100, deduped.size());
        for (size_t i = 0; i < save_count; i++) {
            result.level_candidates.push_back({deduped[i].key.code, deduped[i].val});
        }

        // Denormalize back
        for (auto& e : deduped) {
            e.val = e.val * next_fact;
        }

        current = std::move(deduped);

        if (!quiet) {
            double level_max = 0.0;
            for (const auto& e : current) {
                double v = e.val / next_fact;
                if (v > level_max) level_max = v;
            }
            printf("  [descent v2 div] Level %2d: frontier=%zu, max=%.2e\n",
                   level + 1, current.size(), level_max);
        }
    }

    // Final best
    double final_fact = fact[level];
    for (const auto& entry : current) {
        double actual_val = entry.val / final_fact;
        if (actual_val > result.best_value) {
            result.best_value = actual_val;
            result.best_perm = entry.key;
            result.best_length = level;
        }
    }

    return result;
}

// Run descent beam search UP from identity to target_level (ORIGINAL VERSION)
BeamResult run_descent_beam(int n, size_t beam_width, bool quiet, int target_level) {
    BeamResult result;
    result.best_value = 1.0;
    result.best_length = 0;
    result.total_processed = 1;

    int max_length = n * (n - 1) / 2;

    // Precompute factorials for normalization: S_w = accumulated_val / level!
    std::vector<double> fact(max_length + 2);
    fact[0] = 1.0;
    for (int i = 1; i <= max_length + 1; i++) fact[i] = fact[i-1] * i;

    // Start from identity
    std::vector<int> identity(n);
    for (int i = 0; i < n; i++) identity[i] = i + 1;
    PermKey128 id_key(identity);
    result.best_perm = id_key;

    std::vector<FrontierEntry> current;
    current.push_back({id_key, 1.0});  // Accumulated value starts at 1

    int num_threads = DEFAULT_THREADS;
    if (num_threads == 0) num_threads = 4;

    int level = 0;
    for (; level < max_length && level <= target_level; level++) {
        if (g_stop_flag.load()) break;

        // Track best at current level (need to normalize)
        for (const auto& entry : current) {
            double actual_val = entry.val / fact[level];
            if (actual_val > result.best_value) {
                result.best_value = actual_val;
                result.best_perm = entry.key;
                result.best_length = level;
            }
        }

        // Parallel expansion UP: adjacent swaps where w[i] < w[i+1]
        std::vector<std::vector<FrontierEntry>> buffers(num_threads);
        std::vector<std::thread> threads;
        size_t chunk = (current.size() + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, current.size());
            if (start >= end) continue;

            threads.emplace_back([&, t, start, end]() {
                buffers[t].reserve((end - start) * (n - 1));
                for (size_t k = start; k < end; k++) {
                    const auto& parent = current[k];
                    for (int i = 0; i < n - 1; i++) {
                        // Ascent: swap where w[i] < w[i+1] (increases length)
                        if (parent.key.get(i) < parent.key.get(i + 1)) {
                            PermKey128 child = parent.key.swap_adjacent(i);
                            // Descent formula accumulates: child_accum = parent_accum * (i+1)
                            buffers[t].push_back({child, parent.val * (i + 1)});
                        }
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        // Merge, sort, dedup
        size_t total = 0;
        for (const auto& b : buffers) total += b.size();
        if (total == 0) break;

        std::vector<FrontierEntry> next;
        next.reserve(total);
        for (const auto& b : buffers) next.insert(next.end(), b.begin(), b.end());

        std::sort(next.begin(), next.end());

        // Dedup: merge entries with same key (sum accumulated values)
        std::vector<FrontierEntry> deduped;
        deduped.reserve(next.size());
        if (!next.empty()) {
            deduped.push_back(next[0]);
            for (size_t k = 1; k < next.size(); k++) {
                if (next[k].key.code == deduped.back().key.code) {
                    deduped.back().val += next[k].val;
                } else {
                    deduped.push_back(next[k]);
                }
            }
        }

        result.total_processed += deduped.size();

        // Beam pruning: normalize values first
        double next_fact = fact[level + 1];
        for (auto& e : deduped) {
            e.val = e.val / next_fact;  // Normalize to actual S_w estimate
        }

        if (deduped.size() > beam_width) {
            // Keep top K by value
            std::partial_sort(deduped.begin(), deduped.begin() + beam_width, deduped.end(),
                [](const FrontierEntry& a, const FrontierEntry& b) {
                    return a.val > b.val;  // Descending
                });
            deduped.resize(beam_width);
        }

        // Save top candidates from this level (values are normalized S_w estimates)
        // Save more to have better coverage
        size_t save_count = std::min((size_t)100, deduped.size());
        for (size_t i = 0; i < save_count; i++) {
            result.level_candidates.push_back({deduped[i].key.code, deduped[i].val});
        }

        // Denormalize back for next iteration (multiply by next_fact)
        for (auto& e : deduped) {
            e.val = e.val * next_fact;
        }

        current = std::move(deduped);

        if (!quiet) {
            double level_max = 0.0;
            for (const auto& e : current) {
                double v = e.val / next_fact;
                if (v > level_max) level_max = v;
            }
            printf("  [descent UP] Level %2d: frontier=%zu, max=%.2e\n",
                   level + 1, current.size(), level_max);
        }
    }

    // Build frontier map with normalized S_w values (no longer needed for candidates)
    double final_fact = fact[level];
    for (const auto& entry : current) {
        double actual_val = entry.val / final_fact;
        if (actual_val > result.best_value) {
            result.best_value = actual_val;
            result.best_perm = entry.key;
            result.best_length = level;
        }
    }

    return result;
}

// Check if a permutation has ANY valid cotrans children (going DOWN)
// Used for "Liveness-Aware Pruning" in beam search
inline bool has_valid_cotrans_children(const PermKey128& k) {
    int n = k.n;
    for (int a = 0; a < n; a++) {
        for (int b = a + 1; b < n; b++) {
            // Check if swapping (a,b) is a valid step down
            if (is_bruhat_cover_down(k, a, b)) {
                if (is_valid_cotrans_down(k, a, b, n)) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Run cotrans beam search DOWN with LIVENESS-AWARE PRUNING
// Fix: Identifies "Dead Ends" (high value, no children), archives them,
// but removes them from the beam to allow "Live Nodes" to propagate.
BeamResult run_cotrans_beam(int n, size_t beam_width, bool quiet, int target_level) {
    BeamResult result;
    result.best_value = 1.0;
    result.total_processed = 1;

    int max_length = n * (n - 1) / 2;
    result.best_length = max_length;

    // Start from w0
    std::vector<int> w0(n);
    for (int i = 0; i < n; i++) w0[i] = n - i;
    PermKey128 w0_key(w0);
    result.best_perm = w0_key;

    // Use map for O(1) parent lookup: code -> S value
    std::unordered_map<uint128_t, double, Uint128Hash> parent_map;
    parent_map[w0_key.code] = 1.0;

    int num_threads = DEFAULT_THREADS;
    if (num_threads == 0) num_threads = 4;

    int level = max_length;

    while (level > 0 && level >= target_level) {
        if (g_stop_flag.load()) break;

        // Track best from previous level
        for (const auto& [code, val] : parent_map) {
            if (val > result.best_value) {
                result.best_value = val;
                result.best_perm.code = code;
                result.best_perm.n = n;
                result.best_length = level;
            }
        }

        level--;
        std::vector<std::pair<uint128_t, double>> parents(parent_map.begin(), parent_map.end());

        // Step 1: Generate all unique children (PULL mode)
        std::vector<std::vector<uint128_t>> child_buffers(num_threads);
        std::vector<std::thread> threads;
        size_t chunk = (parents.size() + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, parents.size());
            if (start >= end) continue;

            threads.emplace_back([&, t, start, end, n]() {
                child_buffers[t].reserve((end - start) * n * n / 2);
                for (size_t k = start; k < end; k++) {
                    PermKey128 parent_key;
                    parent_key.code = parents[k].first;
                    parent_key.n = n;
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover_down(parent_key, a, b)) {
                                PermKey128 child = parent_key.swap(a, b);
                                child_buffers[t].push_back(child.code);
                            }
                        }
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        std::vector<uint128_t> all_children;
        size_t total = 0;
        for (const auto& b : child_buffers) total += b.size();
        all_children.reserve(total);
        for (const auto& b : child_buffers) all_children.insert(all_children.end(), b.begin(), b.end());

        if (all_children.empty()) break;

        std::sort(all_children.begin(), all_children.end());
        all_children.erase(std::unique(all_children.begin(), all_children.end()), all_children.end());

        // Step 2: Compute S values (Pull from parents)
        std::vector<std::pair<uint128_t, double>> child_values(all_children.size());
        threads.clear();
        chunk = (all_children.size() + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, all_children.size());
            if (start >= end) continue;

            threads.emplace_back([&, t, start, end, n]() {
                for (size_t k = start; k < end; k++) {
                    uint128_t child_code = all_children[k];
                    PermKey128 child_key;
                    child_key.code = child_code;
                    child_key.n = n;

                    double sum = 0.0;
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (is_bruhat_cover(child_key, a, b)) {
                                PermKey128 parent_key = child_key.swap(a, b);
                                auto it = parent_map.find(parent_key.code);
                                if (it != parent_map.end()) {
                                    if (is_valid_cotrans_down(parent_key, a, b, n)) {
                                        sum += it->second;
                                    }
                                }
                            }
                        }
                    }
                    child_values[k] = {child_code, sum};
                }
            });
        }
        for (auto& t : threads) t.join();

        // Step 3: LIVENESS-AWARE PRUNING
        // We separate candidates into those that can propagate (Live) and those that cannot (Dead)
        std::vector<FrontierEntry> candidates;
        candidates.reserve(child_values.size());

        for (const auto& [code, val] : child_values) {
            if (val > 0) {
                PermKey128 key; key.code = code; key.n = n;
                candidates.push_back({key, val});
            }
        }
        result.total_processed += candidates.size();

        // Sort by value descending
        std::sort(candidates.begin(), candidates.end(),
            [](const FrontierEntry& a, const FrontierEntry& b) {
                return a.val > b.val;
            });

        // ARCHIVAL: Save top raw candidates (Live AND Dead) for final result reporting
        // Dead ends often have local max values, so we must record them before discarding
        size_t save_count = std::min((size_t)10, candidates.size());
        for (size_t i = 0; i < save_count; i++) {
            result.level_candidates.push_back({candidates[i].key.code, candidates[i].val});
        }

        // PROPAGATION: Fill beam strictly with LIVE nodes (parallelized liveness check)
        // Check top candidates for liveness in parallel
        size_t check_count = std::min(candidates.size(), beam_width * 3);  // Check 3x beam to find enough live
        std::vector<bool> is_live(check_count, false);

        threads.clear();
        chunk = (check_count + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; t++) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, check_count);
            if (start >= end) continue;
            threads.emplace_back([&, start, end]() {
                for (size_t i = start; i < end; i++) {
                    is_live[i] = has_valid_cotrans_children(candidates[i].key);
                }
            });
        }
        for (auto& t : threads) t.join();

        std::vector<FrontierEntry> live_beam;
        live_beam.reserve(beam_width);
        int dead_pruned_count = 0;

        for (size_t i = 0; i < check_count && live_beam.size() < beam_width; i++) {
            if (is_live[i]) {
                live_beam.push_back(candidates[i]);
            } else {
                dead_pruned_count++;
            }
        }

        // Setup next map
        parent_map.clear();
        if (live_beam.empty()) {
             if (!quiet) printf("  [cotrans DN] Level %2d: BEAM DIED (all dead ends)\n", level);
             break;
        }

        for (const auto& entry : live_beam) {
            parent_map[entry.key.code] = entry.val;
        }

        if (!quiet) {
            printf("  [cotrans DN] Level %2d: frontier=%zu, max=%.2e (Pruned %d dead ends)\n",
                   level, live_beam.size(), live_beam[0].val, dead_pruned_count);
        }
    }

    // Final best check
    for (const auto& [code, val] : parent_map) {
        if (val > result.best_value) {
            result.best_value = val;
            result.best_perm.code = code;
            result.best_perm.n = n;
            result.best_length = level;
        }
    }

    return result;
}

// =============================================================================
// Large N beam search (n > 25, up to 100) - requires --no-check
// Simplified version using PermKeyLarge
// =============================================================================

BeamResultLarge run_descent_beam_large(int n, size_t beam_width, bool quiet, int target_level) {
    BeamResultLarge result;
    result.best_value = 1.0;
    result.best_length = 0;
    result.total_processed = 1;

    int max_length = n * (n - 1) / 2;

    // Precompute factorials (use log for large n to avoid overflow)
    std::vector<double> log_fact(max_length + 2);
    log_fact[0] = 0.0;
    for (int i = 1; i <= max_length + 1; i++) log_fact[i] = log_fact[i-1] + log((double)i);

    // Start from identity
    std::vector<int> identity(n);
    for (int i = 0; i < n; i++) identity[i] = i + 1;
    PermKeyLarge id_key(identity);
    result.best_perm = id_key;

    std::vector<FrontierEntryLarge> current;
    current.push_back({id_key, 0.0});  // Store log of accumulated value

    int num_threads = DEFAULT_THREADS;
    if (num_threads == 0) num_threads = 4;

    int level = 0;
    for (; level < max_length && level <= target_level; level++) {
        if (g_stop_flag.load()) break;

        // Track best at current level (values are log-normalized estimates)
        for (const auto& entry : current) {
            double actual_log_val = entry.val - log_fact[level];
            if (actual_log_val > log(result.best_value)) {
                result.best_value = exp(actual_log_val);
                result.best_perm = entry.key;
                result.best_length = level;
            }
        }

        // Parallel expansion UP: adjacent swaps where w[i] < w[i+1]
        std::vector<std::vector<FrontierEntryLarge>> buffers(num_threads);
        std::vector<std::thread> threads;
        size_t chunk = (current.size() + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, current.size());
            if (start >= end) continue;

            threads.emplace_back([&, t, start, end]() {
                buffers[t].reserve((end - start) * (n - 1));
                for (size_t k = start; k < end; k++) {
                    const auto& parent = current[k];
                    for (int i = 0; i < n - 1; i++) {
                        if (parent.key.get(i) < parent.key.get(i + 1)) {
                            PermKeyLarge child = parent.key.swap_adjacent(i);
                            // Accumulate log(i+1) instead of multiplying
                            buffers[t].push_back({child, parent.val + log((double)(i + 1))});
                        }
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        // Merge
        size_t total = 0;
        for (const auto& b : buffers) total += b.size();
        if (total == 0) break;

        std::vector<FrontierEntryLarge> next;
        next.reserve(total);
        for (const auto& b : buffers) next.insert(next.end(), b.begin(), b.end());

        // Dedup using hash map (sum in log space = log(sum of exp))
        std::unordered_map<PermKeyLarge, double, PermKeyLargeHash> dedup_map;
        dedup_map.reserve(next.size());
        for (const auto& e : next) {
            auto it = dedup_map.find(e.key);
            if (it == dedup_map.end()) {
                dedup_map[e.key] = e.val;
            } else {
                // Log-sum-exp: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
                double a = it->second, b = e.val;
                double max_ab = std::max(a, b);
                it->second = max_ab + log(1.0 + exp(-fabs(a - b)));
            }
        }

        std::vector<FrontierEntryLarge> deduped;
        deduped.reserve(dedup_map.size());
        for (const auto& [key, val] : dedup_map) {
            deduped.push_back({key, val});
        }

        result.total_processed += deduped.size();

        // Normalize values (subtract log of factorial)
        double log_next_fact = log_fact[level + 1];
        for (auto& e : deduped) {
            e.val = e.val - log_next_fact;
        }

        // Beam pruning: keep top K by (log) value
        if (deduped.size() > beam_width) {
            std::partial_sort(deduped.begin(), deduped.begin() + beam_width, deduped.end(),
                [](const FrontierEntryLarge& a, const FrontierEntryLarge& b) {
                    return a.val > b.val;  // Descending
                });
            deduped.resize(beam_width);
        }

        // Save top candidates from this level
        size_t save_count = std::min((size_t)100, deduped.size());
        for (size_t i = 0; i < save_count; i++) {
            result.level_candidates.push_back({deduped[i].key, exp(deduped[i].val)});
        }

        // Denormalize back for next iteration
        for (auto& e : deduped) {
            e.val = e.val + log_next_fact;
        }

        current = std::move(deduped);

        if (!quiet) {
            double level_max = 0.0;
            for (const auto& e : current) {
                double v = exp(e.val - log_next_fact);
                if (v > level_max) level_max = v;
            }
            printf("  [descent large] Level %2d: frontier=%zu, max=%.2e\n",
                   level + 1, current.size(), level_max);
        }
    }

    // Update best from final frontier
    double log_final_fact = log_fact[level];
    for (const auto& entry : current) {
        double actual_val = exp(entry.val - log_final_fact);
        if (actual_val > result.best_value) {
            result.best_value = actual_val;
            result.best_perm = entry.key;
            result.best_length = level;
        }
    }

    return result;
}

// Heuristic beam search for max S_w
// Descent-only beam search over Cayley graph (adjacent transpositions)
// Explores from identity UP to 0.45 * binom(n,2), then re-evaluates top candidates
void cmd_heuristic(int n, size_t beam_width, int num_threads = 0, double target_ratio = 0.50, bool use_v2 = true, bool exclude_layered = false, bool no_check = false) {
    if (num_threads <= 0) num_threads = DEFAULT_THREADS;

    // For n > MAX_N, require --no-check since we can't compute actual values
    if (n > BEAM_MAX_N) {
        printf("Error: n > %d not supported.\n", BEAM_MAX_N);
        return;
    }
    if (n > MAX_N && !no_check) {
        printf("Error: n > %d requires --no-check flag (can't compute actual values).\n", MAX_N);
        return;
    }

    int max_length = n * (n - 1) / 2;
    int target_level = (int)(target_ratio * max_length);

    printf("Heuristic beam search for max S_w(1^%d) [%s%s]\n", n,
           n > MAX_N ? "large-n mode" : (use_v2 ? "v2: dual-objective" : "v1: pure estimate"),
           exclude_layered ? " | NO LAYERED" : "");
    printf("  beam_width=%zu, max_length=%d, target=%.2f -> level %d\n\n",
           beam_width, max_length, target_ratio, target_level);

    auto start_time = std::chrono::steady_clock::now();

    // For large n, use the large beam search and simplified output
    if (n > MAX_N) {
        BeamResultLarge large_result = run_descent_beam_large(n, beam_width, false, target_level);

        auto mid_time = std::chrono::steady_clock::now();
        double search_elapsed = std::chrono::duration<double>(mid_time - start_time).count();

        printf("\n--- Search Results (large n) ---\n");
        printf("Beam search: best_est=%.2e at ell=%d, saved=%zu candidates, processed=%s\n",
               large_result.best_value, large_result.best_length,
               large_result.level_candidates.size(),
               format_number(large_result.total_processed).c_str());
        printf("Search time: %.2fs\n", search_elapsed);

        // Sort candidates by estimated value
        std::sort(large_result.level_candidates.begin(), large_result.level_candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // Show top 5 candidates
        printf("\nTop 5 candidates by estimated value:\n");
        std::vector<int> layers;
        for (size_t i = 0; i < std::min((size_t)5, large_result.level_candidates.size()); i++) {
            const auto& [key, est_val] = large_result.level_candidates[i];
            std::vector<int> w = key.decode();
            int ell = key.length();
            bool is_layered = detect_layered(w, layers);
            printf("  %zu. ell=%d, est=%.4e%s: ", i + 1, ell, est_val,
                   is_layered ? " [layered]" : "");
            print_permutation(w);
            printf("\n");
        }

        printf("\n[--no-check: Skipping re-evaluation (large n mode)]\n");
        auto end_time = std::chrono::steady_clock::now();
        double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
        printf("Total time: %.4f seconds\n", total_elapsed);
        return;
    }

    // Run descent beam search UP from identity (normal n <= MAX_N case)
    BeamResult descent_result = use_v2
        ? run_descent_beam_v2(n, beam_width, false, target_level, exclude_layered)
        : run_descent_beam(n, beam_width, false, target_level);

    auto mid_time = std::chrono::steady_clock::now();
    double search_elapsed = std::chrono::duration<double>(mid_time - start_time).count();

    printf("\n--- Search Results ---\n");
    printf("Beam search: best_est=%.2e at ell=%d, saved=%zu candidates, processed=%s\n",
           descent_result.best_value, descent_result.best_length,
           descent_result.level_candidates.size(),
           format_number(descent_result.total_processed).c_str());
    printf("Search time: %.2fs\n", search_elapsed);

    // Collect candidates from descent beam search
    std::vector<std::tuple<PermKey128, double, int>> candidates;
    candidates.reserve(descent_result.level_candidates.size());
    for (const auto& [code, val] : descent_result.level_candidates) {
        PermKey128 key;
        key.code = code;
        key.n = n;
        candidates.push_back({key, val, 0});  // source=0 (descent)
    }

    printf("\nTotal candidates to evaluate: %zu\n", candidates.size());

    // Check for Ctrl-C interruption after beam search
    if (g_stop_flag.load()) {
        printf("\n[Interrupted after beam search]\n");
        return;
    }

    // Sort candidates by estimated value (descending)
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });

    // Get MPP layered permutation for Cayley distance comparison
    std::vector<int> mppy_layered_perm;
    if (n >= 1 && n <= 25) {
        const auto& mppy = MPPY_DATA[n - 1];
        mppy_layered_perm = build_layered_permutation(mppy.layers);
    }

    // Show top 5 candidates by estimated value before re-evaluation
    printf("\nTop 5 candidates by estimated value:\n");
    std::vector<int> layers;
    for (size_t i = 0; i < std::min((size_t)5, candidates.size()); i++) {
        const auto& [perm, est_val, unused] = candidates[i];
        std::vector<int> w = perm.decode();
        int ell = perm.length();
        bool is_layered = detect_layered(w, layers);
        int cayley_dist = mppy_layered_perm.empty() ? -1 : cayley_distance(w, mppy_layered_perm);
        printf("  %zu. ell=%d, est=%.4e, d_C=%d%s: ", i + 1, ell, est_val, cayley_dist,
               is_layered ? " [layered]" : "");
        print_permutation(w);
        printf("\n");
    }

    // Check for Ctrl-C interruption before re-evaluation
    if (g_stop_flag.load()) {
        printf("\n[Interrupted before re-evaluation]\n");
        return;
    }

    // Skip re-evaluation if --no-check flag is set
    if (no_check) {
        printf("\n[--no-check: Skipping re-evaluation]\n");
        auto end_time = std::chrono::steady_clock::now();
        double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
        printf("Search time: %.4f seconds\n", search_elapsed);
        printf("Total time: %.4f seconds\n", total_elapsed);
        return;
    }

    // Re-evaluate top candidates with full computation
    // Race descent vs cotrans for each non-layered permutation
    size_t eval_limit = std::min(candidates.size(), (size_t)200);

    printf("\nRe-evaluating top %zu candidates (racing descent vs cotrans)...\n", eval_limit);

    std::mutex print_mtx;
    auto eval_start = std::chrono::steady_clock::now();
    std::atomic<size_t> completed{0};

    // Results storage
    std::vector<std::tuple<size_t, int, double, double, bool, std::string>> all_evaluated;
    std::mutex results_mtx;
    double global_best = 0.0;
    PermKey128 global_best_perm;
    int global_best_length = 0;

    // Per-slot progress (for racing threads)
    int num_slots = std::max(1, num_threads / 2);  // Each slot runs 2 threads (descent + cotrans)
    struct SlotProgress {
        std::atomic<size_t> candidate_idx{SIZE_MAX};  // SIZE_MAX = idle
        std::atomic<int> ell{0};
        std::atomic<size_t> descent_iter{0};
        std::atomic<size_t> cotrans_iter{0};
        std::atomic<size_t> descent_memo{0};
        std::atomic<size_t> cotrans_memo{0};
        std::atomic<int> descent_ell{0};    // for racing progress
        std::atomic<int> cotrans_ell{0};    // for racing progress
        std::atomic<bool> descent_done{false};
        std::atomic<bool> cotrans_done{false};
        std::vector<int> current_perm;
        std::chrono::steady_clock::time_point start_time;
    };
    std::vector<SlotProgress> slot_progress(num_slots);

    // Progress monitor thread (prints every 10 seconds, only unfinished)
    std::atomic<bool> monitor_stop{false};
    std::thread monitor_thread([&]() {
        auto last_print = std::chrono::steady_clock::now();
        while (!monitor_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            auto now = std::chrono::steady_clock::now();
            double since_print = std::chrono::duration<double>(now - last_print).count();
            if (since_print >= 10.0) {
                last_print = now;
                double elapsed = std::chrono::duration<double>(now - eval_start).count();

                // Count active slots
                int active = 0;
                for (int s = 0; s < num_slots; s++) {
                    if (slot_progress[s].candidate_idx.load() != SIZE_MAX) active++;
                }

                if (active > 0) {
                    std::lock_guard<std::mutex> lock(print_mtx);
                    printf("\n  [%.0fs] Done: %zu/%zu | Active: %d slots\n",
                           elapsed, completed.load(), eval_limit, active);
                    for (int s = 0; s < num_slots; s++) {
                        size_t idx = slot_progress[s].candidate_idx.load();
                        if (idx == SIZE_MAX) continue;  // Idle slot

                        double perm_elapsed = std::chrono::duration<double>(now - slot_progress[s].start_time).count();
                        bool d_done = slot_progress[s].descent_done.load();
                        bool c_done = slot_progress[s].cotrans_done.load();

                        printf("    S%d: #%3zu ell=%2d | D:%s%s | C:%s%s | %.0fs\n",
                               s, idx, slot_progress[s].ell.load(),
                               format_number(slot_progress[s].descent_iter.load()).c_str(),
                               d_done ? " DONE" : "",
                               format_number(slot_progress[s].cotrans_iter.load()).c_str(),
                               c_done ? " DONE" : "",
                               perm_elapsed);
                    }
                    fflush(stdout);
                }
            }
        }
    });

    // Work queue
    std::atomic<size_t> next_candidate{0};

    // Worker threads - each slot has 2 threads racing
    std::vector<std::thread> workers;
    for (int slot = 0; slot < num_slots; slot++) {
        workers.emplace_back([&, slot]() {
            std::vector<int> layers;

            while (!g_stop_flag.load()) {
                size_t i = next_candidate.fetch_add(1);
                if (i >= eval_limit) break;

                const auto& [perm, est_val, unused] = candidates[i];
                std::vector<int> w = perm.decode();
                int ell = perm.length();

                // Setup slot progress
                slot_progress[slot].candidate_idx.store(i);
                slot_progress[slot].ell.store(ell);
                slot_progress[slot].descent_iter.store(0);
                slot_progress[slot].cotrans_iter.store(0);
                slot_progress[slot].descent_memo.store(0);
                slot_progress[slot].cotrans_memo.store(0);
                slot_progress[slot].descent_done.store(false);
                slot_progress[slot].cotrans_done.store(false);
                slot_progress[slot].current_perm = w;
                slot_progress[slot].start_time = std::chrono::steady_clock::now();

                double true_val;
                std::string winner;
                bool is_layered = detect_layered(w, layers);

                if (is_layered) {
                    LayeredResult lr = layered_product(layers);
                    true_val = lr.exact_value.convert_to<double>();
                    winner = "product";
                    slot_progress[slot].descent_done.store(true);
                    slot_progress[slot].cotrans_done.store(true);
                } else {
                    // Race descent vs cotrans with 20-minute timeout
                    std::atomic<bool> race_done{false};
                    std::atomic<bool> stop_descent{false};
                    std::atomic<bool> stop_cotrans{false};
                    double descent_val = 0.0, cotrans_val = 0.0;
                    constexpr int TIMEOUT_MINUTES = 20;

                    std::thread descent_thread([&]() {
                        descent_val = schubert_descent_double_verbose(w, stop_descent,
                            slot_progress[slot].descent_iter, slot_progress[slot].descent_memo,
                            slot_progress[slot].descent_ell);
                        slot_progress[slot].descent_done.store(true);
                        if (!race_done.exchange(true)) {
                            stop_cotrans.store(true);
                        }
                    });

                    std::thread cotrans_thread([&]() {
                        cotrans_val = schubert_cotrans_double_verbose(w, stop_cotrans,
                            slot_progress[slot].cotrans_iter, slot_progress[slot].cotrans_memo,
                            slot_progress[slot].cotrans_ell);
                        slot_progress[slot].cotrans_done.store(true);
                        if (!race_done.exchange(true)) {
                            stop_descent.store(true);
                        }
                    });

                    // Timeout watchdog: check every second for 20 minutes
                    auto race_start = std::chrono::steady_clock::now();
                    while (!race_done.load()) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        auto now = std::chrono::steady_clock::now();
                        double race_elapsed = std::chrono::duration<double>(now - race_start).count();
                        if (race_elapsed >= TIMEOUT_MINUTES * 60.0) {
                            // Timeout! Kill both threads
                            stop_descent.store(true);
                            stop_cotrans.store(true);
                            break;
                        }
                    }

                    descent_thread.join();
                    cotrans_thread.join();

                    // Use result from winner (or timeout)
                    bool timed_out = (descent_val == 0.0 && cotrans_val == 0.0);
                    if (timed_out) {
                        true_val = 0.0;
                        winner = "TIMEOUT";
                    } else if (descent_val > 0) {
                        true_val = descent_val;
                        winner = "descent";
                    } else {
                        true_val = cotrans_val;
                        winner = "cotrans";
                    }
                }

                // Store result
                {
                    std::lock_guard<std::mutex> lock(results_mtx);
                    all_evaluated.push_back({i, ell, est_val, true_val, is_layered, winner});
                    if (true_val > global_best) {
                        global_best = true_val;
                        global_best_perm = perm;
                        global_best_length = ell;
                    }
                }

                // Mark slot as idle and report
                slot_progress[slot].candidate_idx.store(SIZE_MAX);
                size_t done = ++completed;
                {
                    std::lock_guard<std::mutex> lock(print_mtx);
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - eval_start).count();
                    printf("\r  [%zu/%zu] ell=%d, est=%.2e, true=%.2e (%s) %.1f/s   ",
                           done, eval_limit, ell, est_val, true_val, winner.c_str(), done / elapsed);
                    fflush(stdout);
                }
            }
        });
    }

    for (auto& w : workers) w.join();
    monitor_stop.store(true);
    monitor_thread.join();
    printf("\n");

    // Check for Ctrl-C interruption during re-evaluation
    if (g_stop_flag.load()) {
        printf("[Interrupted during re-evaluation - completed %zu/%zu]\n", completed.load(), eval_limit);
        if (all_evaluated.empty()) return;
        printf("Showing partial results:\n");
    }

    // Sort results by original index
    std::sort(all_evaluated.begin(), all_evaluated.end(),
        [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    // Find best non-layered
    double best_nonlayered_val = 0.0;
    size_t best_nonlayered_idx = 0;
    int best_nonlayered_ell = 0;
    double best_nonlayered_est = 0.0;
    for (const auto& [idx, ell, est_val, true_val, is_layered, winner] : all_evaluated) {
        if (!is_layered && true_val > best_nonlayered_val) {
            best_nonlayered_val = true_val;
            best_nonlayered_idx = idx;
            best_nonlayered_ell = ell;
            best_nonlayered_est = est_val;
        }
    }

    // Print top results
    for (const auto& [idx, ell, est_val, true_val, is_layered, winner] : all_evaluated) {
        if (idx < 5 || true_val == global_best) {
            printf("  Candidate %zu: ell=%d, est=%.2e, true=%.2e",
                   idx + 1, ell, est_val, true_val);
            if (is_layered) printf(" [layered]");
            if (true_val == global_best) printf(" [BEST]");
            printf("\n");
        }
    }

    // Report best non-layered (compare to known MPPY layered maximum)
    double mppy_layered_val = 0.0;
    if (n >= 1 && n <= 25) {
        const auto& mppy = MPPY_DATA[n - 1];
        LayeredResult lr = layered_product(mppy.layers);
        mppy_layered_val = lr.exact_value.convert_to<double>();
    }
    if (best_nonlayered_val > 0) {
        const auto& [nl_perm, nl_est, nl_unused] = candidates[best_nonlayered_idx];
        std::vector<int> nl_w = nl_perm.decode();
        int nl_cayley = mppy_layered_perm.empty() ? -1 : cayley_distance(nl_w, mppy_layered_perm);
        printf("  Best non-layered: #%zu, ell=%d, d_C=%d, est=%.2e, true=%.2e (%.2fx layered)\n",
               best_nonlayered_idx + 1, best_nonlayered_ell, nl_cayley, best_nonlayered_est, best_nonlayered_val,
               mppy_layered_val > 0 ? best_nonlayered_val / mppy_layered_val : 0.0);
        printf("    Permutation: ");
        print_permutation(nl_w);
        printf("\n");
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::vector<int> w = global_best_perm.decode();
    int result_cayley = mppy_layered_perm.empty() ? -1 : cayley_distance(w, mppy_layered_perm);
    printf("\n--- Result (heuristic beam search) ---\n");
    printf("Winning permutation w = ");
    print_permutation(w);
    printf("\n");
    printf("Length ell(w) = %d\n", global_best_length);
    printf("Cayley distance to MPP layered: %d\n", result_cayley);
    printf("S_w(1^%d) = %.0f\n", n, global_best);
    printf("log2(S_w) / n^2 = %.8f\n", log2(global_best) / (n * n));
    printf("Search time: %.4f seconds | Total time: %.4f seconds\n", search_elapsed, total_elapsed);

    // Compare to MPP3 layered permutation (reuse mppy_layered_val computed earlier)
    if (mppy_layered_val > 0) {
        const auto& mppy = MPPY_DATA[n - 1];
        printf("\n--- Comparison to Layered ---\n");
        printf("Layered w(");
        for (size_t i = 0; i < mppy.layers.size(); i++) {
            printf("%d%s", mppy.layers[i], i + 1 < mppy.layers.size() ? "," : "");
        }
        printf("): %.0f\n", mppy_layered_val);
        printf("Heuristic found: %.0f (%.2fx layered)\n", global_best, global_best / mppy_layered_val);
        if (global_best > mppy_layered_val) {
            printf("*** Heuristic BEAT layered! ***\n");
        } else if (global_best == mppy_layered_val) {
            printf("Heuristic matched layered exactly.\n");
        }
    }
}

void cmd_layered_test(int max_n, Precision prec, Formula formula, bool include_product = false, BfsMode bfs_mode = BfsMode::AUTO) {
    printf("Testing layered permutations (MPP3 data)\n");
    printf("Precision: %s | Formula: %s%s\n\n",
           prec == Precision::DOUBLE ? "double" : "exact",
           formula == Formula::DESCENT ? "descent" :
           formula == Formula::COTRANS ? "cotrans" :
           formula == Formula::TRANSITION ? "transition" : "best (race)",
           include_product ? " | with product comparison" : "");

    int descent_wins = 0, cotrans_wins = 0;

    for (const auto& entry : MPPY_DATA) {
        if (entry.n > max_n) break;
        if (g_stop_flag.load()) break;

        int n = entry.n;
        printf("n=%3d layers=(", n);
        for (size_t i = 0; i < entry.layers.size(); i++) {
            printf("%d%s", entry.layers[i], i < entry.layers.size() - 1 ? "," : "");
        }
        printf("): ");
        fflush(stdout);

        std::vector<int> w = build_layered_permutation(entry.layers);
        std::atomic<bool> stop{false};

        auto start = std::chrono::high_resolution_clock::now();
        double result_d = 0.0;
        mpz_class result_mpz;
        std::string winner = "";

        if (formula == Formula::BEST) {
            ComputeResult res = race_methods(w, prec, true, bfs_mode);
            winner = res.winner;
            if (res.winner == "descent") descent_wins++;
            else cotrans_wins++;
            if (prec == Precision::DOUBLE) {
                result_d = res.double_val;
            } else {
                result_mpz = res.mpz_val;
            }
        } else if (formula == Formula::DESCENT) {
            if (prec == Precision::DOUBLE) {
                result_d = eval_descent_double(w, stop, true, bfs_mode);
            } else {
                Rational r = eval_descent_exact(w, stop, true, bfs_mode);
                result_d = r.to_double();
            }
        } else if (formula == Formula::TRANSITION) {
            if (prec == Precision::DOUBLE) {
                result_d = eval_transition_double(w, stop, true, bfs_mode);
            } else {
                result_mpz = eval_transition_exact(w, stop, true, bfs_mode);
                result_d = mpz_get_d(result_mpz.get_mpz_t());
            }
        } else {
            if (prec == Precision::DOUBLE) {
                result_d = eval_cotrans_double(w, stop, true, bfs_mode);
            } else {
                result_mpz = eval_cotrans_exact(w, stop, true, bfs_mode);
                result_d = mpz_get_d(result_mpz.get_mpz_t());
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        double f_computed = log2(result_d) / (n * n);
        printf("f(n)=%.6f (exp %.6f) | %.4fs",
               f_computed, entry.f_n, elapsed);
        if (!winner.empty()) printf(" [%s]", winner.c_str());
        printf("\n");

        // Product formula comparison (instant, exact)
        if (include_product) {
            auto p_start = std::chrono::high_resolution_clock::now();
            LayeredResult lr = layered_product(entry.layers);
            auto p_end = std::chrono::high_resolution_clock::now();
            double p_elapsed = std::chrono::duration<double>(p_end - p_start).count();
            double pf = lr.log2_value / (n * n);
            std::string val_str = lr.exact_value.str();
            size_t digits = val_str.size();
            printf("      product: f(n)=%.6f | digits=%zu | %.6fs\n", pf, digits, p_elapsed);
        }
    }

    if (formula == Formula::BEST) {
        printf("\nRace summary: descent=%d wins, cotrans=%d wins\n", descent_wins, cotrans_wins);
    }
}

void cmd_layered_eval(const std::vector<int>& layers, Precision prec, Formula formula) {
    if (layers.empty()) {
        fprintf(stderr, "Error: layered tuple is empty\n");
        return;
    }
    for (int b : layers) {
        if (b <= 0) {
            fprintf(stderr, "Error: layer sizes must be positive\n");
            return;
        }
    }

    int n = 0;
    for (int b : layers) n += b;

    printf("Layered permutation w(");
    for (size_t i = 0; i < layers.size(); i++) {
        printf("%d%s", layers[i], i + 1 < layers.size() ? "," : "");
    }
    printf(") | n=%d\n", n);

    // Always use product formula for layered
    auto start = std::chrono::high_resolution_clock::now();
    LayeredResult lr = layered_product(layers);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::string val_str = lr.exact_value.convert_to<std::string>();
    printf("Method: product formula (exact)\n");
    printf("S_w = %s\n", val_str.c_str());
    printf("Digits: %zu\n", val_str.size());
    printf("f(n) = %.6f\n", lr.log2_value / (n * n));
    printf("Time: %.6fs\n", elapsed);
}

// =============================================================================
// Main
// =============================================================================

void print_help(const char* prog) {
    printf("Schubert Polynomial Evaluator - Unified Implementation\n\n");
    printf("Usage: %s [options] <command>\n\n", prog);
    printf("Options:\n");
    printf("  --double         Use double precision (~15 digits) [default]\n");
    printf("  --exact          Use exact arithmetic (rational for descent, GMP for cotrans/transition)\n");
    printf("  --descent        Use descent formula only\n");
    printf("  --cotrans        Use cotransition formula only\n");
    printf("  --transition     Use transition formula only (DFS, no BFS variant)\n");
    printf("  --best           Race both formulas, return first result [default]\n");
    printf("  --threads=N      Use N threads for max search (default: auto-detect)\n");
    printf("  --product        Show product formula comparison in layered_test\n");
    printf("  --no-product     Disable auto-detection of layered permutations (force algorithm)\n");
    printf("  --bfs            Use BFS sort-reduce algorithm (default for n<=25)\n");
    printf("  --dfs            Force DFS memoization algorithm (original method)\n");
    printf("  --no-layered     Exclude layered permutations from heuristic beam search\n");
    printf("  --no-check       Skip re-evaluation in heuristic beam search (show only estimates)\n");
    printf("\n");
    printf("Commands:\n");
    printf("  <perm>                   S_w for given permutation (comma-separated)\n");
    printf("  max:<n>                  Find max S_w over all w in S_n (n<=25)\n");
    printf("  heuristic:<n>:<k>[:<ratio>] Beam search max (default ratio=0.50)\n");
    printf("  layered_test:<n>         Test MPP3 layered permutations up to n\n");
    printf("  layered:<b1,...>         Evaluate layered permutation w(b_k,...,b_1)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s 5,4,3,2,1                      # default: --double --best\n", prog);
    printf("  %s --exact 5,4,3,2,1              # exact, race both\n", prog);
    printf("  %s --double --descent 5,4,3,2,1   # double, descent only\n", prog);
    printf("  %s --threads=4 max:10             # use 4 threads for max search\n", prog);
    printf("  %s layered_test:15                # test layered permutations\n", prog);
    printf("  %s --product layered_test:15      # with product formula comparison\n", prog);
}

int main(int argc, char* argv[]) {
    // Setup signal handler
    signal(SIGINT, signal_handler);

    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_help(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    // Parse options
    Precision prec = Precision::DOUBLE;
    Formula formula = Formula::BEST;
    int num_threads = 0;  // 0 = auto-detect
    bool include_product = false;
    bool no_product = false;  // disable auto-detection of layered permutations
    bool exclude_layered = false;  // exclude layered permutations from heuristic beam
    bool no_check = false;  // skip re-evaluation in heuristic beam search
    BfsMode bfs_mode = BfsMode::AUTO;
    const char* command = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--double") == 0) {
            prec = Precision::DOUBLE;
        } else if (strcmp(argv[i], "--exact") == 0) {
            prec = Precision::EXACT;
        } else if (strcmp(argv[i], "--descent") == 0) {
            formula = Formula::DESCENT;
        } else if (strcmp(argv[i], "--cotrans") == 0) {
            formula = Formula::COTRANS;
        } else if (strcmp(argv[i], "--transition") == 0) {
            formula = Formula::TRANSITION;
        } else if (strcmp(argv[i], "--best") == 0) {
            formula = Formula::BEST;
        } else if (strncmp(argv[i], "--threads=", 10) == 0) {
            num_threads = atoi(argv[i] + 10);
        } else if (strcmp(argv[i], "--product") == 0) {
            include_product = true;
        } else if (strcmp(argv[i], "--no-product") == 0) {
            no_product = true;
        } else if (strcmp(argv[i], "--no-layered") == 0) {
            exclude_layered = true;
        } else if (strcmp(argv[i], "--no-check") == 0) {
            no_check = true;
        } else if (strcmp(argv[i], "--bfs") == 0) {
            bfs_mode = BfsMode::BFS;
        } else if (strcmp(argv[i], "--dfs") == 0) {
            bfs_mode = BfsMode::DFS;
        } else if (argv[i][0] != '-') {
            command = argv[i];
        }
    }

    if (!command) {
        fprintf(stderr, "Error: no command specified\n");
        print_help(argv[0]);
        return 1;
    }

    // Parse command
    if (strncmp(command, "layered_test:", 13) == 0) {
        int max_n = atoi(command + 13);
        if (max_n < 1) max_n = 25;
        cmd_layered_test(max_n, prec, formula, include_product, bfs_mode);
        return 0;
    }

    if (strncmp(command, "layered:", 8) == 0) {
        std::vector<int> layers = parse_permutation(command + 8);
        cmd_layered_eval(layers, prec, formula);
        return 0;
    }

    if (strncmp(command, "verify:", 7) == 0) {
        int n = atoi(command + 7);
        if (n < 1 || n > 8) {
            fprintf(stderr, "Error: verify requires 1 <= n <= 8\n");
            return 1;
        }
        // Test DFS cotrans vs DFS descent vs DFS transition, BFS64 vs DFS, and BFS128 vs DFS on all S_n
        printf("Verifying DFS + BFS64 + BFS128 + transition (double + exact) on all S_%d (%d! = ", n, n);
        int factorial = 1;
        for (int i = 2; i <= n; i++) factorial *= i;
        printf("%d permutations)...\n", factorial);

        std::vector<int> perm(n);
        for (int i = 0; i < n; i++) perm[i] = i + 1;

        std::atomic<bool> stop{false};
        int tested = 0, passed = 0, failed = 0;
        auto start_time = std::chrono::steady_clock::now();

        do {
            double descent_dfs = schubert_descent_double(perm, stop, true);
            double cotrans_dfs = schubert_cotrans_double(perm, stop, true);
            double descent_bfs = schubert_descent_double_bfs(perm, stop, true);
            double cotrans_bfs = schubert_cotrans_double_bfs(perm, stop, true);
            double descent_bfs128 = schubert_descent_double_bfs_128(perm, stop, true);
            double cotrans_bfs128 = schubert_cotrans_double_bfs_128(perm, stop, true);
            double transition_dfs = schubert_transition_double(perm, stop, true);

            // Exact BFS evaluators
            Rational descent_exact_dfs = schubert_descent_rational(perm, stop, true);
            Rational descent_exact_bfs = schubert_descent_rational_bfs(perm, stop, true);
            Rational descent_exact_bfs128 = schubert_descent_rational_bfs_128(perm, stop, true);
            mpz_class cotrans_exact_dfs = schubert_cotrans_exact(perm, stop, true);
            mpz_class cotrans_exact_bfs = schubert_cotrans_exact_bfs(perm, stop, true);
            mpz_class cotrans_exact_bfs128 = schubert_cotrans_exact_bfs_128(perm, stop, true);
            mpz_class transition_exact_dfs = schubert_transition_exact(perm, stop, true);

            tested++;
            bool ok = true;
            // Check all six double values agree
            double ref = descent_dfs;
            auto check = [&](const char* label, double val) {
                double rel_diff = std::abs(val - ref) / std::max(ref, 1.0);
                if (rel_diff >= 1e-9) {
                    ok = false;
                    if (failed < 10) {
                        printf("MISMATCH: w=[");
                        for (int i = 0; i < n; i++) printf("%d%s", perm[i], i < n-1 ? "," : "");
                        printf("] descent_dfs=%.6e %s=%.6e diff=%.2e\n",
                               ref, label, val, rel_diff);
                    }
                }
            };
            check("cotrans_dfs", cotrans_dfs);
            check("descent_bfs", descent_bfs);
            check("cotrans_bfs", cotrans_bfs);
            check("descent_bfs128", descent_bfs128);
            check("cotrans_bfs128", cotrans_bfs128);
            check("transition_dfs", transition_dfs);

            // Check exact values agree (exact comparison, not floating-point tolerance)
            if (!descent_exact_dfs.is_integer()) {
                ok = false;
                if (failed < 10) {
                    printf("NON-INTEGER: w=[");
                    for (int i = 0; i < n; i++) printf("%d%s", perm[i], i < n-1 ? "," : "");
                    printf("] descent_exact_dfs has den!=1\n");
                }
            }
            __int128 exact_ref = descent_exact_dfs.to_int();
            auto check_exact_rational = [&](const char* label, const Rational& val) {
                if (!val.is_integer()) {
                    ok = false;
                    if (failed < 10) {
                        printf("NON-INTEGER: w=[");
                        for (int i = 0; i < n; i++) printf("%d%s", perm[i], i < n-1 ? "," : "");
                        printf("] %s has den!=1\n", label);
                    }
                    return;
                }
                if (val.to_int() != exact_ref) {
                    ok = false;
                    if (failed < 10) {
                        printf("EXACT MISMATCH: w=[");
                        for (int i = 0; i < n; i++) printf("%d%s", perm[i], i < n-1 ? "," : "");
                        printf("] descent_exact_dfs=");
                        print_int128(exact_ref);
                        printf(" %s=", label);
                        print_int128(val.to_int());
                        printf("\n");
                    }
                }
            };
            auto check_exact_mpz = [&](const char* label, const mpz_class& val) {
                mpz_class ref_mpz = cotrans_exact_dfs;
                if (val != ref_mpz) {
                    ok = false;
                    if (failed < 10) {
                        printf("EXACT MISMATCH: w=[");
                        for (int i = 0; i < n; i++) printf("%d%s", perm[i], i < n-1 ? "," : "");
                        gmp_printf("] cotrans_exact_dfs=%Zd %s=%Zd\n",
                                   ref_mpz.get_mpz_t(), label, val.get_mpz_t());
                    }
                }
            };
            check_exact_rational("descent_exact_bfs", descent_exact_bfs);
            check_exact_rational("descent_exact_bfs128", descent_exact_bfs128);
            check_exact_mpz("cotrans_exact_bfs", cotrans_exact_bfs);
            check_exact_mpz("cotrans_exact_bfs128", cotrans_exact_bfs128);
            check_exact_mpz("transition_exact_dfs", transition_exact_dfs);

            // Cross-check: exact descent and cotrans should agree
            if (descent_exact_dfs.is_integer()) {
                mpz_class descent_mpz(int128_to_string(descent_exact_dfs.to_int()));
                if (descent_mpz != cotrans_exact_dfs) {
                    ok = false;
                    if (failed < 10) {
                        printf("EXACT CROSS MISMATCH: w=[");
                        for (int i = 0; i < n; i++) printf("%d%s", perm[i], i < n-1 ? "," : "");
                        gmp_printf("] descent=%Zd cotrans=%Zd\n",
                                   descent_mpz.get_mpz_t(), cotrans_exact_dfs.get_mpz_t());
                    }
                }
            }

            if (ok) passed++;
            else failed++;

            if (tested % 100 == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                printf("\r  %d/%d (%.1f%%) %.1f/s   ", tested, factorial,
                       100.0 * tested / factorial, tested / elapsed);
                fflush(stdout);
            }
        } while (std::next_permutation(perm.begin(), perm.end()));

        auto end_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();

        printf("\n\nResults: %d tested, %d passed, %d failed (%.2fs)\n",
               tested, passed, failed, elapsed);
        if (failed == 0) {
            printf("SUCCESS: DFS, BFS64, BFS128, transition (double + exact) agree on all S_%d\n", n);
        } else {
            printf("FAILURE: %d mismatches found\n", failed);
        }
        return (failed == 0) ? 0 : 1;
    }

    if (strncmp(command, "heuristic:", 10) == 0) {
        // Parse heuristic:<n>:<beam_width>[:<target_ratio>]
        int n = 0;
        size_t beam_width = 0;
        double target_ratio = 0.50;  // default
        const char* p = command + 10;
        n = atoi(p);
        while (*p && *p != ':') p++;
        if (*p == ':') {
            p++;
            beam_width = (size_t)atoll(p);
            while (*p && *p != ':') p++;
            if (*p == ':') {
                p++;
                target_ratio = atof(p);
            }
        }
        if (n < 1 || n > BEAM_MAX_N) {
            fprintf(stderr, "Error: heuristic requires 1 <= n <= %d\n", BEAM_MAX_N);
            return 1;
        }
        if (beam_width < 1) {
            fprintf(stderr, "Error: beam width must be >= 1\n");
            return 1;
        }
        cmd_heuristic(n, beam_width, num_threads, target_ratio, true /* v2 */, exclude_layered, no_check);
        return 0;
    }

    if (strncmp(command, "max:", 4) == 0) {
        int n = atoi(command + 4);
        if (n < 1 || n > 25) {
            fprintf(stderr, "Error: max mode requires 1 <= n <= 25\n");
            return 1;
        }
        cmd_max_optimized(n, num_threads);
        return 0;
    }

    // Require comma-separated permutation
    if (!strchr(command, ',')) {
        fprintf(stderr, "Error: permutation must be comma-separated (e.g., 3,1,4,2)\n");
        return 1;
    }

    std::vector<int> w = parse_permutation(command);
    if (w.empty() || (int)w.size() > MAX_N) {
        fprintf(stderr, "Error: invalid permutation or n > %d\n", MAX_N);
        return 1;
    }

    cmd_eval(w, prec, formula, no_product, bfs_mode);
    return 0;
}
