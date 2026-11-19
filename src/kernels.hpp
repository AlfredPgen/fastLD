// Exact integer cross-product kernels for the dense panel layout.
//
// Layout: SNPs are grouped in panels of 16. For panel p and sample pair q (samples 2q, 2q+1)
// one 64-byte line holds int16 values [lane0 s0, lane0 s1, lane1 s0, lane1 s1, ..., lane15 s1].
// Panel p starts at G + p * pstride (pstride = 32 * number of padded sample pairs).
//
// A kernel call computes, for IP consecutive panels i (16*IP SNPs) and J consecutive SNPs j of
// one panel, the dot products over nq sample pairs, and ADDS them as int64 into
//     L[jj * ldL + ip * 16 + lane]      (jj = 0..J-1, i-SNP = 16*ip + lane)
// The caller guarantees 2 * nq * max|k|^2 < 2^31 so the int32 accumulators cannot overflow.
// Outer-product formulation: each 32-bit lane of an accumulator is one (i, j) pair, so no
// horizontal reductions are needed. Runtime dispatch: AVX-512 VNNI -> AVX2 -> scalar.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <immintrin.h>

namespace ldk {

constexpr int PANEL = 16;

// ---------------------------------------------------------------- AVX-512 VNNI: IP x 8
// add 16 int32 lanes into 16 int64
__attribute__((target("avx512f,avx512bw,avx512vnni"), always_inline)) inline
void vnni_acc(__m512i c, int64_t* d) {
    __m512i lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(c));
    __m512i hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(c, 1));
    _mm512_storeu_si512((void*)d, _mm512_add_epi64(_mm512_loadu_si512((const void*)d), lo));
    _mm512_storeu_si512((void*)(d + 8), _mm512_add_epi64(_mm512_loadu_si512((const void*)(d + 8)), hi));
}

template <int IP>
__attribute__((target("avx512f,avx512bw,avx512vnni"), always_inline)) inline
void vnni_body(const int16_t* A, size_t pstride, const int16_t* B, size_t nq, int64_t* L, size_t ldL) {
    __m512i c00, c01, c02, c03, c04, c05, c06, c07;
    __m512i c10, c11, c12, c13, c14, c15, c16, c17;
    __m512i c20, c21, c22, c23, c24, c25, c26, c27;
    c00 = c01 = c02 = c03 = c04 = c05 = c06 = c07 = _mm512_setzero_si512();
    c10 = c11 = c12 = c13 = c14 = c15 = c16 = c17 = _mm512_setzero_si512();
    c20 = c21 = c22 = c23 = c24 = c25 = c26 = c27 = _mm512_setzero_si512();
    const int16_t* A0 = A;
    const int16_t* A1 = A + pstride;
    const int16_t* A2 = A + 2 * pstride;
    for (size_t q = 0; q < nq; q++) {
        const size_t o = q * 32;
        const int32_t* b = reinterpret_cast<const int32_t*>(B + o);
        __m512i a0 = _mm512_load_si512((const void*)(A0 + o)), a1, a2;
        if (IP > 1) a1 = _mm512_load_si512((const void*)(A1 + o));
        if (IP > 2) a2 = _mm512_load_si512((const void*)(A2 + o));
// in-place accumulate with a broadcast memory operand; inline asm because GCC otherwise copies
// the accumulators around every vpdpwssd (review finding F1)
#define LDK_DP(acc, a, mem) asm("vpdpwssd %2%{1to16%}, %1, %0" : "+v"(acc) : "v"(a), "m"(mem))
#define LDK_STEP(jj)                                                              \
        {                                                                         \
            LDK_DP(c0##jj, a0, b[jj]);                                            \
            if (IP > 1) LDK_DP(c1##jj, a1, b[jj]);                                \
            if (IP > 2) LDK_DP(c2##jj, a2, b[jj]);                                \
        }
        LDK_STEP(0) LDK_STEP(1) LDK_STEP(2) LDK_STEP(3) LDK_STEP(4) LDK_STEP(5) LDK_STEP(6) LDK_STEP(7)
#undef LDK_STEP
#undef LDK_DP
    }
#define acc(c, jj, ip) vnni_acc(c, L + (jj) * ldL + (ip) * 16)
    acc(c00, 0, 0); acc(c01, 1, 0); acc(c02, 2, 0); acc(c03, 3, 0); acc(c04, 4, 0); acc(c05, 5, 0); acc(c06, 6, 0); acc(c07, 7, 0);
    if (IP > 1) { acc(c10, 0, 1); acc(c11, 1, 1); acc(c12, 2, 1); acc(c13, 3, 1); acc(c14, 4, 1); acc(c15, 5, 1); acc(c16, 6, 1); acc(c17, 7, 1); }
    if (IP > 2) { acc(c20, 0, 2); acc(c21, 1, 2); acc(c22, 2, 2); acc(c23, 3, 2); acc(c24, 4, 2); acc(c25, 5, 2); acc(c26, 6, 2); acc(c27, 7, 2); }
#undef acc
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
static void dense_vnni(int IP, const int16_t* A, size_t pstride, const int16_t* B, size_t nq, int64_t* L, size_t ldL) {
    if (IP == 3) vnni_body<3>(A, pstride, B, nq, L, ldL);
    else if (IP == 2) vnni_body<2>(A, pstride, B, nq, L, ldL);
    else vnni_body<1>(A, pstride, B, nq, L, ldL);
}

// ---------------------------------------------------------------- AVX2: 1 panel x 4, called for IP panels x 8
__attribute__((target("avx2"), always_inline)) inline
void avx2_acc(__m256i c, int64_t* d) {
    __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(c));
    __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(c, 1));
    _mm256_storeu_si256((__m256i*)d, _mm256_add_epi64(_mm256_loadu_si256((const __m256i*)d), lo));
    _mm256_storeu_si256((__m256i*)(d + 4), _mm256_add_epi64(_mm256_loadu_si256((const __m256i*)(d + 4)), hi));
}

__attribute__((target("avx2"), always_inline)) inline
void avx2_body(const int16_t* A, const int16_t* B, size_t nq, int64_t* L, size_t ldL) {
    __m256i c00, c01, c02, c03, c10, c11, c12, c13;   // c<half><jj>
    c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = _mm256_setzero_si256();
    for (size_t q = 0; q < nq; q++) {
        const size_t o = q * 32;
        const int32_t* b = reinterpret_cast<const int32_t*>(B + o);
        __m256i a0 = _mm256_load_si256((const __m256i*)(A + o));
        __m256i a1 = _mm256_load_si256((const __m256i*)(A + o + 16));
#define LDK_ADD2(acc, t) asm("vpaddd %1, %0, %0" : "+x"(acc) : "x"(t))
#define LDK_STEP2(jj)                                                             \
        {                                                                         \
            __m256i bb = _mm256_set1_epi32(b[jj]);                                \
            __m256i t0 = _mm256_madd_epi16(a0, bb), t1 = _mm256_madd_epi16(a1, bb); \
            LDK_ADD2(c0##jj, t0); LDK_ADD2(c1##jj, t1);                           \
        }
        LDK_STEP2(0) LDK_STEP2(1) LDK_STEP2(2) LDK_STEP2(3)
#undef LDK_STEP2
#undef LDK_ADD2
    }
#define acc(c, jj, half) avx2_acc(c, L + (jj) * ldL + (half) * 8)
    acc(c00, 0, 0); acc(c01, 1, 0); acc(c02, 2, 0); acc(c03, 3, 0);
    acc(c10, 0, 1); acc(c11, 1, 1); acc(c12, 2, 1); acc(c13, 3, 1);
#undef acc
}

__attribute__((target("avx2")))
static void dense_avx2(int IP, const int16_t* A, size_t pstride, const int16_t* B, size_t nq, int64_t* L, size_t ldL) {
    for (int ip = 0; ip < IP; ip++)
        for (int jh = 0; jh < 2; jh++)
            avx2_body(A + ip * pstride, B + jh * 8, nq, L + jh * 4 * ldL + ip * 16, ldL);
}

// ---------------------------------------------------------------- portable scalar
static void dense_scalar(int IP, const int16_t* A, size_t pstride, const int16_t* B, size_t nq, int64_t* L, size_t ldL) {
    for (int ip = 0; ip < IP; ip++)
        for (int jj = 0; jj < 8; jj++) {
            int32_t c[16] = {0};
            const int16_t* a = A + ip * pstride;
            for (size_t q = 0; q < nq; q++) {
                const int16_t b0 = B[q * 32 + 2 * jj], b1 = B[q * 32 + 2 * jj + 1];
                for (int l = 0; l < 16; l++) c[l] += int32_t(a[q * 32 + 2 * l]) * b0 + int32_t(a[q * 32 + 2 * l + 1]) * b1;
            }
            for (int l = 0; l < 16; l++) L[jj * ldL + ip * 16 + l] += c[l];
        }
}

using DenseFn = void (*)(int, const int16_t*, size_t, const int16_t*, size_t, int64_t*, size_t);

static inline bool cpu_vnni() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vnni");
}
static inline bool cpu_avx2() { __builtin_cpu_init(); return __builtin_cpu_supports("avx2"); }

// FASTLD_KERNEL=avx2|scalar forces a slower kernel (testing)
static inline DenseFn pick_dense(const char** name) {
    const char* force = getenv("FASTLD_KERNEL");
    if (force && !strcmp(force, "scalar")) { *name = "scalar"; return dense_scalar; }
    if (force && !strcmp(force, "avx2") && cpu_avx2()) { *name = "AVX2"; return dense_avx2; }
    if (cpu_vnni()) { *name = "AVX-512 VNNI"; return dense_vnni; }
    if (cpu_avx2()) { *name = "AVX2"; return dense_avx2; }
    *name = "scalar"; return dense_scalar;
}

// ================================================================ dense x sparse
// For the panels p = 0..NP-1 starting at G (pstride apart) and one sparse SNP given by its
// non-zero entries (sample idx[t], value val[t]), add  sum_t k_p,lane[idx[t]] * val[t]  into the
// int32 accumulators D[p*16 + lane]. The broadcast word (v,0) or (0,v) selects the sample of the
// pair, so one vpdpwssd handles 16 dense SNPs. Caller bounds the number of non-zeros between
// flushes so int32 cannot overflow.
static inline uint32_t ds_word(uint32_t s, int16_t v) {
    return (s & 1) ? (uint32_t(uint16_t(v)) << 16) : uint32_t(uint16_t(v));
}

template <int NP>
__attribute__((target("avx512f,avx512bw,avx512vnni"), always_inline)) inline
void ds_vnni_body(const int16_t* G, size_t pstride, const uint32_t* idx, const int16_t* val, size_t cnt, int32_t* D) {
    __m512i c[NP];
#pragma GCC unroll 32
    for (int p = 0; p < NP; p++) c[p] = _mm512_loadu_si512((const void*)(D + 16 * p));
    for (size_t t = 0; t < cnt; t++) {
        const uint32_t s = idx[t];
        const __m512i b = _mm512_set1_epi32((int)ds_word(s, val[t]));
        const int16_t* line = G + size_t(s >> 1) * 32;
#pragma GCC unroll 32
        for (int p = 0; p < NP; p++)
            asm("vpdpwssd %2, %1, %0" : "+v"(c[p]) : "v"(b), "m"(*(const __m512i*)(line + p * pstride)));
    }
#pragma GCC unroll 32
    for (int p = 0; p < NP; p++) _mm512_storeu_si512((void*)(D + 16 * p), c[p]);
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
static void ds_vnni(int np, const int16_t* G, size_t pstride, const uint32_t* idx, const int16_t* val, size_t cnt, int32_t* D) {
    int p = 0;
    for (; p + 24 <= np; p += 24) ds_vnni_body<24>(G + p * pstride, pstride, idx, val, cnt, D + 16 * p);
    for (; p + 8 <= np; p += 8) ds_vnni_body<8>(G + p * pstride, pstride, idx, val, cnt, D + 16 * p);
    for (; p < np; p++) ds_vnni_body<1>(G + p * pstride, pstride, idx, val, cnt, D + 16 * p);
}

template <int NP>
__attribute__((target("avx2"), always_inline)) inline
void ds_avx2_body(const int16_t* G, size_t pstride, const uint32_t* idx, const int16_t* val, size_t cnt, int32_t* D) {
    __m256i c[2 * NP];
#pragma GCC unroll 16
    for (int h = 0; h < 2 * NP; h++) c[h] = _mm256_loadu_si256((const __m256i*)(D + 8 * h));
    for (size_t t = 0; t < cnt; t++) {
        const uint32_t s = idx[t];
        const __m256i b = _mm256_set1_epi32((int)ds_word(s, val[t]));
        const int16_t* line = G + size_t(s >> 1) * 32;
#pragma GCC unroll 16
        for (int h = 0; h < 2 * NP; h++) {
            __m256i m = _mm256_madd_epi16(_mm256_load_si256((const __m256i*)(line + (h >> 1) * pstride + (h & 1) * 16)), b);
            asm("vpaddd %1, %0, %0" : "+x"(c[h]) : "x"(m));
        }
    }
#pragma GCC unroll 16
    for (int h = 0; h < 2 * NP; h++) _mm256_storeu_si256((__m256i*)(D + 8 * h), c[h]);
}

__attribute__((target("avx2")))
static void ds_avx2(int np, const int16_t* G, size_t pstride, const uint32_t* idx, const int16_t* val, size_t cnt, int32_t* D) {
    int p = 0;
    for (; p + 6 <= np; p += 6) ds_avx2_body<6>(G + p * pstride, pstride, idx, val, cnt, D + 16 * p);
    for (; p < np; p++) ds_avx2_body<1>(G + p * pstride, pstride, idx, val, cnt, D + 16 * p);
}

static void ds_scalar(int np, const int16_t* G, size_t pstride, const uint32_t* idx, const int16_t* val, size_t cnt, int32_t* D) {
    for (int p = 0; p < np; p++)
        for (size_t t = 0; t < cnt; t++) {
            const uint32_t s = idx[t];
            const int16_t* line = G + p * pstride + size_t(s >> 1) * 32 + (s & 1);
            for (int l = 0; l < 16; l++) D[16 * p + l] += int32_t(line[2 * l]) * val[t];
        }
}

using SparseFn = void (*)(int, const int16_t*, size_t, const uint32_t*, const int16_t*, size_t, int32_t*);

static inline SparseFn pick_sparse() {
    const char* force = getenv("FASTLD_KERNEL");
    if (force && !strcmp(force, "scalar")) return ds_scalar;
    if (force && !strcmp(force, "avx2") && cpu_avx2()) return ds_avx2;
    if (cpu_vnni()) return ds_vnni;
    if (cpu_avx2()) return ds_avx2;
    return ds_scalar;
}

// ================================================================ hard calls: bit planes
// Per SNP: p1 = (g >= 1), p2 = (g == 2), g in {0,1,2} (missing: both 0), optional observed mask o.
// For one SNP j and NI SNPs i (NI <= 8), over nw 64-bit words (nw % 8 == 0), add into out[i*4..]:
//   [0] sum g_i g_j  = popc(x) + popc(x & (p2i ^ p2j)) + 3 popc(x & p2i & p2j),  x = p1i & p1j
//   [1] sum of g_i over samples where j is observed    (MISS only)
//   [2] sum of g_j over samples where i is observed    (MISS only)
//   [3] number of samples observed in both             (MISS only)
struct BitArgs {
    const uint64_t* p1i[8]; const uint64_t* p2i[8]; const uint64_t* oi[8];
    const uint64_t *p1j, *p2j, *oj;
    int NI; size_t nw;
    int64_t* out;   // NI * 4
};

template <bool MISS>
__attribute__((target("avx512f,avx512vpopcntdq"))) inline
void bits_vpopcnt_body(const BitArgs& g) {
    __m512i c1[8], c2[8], c3[8], ai[8], aj[8], nij[8];
    for (int i = 0; i < 8; i++) c1[i] = c2[i] = c3[i] = ai[i] = aj[i] = nij[i] = _mm512_setzero_si512();
    for (size_t w = 0; w < g.nw; w += 8) {
        const __m512i a1 = _mm512_load_si512((const void*)(g.p1j + w)), a2 = _mm512_load_si512((const void*)(g.p2j + w));
        __m512i oj = MISS ? _mm512_load_si512((const void*)(g.oj + w)) : _mm512_setzero_si512();
        for (int i = 0; i < g.NI; i++) {
            const __m512i b1 = _mm512_load_si512((const void*)(g.p1i[i] + w)), b2 = _mm512_load_si512((const void*)(g.p2i[i] + w));
            const __m512i x = _mm512_and_si512(a1, b1);
            const __m512i t1 = _mm512_ternarylogic_epi64(x, a2, b2, 0x60);   // x & (a2 ^ b2): truth-table bits 6 (110) and 5 (101)
            const __m512i t2 = _mm512_ternarylogic_epi64(x, a2, b2, 0x80);   // x & a2 & b2
            c1[i] = _mm512_add_epi64(c1[i], _mm512_popcnt_epi64(x));
            c2[i] = _mm512_add_epi64(c2[i], _mm512_popcnt_epi64(t1));
            c3[i] = _mm512_add_epi64(c3[i], _mm512_popcnt_epi64(t2));
            if (MISS) {
                const __m512i oi = _mm512_load_si512((const void*)(g.oi[i] + w));
                ai[i] = _mm512_add_epi64(ai[i], _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(b1, oj)), _mm512_popcnt_epi64(_mm512_and_si512(b2, oj))));
                aj[i] = _mm512_add_epi64(aj[i], _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(a1, oi)), _mm512_popcnt_epi64(_mm512_and_si512(a2, oi))));
                nij[i] = _mm512_add_epi64(nij[i], _mm512_popcnt_epi64(_mm512_and_si512(oi, oj)));
            }
        }
    }
    for (int i = 0; i < g.NI; i++) {
        g.out[4 * i] += _mm512_reduce_add_epi64(c1[i]) + _mm512_reduce_add_epi64(c2[i]) + 3 * _mm512_reduce_add_epi64(c3[i]);
        if (MISS) {
            g.out[4 * i + 1] += _mm512_reduce_add_epi64(ai[i]);
            g.out[4 * i + 2] += _mm512_reduce_add_epi64(aj[i]);
            g.out[4 * i + 3] += _mm512_reduce_add_epi64(nij[i]);
        }
    }
}
__attribute__((target("avx512f,avx512vpopcntdq")))
static void bits_vpopcnt(const BitArgs& g, bool miss) { if (miss) bits_vpopcnt_body<true>(g); else bits_vpopcnt_body<false>(g); }

// AVX2: nibble-LUT popcount (Mula), byte counts summed with vpsadbw
__attribute__((target("avx2"), always_inline)) inline __m256i popcnt256(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(v, low), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));   // per-byte counts 0..8
}
__attribute__((target("avx2"), always_inline)) inline int64_t hsum64_256(__m256i v) {
    alignas(32) int64_t t[4]; _mm256_store_si256((__m256i*)t, v); return t[0] + t[1] + t[2] + t[3];
}
template <bool MISS>
__attribute__((target("avx2"))) inline
void bits_avx2_body(const BitArgs& g) {
    const __m256i zero = _mm256_setzero_si256();
    int64_t acc[8][6] = {};
    // byte counters are flushed with vpsadbw every 8 words-of-256 (each byte <= 8*... safe: max 3*8*8 < 255)
    for (int i = 0; i < g.NI; i++) {
        __m256i s1 = zero, s2 = zero, s3 = zero, sa = zero, sb = zero, sn = zero;
        for (size_t w = 0; w < g.nw; w += 4) {
            const __m256i a1 = _mm256_load_si256((const __m256i*)(g.p1j + w)), a2 = _mm256_load_si256((const __m256i*)(g.p2j + w));
            const __m256i b1 = _mm256_load_si256((const __m256i*)(g.p1i[i] + w)), b2 = _mm256_load_si256((const __m256i*)(g.p2i[i] + w));
            const __m256i x = _mm256_and_si256(a1, b1);
            const __m256i t1 = _mm256_and_si256(x, _mm256_xor_si256(a2, b2));
            const __m256i t2 = _mm256_and_si256(x, _mm256_and_si256(a2, b2));
            s1 = _mm256_add_epi64(s1, _mm256_sad_epu8(popcnt256(x), zero));
            s2 = _mm256_add_epi64(s2, _mm256_sad_epu8(popcnt256(t1), zero));
            s3 = _mm256_add_epi64(s3, _mm256_sad_epu8(popcnt256(t2), zero));
            if (MISS) {
                const __m256i oj = _mm256_load_si256((const __m256i*)(g.oj + w)), oi = _mm256_load_si256((const __m256i*)(g.oi[i] + w));
                sa = _mm256_add_epi64(sa, _mm256_sad_epu8(_mm256_add_epi8(popcnt256(_mm256_and_si256(b1, oj)), popcnt256(_mm256_and_si256(b2, oj))), zero));
                sb = _mm256_add_epi64(sb, _mm256_sad_epu8(_mm256_add_epi8(popcnt256(_mm256_and_si256(a1, oi)), popcnt256(_mm256_and_si256(a2, oi))), zero));
                sn = _mm256_add_epi64(sn, _mm256_sad_epu8(popcnt256(_mm256_and_si256(oi, oj)), zero));
            }
        }
#define h hsum64_256
        acc[i][0] = h(s1) + h(s2) + 3 * h(s3);
        if (MISS) { acc[i][1] = h(sa); acc[i][2] = h(sb); acc[i][3] = h(sn); }
#undef h
    }
    for (int i = 0; i < g.NI; i++) {
        g.out[4 * i] += acc[i][0];
        if (MISS) { g.out[4 * i + 1] += acc[i][1]; g.out[4 * i + 2] += acc[i][2]; g.out[4 * i + 3] += acc[i][3]; }
    }
}
__attribute__((target("avx2")))
static void bits_avx2(const BitArgs& g, bool miss) { if (miss) bits_avx2_body<true>(g); else bits_avx2_body<false>(g); }

static void bits_scalar(const BitArgs& g, bool miss) {
    for (int i = 0; i < g.NI; i++) {
        int64_t c = 0, a = 0, b = 0, nn = 0;
        for (size_t w = 0; w < g.nw; w++) {
            const uint64_t a1 = g.p1j[w], a2 = g.p2j[w], b1 = g.p1i[i][w], b2 = g.p2i[i][w];
            const uint64_t x = a1 & b1;
            c += __builtin_popcountll(x) + __builtin_popcountll(x & (a2 ^ b2)) + 3 * __builtin_popcountll(x & a2 & b2);
            if (miss) {
                const uint64_t oj = g.oj[w], oi = g.oi[i][w];
                a += __builtin_popcountll(b1 & oj) + __builtin_popcountll(b2 & oj);
                b += __builtin_popcountll(a1 & oi) + __builtin_popcountll(a2 & oi);
                nn += __builtin_popcountll(oi & oj);
            }
        }
        g.out[4 * i] += c;
        if (miss) { g.out[4 * i + 1] += a; g.out[4 * i + 2] += b; g.out[4 * i + 3] += nn; }
    }
}

// Register-blocked no-missing tile: 4 i-SNPs x 2 j-SNPs over nw words (nw % 8 == 0).
// out[jj*ldo + ii] += sum g_i g_j. Two accumulators per pair: popc(x)+popc(x&(a2^b2)) and
// popc(x&a2&b2) (weight 3); 8 accumulators + 12 plane registers stay in zmm registers.
__attribute__((target("avx512f,avx512vpopcntdq")))
static void bits_tile42_vpopcnt(const uint64_t* const p1i[4], const uint64_t* const p2i[4], const uint64_t* const p1j[2],
                                const uint64_t* const p2j[2], size_t nw, int64_t* out, size_t ldo) {
    __m512i u00 = _mm512_setzero_si512(), u01 = u00, u10 = u00, u11 = u00, u20 = u00, u21 = u00, u30 = u00, u31 = u00;
    __m512i v00 = u00, v01 = u00, v10 = u00, v11 = u00, v20 = u00, v21 = u00, v30 = u00, v31 = u00;
    for (size_t w = 0; w < nw; w += 8) {
        const __m512i a10 = _mm512_load_si512((const void*)(p1j[0] + w)), a20 = _mm512_load_si512((const void*)(p2j[0] + w));
        const __m512i a11 = _mm512_load_si512((const void*)(p1j[1] + w)), a21 = _mm512_load_si512((const void*)(p2j[1] + w));
#define LDK_BP(ii)                                                                                          \
        {                                                                                                   \
            const __m512i b1 = _mm512_load_si512((const void*)(p1i[ii] + w));                               \
            const __m512i b2 = _mm512_load_si512((const void*)(p2i[ii] + w));                               \
            __m512i x = _mm512_and_si512(a10, b1);                                                          \
            u##ii##0 = _mm512_add_epi64(u##ii##0, _mm512_add_epi64(_mm512_popcnt_epi64(x),                  \
                         _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(x, a20, b2, 0x60))));                \
            v##ii##0 = _mm512_add_epi64(v##ii##0, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(x, a20, b2, 0x80))); \
            x = _mm512_and_si512(a11, b1);                                                                  \
            u##ii##1 = _mm512_add_epi64(u##ii##1, _mm512_add_epi64(_mm512_popcnt_epi64(x),                  \
                         _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(x, a21, b2, 0x60))));                \
            v##ii##1 = _mm512_add_epi64(v##ii##1, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(x, a21, b2, 0x80))); \
        }
        LDK_BP(0) LDK_BP(1) LDK_BP(2) LDK_BP(3)
#undef LDK_BP
    }
#define LDK_BO(ii, jj) out[jj * ldo + ii] += _mm512_reduce_add_epi64(u##ii##jj) + 3 * _mm512_reduce_add_epi64(v##ii##jj);
    LDK_BO(0, 0) LDK_BO(1, 0) LDK_BO(2, 0) LDK_BO(3, 0) LDK_BO(0, 1) LDK_BO(1, 1) LDK_BO(2, 1) LDK_BO(3, 1)
#undef LDK_BO
}

// Register-blocked tile with missing data: 2 i-SNPs x 2 j-SNPs over nw words (nw % 8 == 0).
// out[(jj*ldo + ii)*4 + {0,1,2,3}] += {sum g_i g_j, sum g_i over o_j, sum g_j over o_i, popc(o_i & o_j)}
// 16 accumulators + 12 plane registers.
__attribute__((target("avx512f,avx512vpopcntdq")))
static void bits_tile22m_vpopcnt(const uint64_t* const p1i[2], const uint64_t* const p2i[2], const uint64_t* const oi[2],
                                 const uint64_t* const p1j[2], const uint64_t* const p2j[2], const uint64_t* const oj[2],
                                 size_t nw, int64_t* out, size_t ldo) {
    __m512i c00 = _mm512_setzero_si512(), c01 = c00, c10 = c00, c11 = c00;
    __m512i a00 = c00, a01 = c00, a10 = c00, a11 = c00;   // sum g_i over o_j
    __m512i b00 = c00, b01 = c00, b10 = c00, b11 = c00;   // sum g_j over o_i
    __m512i n00 = c00, n01 = c00, n10 = c00, n11 = c00;
    for (size_t w = 0; w < nw; w += 8) {
        const __m512i j10 = _mm512_load_si512((const void*)(p1j[0] + w)), j20 = _mm512_load_si512((const void*)(p2j[0] + w)), jo0 = _mm512_load_si512((const void*)(oj[0] + w));
        const __m512i j11 = _mm512_load_si512((const void*)(p1j[1] + w)), j21 = _mm512_load_si512((const void*)(p2j[1] + w)), jo1 = _mm512_load_si512((const void*)(oj[1] + w));
#define LDK_MP(ii, jj)                                                                                       \
        {                                                                                                    \
            const __m512i x = _mm512_and_si512(j1##jj, i1);                                                  \
            const __m512i t2 = _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(x, j2##jj, i2, 0x80));          \
            c##ii##jj = _mm512_add_epi64(c##ii##jj, _mm512_add_epi64(                                        \
                _mm512_add_epi64(_mm512_popcnt_epi64(x), _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(x, j2##jj, i2, 0x60))), \
                _mm512_add_epi64(t2, _mm512_slli_epi64(t2, 1))));                                            \
            a##ii##jj = _mm512_add_epi64(a##ii##jj, _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(i1, jo##jj)), \
                                                                     _mm512_popcnt_epi64(_mm512_and_si512(i2, jo##jj)))); \
            b##ii##jj = _mm512_add_epi64(b##ii##jj, _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(j1##jj, io)), \
                                                                     _mm512_popcnt_epi64(_mm512_and_si512(j2##jj, io)))); \
            n##ii##jj = _mm512_add_epi64(n##ii##jj, _mm512_popcnt_epi64(_mm512_and_si512(io, jo##jj)));    \
        }
        {
            const __m512i i1 = _mm512_load_si512((const void*)(p1i[0] + w)), i2 = _mm512_load_si512((const void*)(p2i[0] + w)), io = _mm512_load_si512((const void*)(oi[0] + w));
            LDK_MP(0, 0) LDK_MP(0, 1)
        }
        {
            const __m512i i1 = _mm512_load_si512((const void*)(p1i[1] + w)), i2 = _mm512_load_si512((const void*)(p2i[1] + w)), io = _mm512_load_si512((const void*)(oi[1] + w));
            LDK_MP(1, 0) LDK_MP(1, 1)
        }
#undef LDK_MP
    }
#define LDK_MO(ii, jj)                                                       \
    {                                                                        \
        int64_t* o = out + ((jj) * ldo + (ii)) * 4;                          \
        o[0] += _mm512_reduce_add_epi64(c##ii##jj);                          \
        o[1] += _mm512_reduce_add_epi64(a##ii##jj);                          \
        o[2] += _mm512_reduce_add_epi64(b##ii##jj);                          \
        o[3] += _mm512_reduce_add_epi64(n##ii##jj);                          \
    }
    LDK_MO(0, 0) LDK_MO(1, 0) LDK_MO(0, 1) LDK_MO(1, 1)
#undef LDK_MO
}

using BitsFn = void (*)(const BitArgs&, bool);
static inline bool cpu_vpopcnt() { __builtin_cpu_init(); return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vpopcntdq"); }
static inline BitsFn pick_bits(const char** name) {
    const char* force = getenv("FASTLD_KERNEL");
    if (force && !strcmp(force, "scalar")) { *name = "scalar popcnt"; return bits_scalar; }
    if (force && !strcmp(force, "avx2") && cpu_avx2()) { *name = "AVX2 popcount"; return bits_avx2; }
    if (cpu_vpopcnt()) { *name = "AVX-512 VPOPCNTDQ"; return bits_vpopcnt; }
    if (cpu_avx2()) { *name = "AVX2 popcount"; return bits_avx2; }
    *name = "scalar popcnt"; return bits_scalar;
}

// ================================================================ bit-plane packing (decode side)
// p1 bit = (k != 0), p2 bit = (k == two), o bit = !miss (o may be null). n samples -> ceil(n/64) words.
__attribute__((target("avx512f,avx512bw")))
static void pack_planes_avx512(const int32_t* k, const uint8_t* miss, size_t n, int32_t two, uint64_t* p1, uint64_t* p2, uint64_t* o) {
    const __m512i z = _mm512_setzero_si512(), t = _mm512_set1_epi32(two);
    size_t w = 0;
    for (; (w + 1) * 64 <= n; w++) {
        uint64_t a = 0, b = 0;
        for (int q = 0; q < 4; q++) {
            const __m512i v = _mm512_loadu_si512((const void*)(k + w * 64 + q * 16));
            a |= uint64_t(_mm512_cmpneq_epi32_mask(v, z)) << (16 * q);
            b |= uint64_t(_mm512_cmpeq_epi32_mask(v, t)) << (16 * q);
        }
        p1[w] = a; p2[w] = b;
        if (o) o[w] = ~uint64_t(_mm512_test_epi8_mask(_mm512_loadu_si512((const void*)(miss + w * 64)), _mm512_set1_epi8(1)));
    }
    for (size_t i = w * 64; i < n; i++) {
        p1[i >> 6] |= uint64_t(k[i] != 0) << (i & 63); p2[i >> 6] |= uint64_t(k[i] == two) << (i & 63);
        if (o && !miss[i]) o[i >> 6] |= uint64_t(1) << (i & 63);
    }
}
__attribute__((target("avx2")))
static void pack_planes_avx2(const int32_t* k, const uint8_t* miss, size_t n, int32_t two, uint64_t* p1, uint64_t* p2, uint64_t* o) {
    const __m256i z = _mm256_setzero_si256(), t = _mm256_set1_epi32(two);
    size_t w = 0;
    for (; (w + 1) * 64 <= n; w++) {
        uint64_t a = 0, b = 0;
        for (int q = 0; q < 8; q++) {
            const __m256i v = _mm256_loadu_si256((const __m256i*)(k + w * 64 + q * 8));
            a |= uint64_t(uint8_t(~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, z))))) << (8 * q);
            b |= uint64_t(uint8_t(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, t))))) << (8 * q);
        }
        p1[w] = a; p2[w] = b;
        if (o) {
            uint64_t m = 0;
            for (int q = 0; q < 2; q++) {
                const __m256i v = _mm256_loadu_si256((const __m256i*)(miss + w * 64 + q * 32));
                m |= uint64_t(uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_setzero_si256())))) << (32 * q);
            }
            o[w] = m;
        }
    }
    for (size_t i = w * 64; i < n; i++) {
        p1[i >> 6] |= uint64_t(k[i] != 0) << (i & 63); p2[i >> 6] |= uint64_t(k[i] == two) << (i & 63);
        if (o && !miss[i]) o[i >> 6] |= uint64_t(1) << (i & 63);
    }
}
static void pack_planes_scalar(const int32_t* k, const uint8_t* miss, size_t n, int32_t two, uint64_t* p1, uint64_t* p2, uint64_t* o) {
    for (size_t w = 0; w * 64 < n; w++) {
        uint64_t a = 0, b = 0, c = 0;
        const size_t e = std::min<size_t>(64, n - w * 64);
        for (size_t t = 0; t < e; t++) {
            const int32_t x = k[w * 64 + t];
            a |= uint64_t(x != 0) << t; b |= uint64_t(x == two) << t; c |= uint64_t(miss[w * 64 + t] == 0) << t;
        }
        p1[w] = a; p2[w] = b; if (o) o[w] = c;
    }
}
using PackFn = void (*)(const int32_t*, const uint8_t*, size_t, int32_t, uint64_t*, uint64_t*, uint64_t*);
static inline PackFn pick_pack() {
    __builtin_cpu_init();
    const char* force = getenv("FASTLD_KERNEL");
    if (force && !strcmp(force, "scalar")) return pack_planes_scalar;
    if (force && !strcmp(force, "avx2") && cpu_avx2()) return pack_planes_avx2;
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) return pack_planes_avx512;
    if (cpu_avx2()) return pack_planes_avx2;
    return pack_planes_scalar;
}

static inline std::string compiled_kernels() {
    std::string s = "dense int16: AVX-512 VNNI";
    s += cpu_vnni() ? " (usable on this CPU)" : " (not usable on this CPU)";
    s += ", AVX2"; s += cpu_avx2() ? " (usable)" : " (not usable)";
    s += ", scalar (usable)";
    s += "; dense x sparse: same set";
    s += "; hard-call bit planes: AVX-512 VPOPCNTDQ"; s += cpu_vpopcnt() ? " (usable)" : " (not usable)";
    s += ", AVX2"; s += cpu_avx2() ? " (usable)" : " (not usable)";
    s += ", scalar popcnt (usable)";
    return s;
}

}  // namespace ldk
