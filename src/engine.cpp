// LD engine.
//
// Storage: every active SNP (not NA, not monomorphic) gets a "slot" in Z-file order. Dense
// slots live in 16-SNP panels interleaved by sample pair (see kernels.hpp); missing values are
// stored as 0 and listed per SNP. Integer path results go into a packed triangle (PackedTri),
// the floating-point (dsyrk) path into a full matrix.
//
// Exactness contract: the integer path computes the exact integer sums of the baseline and
// then evaluates r exactly as the baseline does:
//   num = n*C_ij - sx_i*sx_j (int128),  V = n*sxx - sx^2 (int128),
//   r = double(num) / sqrt(double(Vi) * double(Vj))
#include <omp.h>
#include <malloc.h>
#include <sys/mman.h>

#include "engine.hpp"
#include "kernels.hpp"

extern "C" {
#include <cblas.h>
}

static long rss_mb() {   // for FASTLD_PROFILE
    FILE* f = fopen("/proc/self/statm", "r"); long a = 0, b = 0;
    if (f) { if (fscanf(f, "%ld %ld", &a, &b) != 2) b = 0; fclose(f); }
    return b * 4096 / (1 << 20);
}
static size_t resident_mb(const void* p, size_t bytes) {
    size_t pg = (bytes + 4095) / 4096; vector<unsigned char> mv(pg);
    if (mincore(const_cast<void*>(p), bytes, mv.data()) != 0) return 0;
    size_t res = 0; for (auto c : mv) res += c & 1;
    return res * 4096 >> 20;
}
static size_t env_size(const char* name, size_t def) {
    const char* v = getenv(name);
    return v && *v ? strtoull(v, nullptr, 10) : def;
}

static void set_stats(const Var& v, size_t n, double sample_miss, SnpStats& s) {
    s.sx = v.sx; s.sxx = v.sxx; s.n_miss = v.n_miss; s.den = v.den; s.bits = v.bits;
    s.hard = v.hard; s.hard_d = v.hard_d;
    const size_t nobs = n - v.n_miss;
    __int128 ssn = (__int128)nobs * v.sxx - (__int128)v.sx * v.sx;  // = nobs * sum (k - mean)^2, exact
    s.mono = nobs == 0 || ssn <= 0;
    s.mean = nobs ? double(v.sx) / double(nobs) : 0.0;
    s.inv_sd = s.mono ? 0.0 : 1.0 / sqrt(double(ssn) / double(nobs));
    s.na = n == 0 || double(v.n_miss) / double(n) >= sample_miss;
    s.flip = nobs && v.sx > int64_t(v.den) * int64_t(nobs);
    s.nnz = uint32_t(nobs - (s.flip ? v.c2 : v.c0));
}

// ------------------------------------------------------------------ dense panel store
struct DenseStore {
    size_t n = 0, npair = 0, pstride = 0, cap_panels = 0, nslots = 0;
    int16_t* G = nullptr;               // int16 numerators (int16 mode); anonymous mmap = zero-filled
    size_t G_bytes = 0;
    vector<float> F;                    // float columns (slot-major, n each) for > 14-bit data
    bool float_mode = false;
    vector<uint32_t> slot2snp;
    // sparse SNPs: non-zero values after orienting to the minor allele (k' = 2den - k if flipped), CSC
    vector<uint32_t> sp_snp;
    vector<uint64_t> sp_off{0};
    vector<uint32_t> sp_idx;
    vector<int16_t> sp_val;
    size_t nsparse() const { return sp_snp.size(); }
    ~DenseStore() { release(); }
    void release() { if (G) munmap(G, G_bytes); G = nullptr; }
    void init(size_t n_, size_t max_slots) {
        n = n_; npair = (n + 1) / 2; pstride = 32 * std::max<size_t>(npair, 1);
        cap_panels = (max_slots + ldk::PANEL - 1) / ldk::PANEL;
        G_bytes = std::max<size_t>(cap_panels * pstride * sizeof(int16_t), 4096);
        // anonymous mapping: zero-filled (padding needs no memset), pages are touched only when used
        void* p = mmap(nullptr, G_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) die("out of memory allocating the dosage store");
        madvise(p, G_bytes, MADV_HUGEPAGE);
        G = (int16_t*)p;
    }
    int16_t* panel(size_t p) const { return G + p * pstride; }
    size_t bytes_used() const {
        return (float_mode ? F.size() * sizeof(float) : ((nslots + 15) / 16) * pstride * 2) + sp_idx.size() * 6;
    }
    // switch to float columns (exact) when numerators do not fit int16
    void to_float() {
        F.assign(cap_panels * ldk::PANEL * n, 0.f);
        for (size_t s = 0; s < nslots; s++) {
            const int16_t* g = panel(s / 16) + (s % 16) * 2;
            float* f = &F[s * n];
            for (size_t i = 0; i < n; i++) f[i] = float(g[(i / 2) * 32 + (i & 1)]);
        }
        release(); float_mode = true;
    }
};


// ------------------------------------------------------------------ hard-call bit-plane store
// slot s: p1 = (g >= 1), p2 = (g == 2) with k = d * g; missing samples have both bits 0 and an
// observed mask o (only SNPs with missing values; the others use the shared all-ones mask)
struct BitStore {
    size_t n = 0, nw = 0, cap = 0, nslots = 0;
    uint64_t *P1 = nullptr, *P2 = nullptr, *O = nullptr, *ONES = nullptr;
    size_t bytes_each = 0;
    vector<uint32_t> slot2snp;
    vector<int32_t> omap;       // slot -> index into O, or -1
    size_t no = 0;
    static uint64_t* map_zero(size_t bytes) {
        void* p = mmap(nullptr, std::max<size_t>(bytes, 4096), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) die("out of memory allocating the bit-plane store");
        return (uint64_t*)p;
    }
    void init(size_t n_, size_t max_slots) {
        n = n_; nw = ((n + 63) / 64 + 7) / 8 * 8; cap = max_slots;
        bytes_each = std::max<size_t>(cap * nw * 8, 4096);
        P1 = map_zero(bytes_each); P2 = map_zero(bytes_each); O = map_zero(bytes_each); ONES = map_zero(nw * 8);
        for (size_t i = 0; i < n; i++) ONES[i >> 6] |= uint64_t(1) << (i & 63);
    }
    void release() {
        if (P1) { munmap(P1, bytes_each); munmap(P2, bytes_each); munmap(O, bytes_each); munmap(ONES, std::max<size_t>(nw * 8, 4096)); }
        P1 = P2 = O = ONES = nullptr;
    }
    ~BitStore() { release(); }
    const uint64_t* p1(size_t s) const { return P1 + s * nw; }
    const uint64_t* p2(size_t s) const { return P2 + s * nw; }
    const uint64_t* o(size_t s) const { return omap[s] < 0 ? ONES : O + size_t(omap[s]) * nw; }
    size_t bytes_used() const { return nslots * nw * 16 + no * nw * 8; }
    void add(size_t j, const Var& v, uint32_t d) {
        const size_t s = nslots++;
        slot2snp.push_back((uint32_t)j);
        static const ldk::PackFn pack = ldk::pick_pack();
        uint64_t* oo = v.n_miss ? O + no * nw : nullptr;
        pack(v.k.data(), v.miss.data(), n, int32_t(2 * d), P1 + s * nw, P2 + s * nw, oo);
        if (v.n_miss) omap.push_back((int32_t)no++);
        else omap.push_back(-1);
    }
    // rebuild exact numerators k = d*g (0 where missing)
    void get(size_t s, uint32_t d, vector<int32_t>& k) const {
        k.assign(n, 0);
        const uint64_t* a = p1(s); const uint64_t* b = p2(s);
        for (size_t i = 0; i < n; i++) {
            const int g = int((a[i >> 6] >> (i & 63)) & 1) + int((b[i >> 6] >> (i & 63)) & 1);
            k[i] = int32_t(d) * g;
        }
    }
};

// ------------------------------------------------------------------ decode
// Decodes all SNPs in Z order (parallel within batches) and fills st. Active SNPs (not NA, not
// monomorphic) go to the bit-plane store while every one of them is a hard call (bits mode);
// at the first one that is not, the stored ones are rebuilt exactly into the dense/sparse store.
extern double g_prof_inflate, g_prof_emit, g_prof_finish;

struct Placer {
    DenseStore& ds; vector<SnpStats>& st; size_t n; int nt; double sparse_thr;
    // place SNPs js (Z order) whose numerators are ks[t] (n values, 0 where missing)
    void place(const vector<size_t>& js, const vector<const int32_t*>& ks) {
        vector<std::pair<size_t, size_t>> pl;   // (index into js, slot)
        bool need_float = false;
        for (size_t t = 0; t < js.size(); t++) {
            const size_t j = js[t];
            if (st[j].bits > 14) need_float = true;
            if (sparse_thr > 0 && st[j].bits <= 10 && st[j].n_miss == 0 && st[j].nnz <= sparse_thr * double(n)) {
                const int32_t* k = ks[t];
                const int32_t two = int32_t(2 * st[j].den), zero = st[j].flip ? two : 0;
                for (size_t i = 0; i < n; i++)
                    if (k[i] != zero) { ds.sp_idx.push_back((uint32_t)i); ds.sp_val.push_back(int16_t(st[j].flip ? two - k[i] : k[i])); }
                ds.sp_off.push_back(ds.sp_idx.size());
                ds.sp_snp.push_back((uint32_t)j);
                continue;
            }
            pl.push_back({t, ds.nslots++});
            ds.slot2snp.push_back((uint32_t)j);
        }
        if (need_float && !ds.float_mode) ds.to_float();
        if (pl.empty()) return;
        if (ds.float_mode) {
#pragma omp parallel for num_threads(nt) schedule(dynamic, 4)
            for (size_t x = 0; x < pl.size(); x++) {
                const int32_t* k = ks[pl[x].first];
                float* f = &ds.F[pl[x].second * n];
                for (size_t i = 0; i < n; i++) f[i] = float(k[i]);
            }
            return;
        }
        // panels touched; one thread per panel (no shared cache lines)
        const size_t p0 = pl.front().second / 16, p1 = pl.back().second / 16;
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (size_t p = p0; p <= p1; p++) {
            int16_t* P = ds.panel(p);
            const int32_t* kk[16]; size_t lanes[16]; int nl = 0;
            for (auto& x : pl)
                if (x.second / 16 == p) { lanes[nl] = x.second % 16; kk[nl] = ks[x.first]; nl++; }
            // blocked 32-bit transpose: 256 sample pairs (16 KB of the panel) stay in L1
            uint32_t* W = reinterpret_cast<uint32_t*>(P);
            const size_t np = n / 2;
            for (size_t q0 = 0; q0 < np; q0 += 256) {
                const size_t q1 = std::min(np, q0 + 256);
                for (int t = 0; t < nl; t++) {
                    const int32_t* k = kk[t]; uint32_t* d = W + lanes[t];
                    for (size_t q = q0; q < q1; q++)
                        d[q * 16] = uint32_t(uint16_t(k[2 * q])) | (uint32_t(uint16_t(k[2 * q + 1])) << 16);
                }
            }
            if (n & 1) for (int t = 0; t < nl; t++) W[np * 16 + lanes[t]] = uint32_t(uint16_t(kk[t][n - 1]));
        }
    }
};

// move all sparse SNPs into dense panels (appended after the dense slots; values rebuilt exactly
// as k = a + s*k'). Slot order then differs from Z order; the finalizers use min/max indices.
static void fold_sparse(DenseStore& ds, const vector<SnpStats>& st) {
    const size_t n = ds.n;
    for (size_t u = 0; u < ds.nsparse(); u++) {
        const size_t j = ds.sp_snp[u], slot = ds.nslots++;
        ds.slot2snp.push_back((uint32_t)j);
        const SnpStats& x = st[j];
        const int64_t a = x.flip ? 2 * int64_t(x.den) : 0, sg = x.flip ? -1 : 1;
        int16_t* P = ds.panel(slot / 16) + (slot % 16) * 2;
        for (size_t i = 0; i < n; i++) P[(i / 2) * 32 + (i & 1)] = int16_t(a);
        for (uint64_t t = ds.sp_off[u]; t < ds.sp_off[u + 1]; t++) {
            const size_t i = ds.sp_idx[t];
            P[(i / 2) * 32 + (i & 1)] = int16_t(a + sg * ds.sp_val[t]);
        }
    }
    ds.sp_snp.clear(); ds.sp_off.assign(1, 0); ds.sp_idx.clear(); ds.sp_val.clear();
    ds.sp_idx.shrink_to_fit(); ds.sp_val.shrink_to_fit();
}

static void decode_all(const Source& src, size_t M, const Opts& o, vector<SnpStats>& st, DenseStore& ds, BitStore& bs,
                       bool& bits_mode, vector<vector<uint32_t>>& miss_lists, size_t max_slots = SIZE_MAX) {
    const bool blocked = max_slots != SIZE_MAX;   // --memory: dense only, store at most max_slots active SNPs
    const int nt = o.n_threads;
    const size_t n = src.n_samples();
    st.assign(M, SnpStats());
    miss_lists.assign(M, {});
    ds.init(n, blocked ? max_slots + 256 : M);
    bits_mode = !blocked && !o.force_double && env_size("FASTLD_BITS", 1) != 0;
    if (bits_mode) bs.init(n, M);
    const size_t BATCH = std::max<size_t>(64, std::min<size_t>(256, (size_t(16) << 20) / (5 * std::max<size_t>(n, 1))));
    vector<Var> vars(BATCH);
    vector<std::unique_ptr<ThreadCtx, void (*)(ThreadCtx*)>> ctx;
    for (int t = 0; t < nt; t++) ctx.push_back(src.make_ctx());
    // density threshold for the sparse path (oriented non-zeros / samples); 0 disables
    const double sparse_thr = blocked ? 0.0 : getenv("FASTLD_SPARSE") ? atof(getenv("FASTLD_SPARSE")) : 0.10;
    Placer placer{ds, st, n, nt, sparse_thr};
    const bool prof = getenv("FASTLD_PROFILE") != nullptr;
    double t_dec = 0, t_place = 0;
    vector<size_t> js; vector<const int32_t*> ks;
    for (size_t b0 = 0; b0 < M; b0 += BATCH) {
        const size_t b1 = std::min(M, b0 + BATCH);
        auto tq0 = Clock::now();
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (size_t j = b0; j < b1; j++) {
            Var& v = vars[j - b0];
            src.decode(j, v, ctx[omp_get_thread_num()].get());
            set_stats(v, n, o.sample_miss, st[j]);
        }
        t_dec += secs(tq0); tq0 = Clock::now();
        js.clear(); ks.clear();
        bool all_hard = true;
        for (size_t j = b0; j < b1; j++) {
            if (st[j].na || st[j].mono) continue;
            if (vars[j - b0].n_miss) miss_lists[j] = vars[j - b0].miss_idx;
            js.push_back(j); ks.push_back(vars[j - b0].k.data());
            all_hard &= st[j].hard_d > 0;
        }
        if (bits_mode && all_hard) {
            for (size_t j : js) bs.add(j, vars[j - b0], st[j].hard_d);
            t_place += secs(tq0);
            continue;
        }
        if (bits_mode) {   // first non-hard-call SNP: rebuild the stored ones exactly, in Z order
            bits_mode = false;
            vector<vector<int32_t>> kb(BATCH);
            for (size_t s0 = 0; s0 < bs.nslots; s0 += BATCH) {
                const size_t s1 = std::min(bs.nslots, s0 + BATCH);
                vector<size_t> cj; vector<const int32_t*> ck;
                for (size_t s = s0; s < s1; s++) {
                    const size_t j = bs.slot2snp[s];
                    bs.get(s, st[j].hard_d, kb[s - s0]);
                    cj.push_back(j); ck.push_back(kb[s - s0].data());
                }
                placer.place(cj, ck);
            }
            bs.release();
        }
        if (blocked) {   // keep only the first max_slots active SNPs
            const size_t room = ds.nslots < max_slots ? max_slots - ds.nslots : 0;
            if (js.size() > room) { js.resize(room); ks.resize(room); }
        }
        placer.place(js, ks);
        t_place += secs(tq0);
    }
    if (bits_mode) ds.release();
    // too few sparse SNPs do not pay for the extra pass over the dense store: fold them back
    if (!bits_mode && ds.nsparse() && ds.nsparse() < env_size("FASTLD_SPARSE_MIN", 64) && !ds.float_mode) fold_sparse(ds, st);
    if (prof) fprintf(stderr, "profile: decode %.3f s (read+inflate %.3f, numerators %.3f, sums %.3f; thread-summed), placement %.3f s\n",
                      t_dec, g_prof_inflate, g_prof_emit, g_prof_finish, t_place);
}

// exact r with missing data (WP5 definition; all quantities in numerator units):
//   num = n_i n_j C - n_i sx_j A_ij - n_j sx_i A_ji + sx_i sx_j n_ij,  V = n sxx - sx^2
//   r = num / sqrt(n_i n_j V_i V_j)
static inline double r_missing(__int128 C, __int128 Aij, __int128 Aji, __int128 nij, const SnpStats& a, const SnpStats& b, size_t n) {
    const __int128 ni = (__int128)(n - a.n_miss), nj = (__int128)(n - b.n_miss);
    const __int128 num = ni * nj * C - ni * b.sx * Aij - nj * a.sx * Aji + (__int128)a.sx * b.sx * nij;
    const __int128 Vi = ni * a.sxx - (__int128)a.sx * a.sx, Vj = nj * b.sxx - (__int128)b.sx * b.sx;
    return double(num) / sqrt(double(ni) * double(nj) * double(Vi) * double(Vj));
}

// ------------------------------------------------------------------ hard-call path
static std::unique_ptr<LDMatrix> ld_bits(const BitStore& bs, size_t M, const vector<SnpStats>& st, int nt) {
    auto T = std::make_unique<PackedTri>(M);
    const size_t ns = bs.nslots, n = bs.n;
    const char* kname; ldk::BitsFn kern = ldk::pick_bits(&kname);
    const bool miss = bs.no > 0;
    info("Correlation path", string("exact hard-call bit planes (") + kname + "), " + std::to_string(ns) + " SNPs" +
                             (miss ? ", missing data exact" : ""));
    if (ns < 2) return T;
    const size_t WC = std::max<size_t>(8, env_size("FASTLD_WC", 512) / 8 * 8);   // words per chunk
    const size_t JBdef = std::max<size_t>(32, std::min<size_t>(64, (ns / (3 * size_t(nt)) + 7) / 8 * 8));
    const size_t JB = std::max<size_t>(8, env_size("FASTLD_BJB", JBdef));
    const size_t nblk = (ns + JB - 1) / JB;
    const __int128 nn = (__int128)n;
    vector<double> dV(ns);
    for (size_t s = 0; s < ns; s++) { const SnpStats& a = st[bs.slot2snp[s]]; dV[s] = double(nn * a.sxx - (__int128)a.sx * a.sx); }
    // register-blocked 4x2 VPOPCNTDQ tile for regions without missing data
    const bool tile42 = !miss && kern == (ldk::BitsFn)ldk::bits_vpopcnt;
    const size_t WC2 = std::max<size_t>(8, env_size("FASTLD_WC2", 512) / 8 * 8);
    vector<uint64_t> zero_plane(bs.nw + 8, 0);
    double t_fin_b = 0;
    const uint64_t* Z = zero_plane.data() + ((64 - (uintptr_t(zero_plane.data()) & 63)) & 63) / 8;
#pragma omp parallel num_threads(nt) reduction(+ : t_fin_b)
    {
        vector<int64_t> Lb;
#pragma omp for schedule(dynamic, 1)
        for (size_t bb = 0; bb < nblk; bb++) {
            const size_t blk = nblk - 1 - bb;
            const size_t J0 = blk * JB, J1 = std::min(ns, J0 + JB);
            if (tile42) {
                const size_t rows = J1 - J0 + 2, ldo = (J1 + 3) / 4 * 4;   // j may start at an odd offset: tile row j+1 <= J1
                Lb.assign(rows * ldo, 0);
                for (size_t w0 = 0; w0 < bs.nw; w0 += WC2) {
                    const size_t nwc = std::min(WC2, bs.nw - w0);
                    for (size_t i0 = 0; i0 + 1 < J1; i0 += 4) {
                        const uint64_t *p1i[4], *p2i[4];
                        for (int t = 0; t < 4; t++) {
                            const bool ok = i0 + t < J1;
                            p1i[t] = ok ? bs.p1(i0 + t) + w0 : Z; p2i[t] = ok ? bs.p2(i0 + t) + w0 : Z;
                        }
                        for (size_t j = std::max(J0, i0 + 1); j < J1; j += 2) {
                            const uint64_t *p1j[2], *p2j[2];
                            for (int t = 0; t < 2; t++) {
                                const bool ok = j + t < J1;
                                p1j[t] = ok ? bs.p1(j + t) + w0 : Z; p2j[t] = ok ? bs.p2(j + t) + w0 : Z;
                            }
                            ldk::bits_tile42_vpopcnt(p1i, p2i, p1j, p2j, nwc, &Lb[(j - J0) * ldo + i0], ldo);
                        }
                    }
                }
                for (size_t j = J0; j < J1; j++) {
                    const size_t sj = bs.slot2snp[j];
                    const SnpStats& b = st[sj];
                    for (size_t i = 0; i < j; i++) {
                        const size_t si = bs.slot2snp[i];
                        const SnpStats& a = st[si];
                        const __int128 C = (__int128)a.hard_d * b.hard_d * Lb[(j - J0) * ldo + i];
                        T->at(si, sj) = double(nn * C - (__int128)a.sx * b.sx) / sqrt(dV[i] * dV[j]);   // baseline formula
                    }
                }
                continue;
            }
            if (miss && kern == (ldk::BitsFn)ldk::bits_vpopcnt) {   // register-blocked 2x2 tile with missing data
                const size_t rows = J1 - J0 + 2, ldo = (J1 + 1) / 2 * 2;   // j may start at an odd offset: tile row j+1 <= J1
                Lb.assign(rows * ldo * 4, 0);
                for (size_t w0 = 0; w0 < bs.nw; w0 += WC2) {
                    const size_t nwc = std::min(WC2, bs.nw - w0);
                    for (size_t i0 = 0; i0 + 1 < J1; i0 += 2) {
                        const uint64_t *p1i[2], *p2i[2], *oi[2];
                        for (int t = 0; t < 2; t++) {
                            const bool ok = i0 + t < J1;
                            p1i[t] = ok ? bs.p1(i0 + t) + w0 : Z; p2i[t] = ok ? bs.p2(i0 + t) + w0 : Z; oi[t] = ok ? bs.o(i0 + t) + w0 : Z;
                        }
                        for (size_t j = std::max(J0, i0 + 1); j < J1; j += 2) {
                            const uint64_t *p1j[2], *p2j[2], *oj[2];
                            for (int t = 0; t < 2; t++) {
                                const bool ok = j + t < J1;
                                p1j[t] = ok ? bs.p1(j + t) + w0 : Z; p2j[t] = ok ? bs.p2(j + t) + w0 : Z; oj[t] = ok ? bs.o(j + t) + w0 : Z;
                            }
                            ldk::bits_tile22m_vpopcnt(p1i, p2i, oi, p1j, p2j, oj, nwc, &Lb[((j - J0) * ldo + i0) * 4], ldo);
                        }
                    }
                }
                double tf0 = omp_get_wtime();
                for (size_t j = J0; j < J1; j++) {
                    const size_t sj = bs.slot2snp[j];
                    const SnpStats& b = st[sj];
                    const __int128 dj = b.hard_d;
                    for (size_t i = 0; i < j; i++) {
                        const size_t si = bs.slot2snp[i];
                        const SnpStats& a = st[si];
                        const int64_t* L = &Lb[((j - J0) * ldo + i) * 4];
                        const __int128 di = a.hard_d, C = di * dj * L[0];
                        T->at(si, sj) = (!a.n_miss && !b.n_miss) ? double(nn * C - (__int128)a.sx * b.sx) / sqrt(dV[i] * dV[j])
                                                                 : r_missing(C, di * L[1], dj * L[2], L[3], a, b, n);
                    }
                }
                t_fin_b += omp_get_wtime() - tf0;
                continue;
            }
            Lb.assign((J1 - J0) * J1 * 4, 0);
            for (size_t w0 = 0; w0 < bs.nw; w0 += WC) {
                const size_t nwc = std::min(WC, bs.nw - w0);
                for (size_t i0 = 0; i0 + 1 < J1; i0 += 8) {
                    ldk::BitArgs g;
                    g.NI = (int)std::min<size_t>(8, J1 - i0); g.nw = nwc;
                    for (int t = 0; t < g.NI; t++) { g.p1i[t] = bs.p1(i0 + t) + w0; g.p2i[t] = bs.p2(i0 + t) + w0; g.oi[t] = bs.o(i0 + t) + w0; }
                    for (size_t j = std::max(J0, i0 + 1); j < J1; j++) {
                        g.p1j = bs.p1(j) + w0; g.p2j = bs.p2(j) + w0; g.oj = bs.o(j) + w0;
                        g.NI = (int)std::min<size_t>(std::min<size_t>(8, J1 - i0), j - i0);   // only i < j
                        g.out = &Lb[((j - J0) * J1 + i0) * 4];
                        kern(g, miss);
                    }
                }
            }
            for (size_t j = J0; j < J1; j++) {
                const size_t sj = bs.slot2snp[j];
                const SnpStats& b = st[sj];
                const __int128 dj = b.hard_d;
                for (size_t i = 0; i < j; i++) {
                    const size_t si = bs.slot2snp[i];
                    const SnpStats& a = st[si];
                    const int64_t* L = &Lb[((j - J0) * J1 + i) * 4];
                    const __int128 di = a.hard_d;
                    const __int128 C = di * dj * L[0];
                    double r;
                    if (!a.n_miss && !b.n_miss) r = double(nn * C - (__int128)a.sx * b.sx) / sqrt(dV[i] * dV[j]);   // baseline formula
                    else r = r_missing(C, di * L[1], dj * L[2], L[3], a, b, n);
                    T->at(si, sj) = r;
                }
            }
        }
    }
    if (getenv("FASTLD_PROFILE")) fprintf(stderr, "profile: bit-plane finalize (r) %.3f s\n", t_fin_b);
    return T;
}

// oriented representation of a sparse SNP: k = a + s * k'  (a = 2den, s = -1 if flipped)
struct Orient { int64_t a, s, sxp; };
static Orient orient(const SnpStats& x, size_t n) {
    if (!x.flip) return {0, 1, x.sx};
    const int64_t two = 2 * int64_t(x.den);
    return {two, -1, two * int64_t(n) - x.sx};
}
static inline double r_from_C(__int128 nn, __int128 C, const SnpStats& a, const SnpStats& b, double dVa, double dVb) {
    const __int128 num = nn * C - (__int128)a.sx * b.sx;
    return double(num) / sqrt(dVa * dVb);
}

// dense x sparse pairs: D_ui = sum_s k_i[s] * k'_u[s] over the non-zeros of sparse u
static void ld_dense_sparse(const DenseStore& ds, const vector<SnpStats>& st, PackedTri& T, int nt, uint64_t maxk,
                            const vector<double>& dVd, const vector<double>& dVs) {
    const size_t Md = ds.nslots, Ms = ds.nsparse(), n = ds.n;
    if (!Md || !Ms) return;
    const size_t np = (Md + 15) / 16, W = np * 16;
    ldk::SparseFn kern = ldk::pick_sparse();
    const size_t F = std::max<size_t>(2, size_t(INT32_MAX) / (maxk * maxk)) / 2 * 2;   // samples between int32 flushes
    const size_t CS = std::min(F, 2 * env_size("FASTLD_SQC", 1024));            // samples per chunk
    const size_t PG = std::max<size_t>(1, env_size("FASTLD_SPG", 24));           // panels per group
    const size_t UB = std::max<size_t>(8, std::min<size_t>(Ms, (size_t(48) << 20) / (W * 12)));
    const size_t nub = (Ms + UB - 1) / UB;
    const __int128 nn = (__int128)n;
#pragma omp parallel num_threads(nt)
    {
        vector<int32_t> D32; vector<int64_t> D64; vector<uint64_t> cur, endp;
#pragma omp for schedule(dynamic, 1)
        for (size_t ub = 0; ub < nub; ub++) {
            const size_t u0 = ub * UB, u1 = std::min(Ms, u0 + UB), nu = u1 - u0;
            D32.assign(nu * W, 0); D64.assign(nu * W, 0);
            cur.assign(ds.sp_off.begin() + u0, ds.sp_off.begin() + u1); endp.resize(nu);
            size_t since = 0;
            for (size_t cs = 0; cs < n; cs += CS) {
                const size_t ce = std::min(n, cs + CS);
                if (since + (ce - cs) > F) {   // flush int32 partials before they could overflow
                    for (size_t x = 0; x < nu * W; x++) { D64[x] += D32[x]; D32[x] = 0; }
                    since = 0;
                }
                for (size_t u = 0; u < nu; u++) {
                    uint64_t e = cur[u], lim = ds.sp_off[u0 + u + 1];
                    while (e < lim && ds.sp_idx[e] < ce) e++;
                    endp[u] = e;
                }
                for (size_t pg = 0; pg < np; pg += PG) {
                    const int npg = (int)std::min(PG, np - pg);
                    const int16_t* G = ds.panel(pg);
                    for (size_t u = 0; u < nu; u++) {
                        const size_t cnt = endp[u] - cur[u];
                        if (cnt) kern(npg, G, ds.pstride, &ds.sp_idx[cur[u]], &ds.sp_val[cur[u]], cnt, &D32[u * W + pg * 16]);
                    }
                }
                cur = endp;
                since += ce - cs;
            }
            for (size_t x = 0; x < nu * W; x++) D64[x] += D32[x];
            for (size_t u = 0; u < nu; u++) {
                const size_t su = ds.sp_snp[u0 + u];
                const SnpStats& b = st[su];
                const Orient ob = orient(b, n);
                for (size_t i = 0; i < Md; i++) {
                    const size_t si = ds.slot2snp[i];
                    const SnpStats& a = st[si];
                    const __int128 C = (__int128)ob.a * a.sx + (__int128)ob.s * D64[u * W + i];   // sum k_i k_u, original orientation
                    const double r = r_from_C(nn, C, a, b, dVd[i], dVs[u0 + u]);
                    if (si < su) T.at(si, su) = r; else T.at(su, si) = r;
                }
            }
        }
    }
}

// sparse x sparse pairs: row-wise outer products over a CSR copy, blocked over j
static void ld_sparse_sparse(const DenseStore& ds, const vector<SnpStats>& st, PackedTri& T, int nt, const vector<double>& dVs) {
    const size_t Ms = ds.nsparse(), n = ds.n;
    if (Ms < 2) return;
    const size_t nnz = ds.sp_idx.size();
    vector<uint64_t> rp(n + 1, 0);
    for (size_t t = 0; t < nnz; t++) rp[ds.sp_idx[t] + 1]++;
    for (size_t s = 0; s < n; s++) rp[s + 1] += rp[s];
    vector<uint32_t> cu(nnz); vector<int16_t> cv(nnz);
    {
        vector<uint64_t> pos(rp.begin(), rp.end() - 1);
        for (size_t u = 0; u < Ms; u++)
            for (uint64_t t = ds.sp_off[u]; t < ds.sp_off[u + 1]; t++) { uint64_t p = pos[ds.sp_idx[t]]++; cu[p] = (uint32_t)u; cv[p] = ds.sp_val[t]; }
    }
    vector<Orient> O(Ms);
    for (size_t u = 0; u < Ms; u++) O[u] = orient(st[ds.sp_snp[u]], n);
    const size_t JB = std::max<size_t>(16, std::min<size_t>(env_size("FASTLD_SJB", 512), (size_t(32) << 20) / (8 * Ms)));
    const size_t njb = (Ms + JB - 1) / JB;
    const __int128 nn = (__int128)n;
#pragma omp parallel num_threads(nt)
    {
        vector<int64_t> Cs;
#pragma omp for schedule(dynamic, 1)
        for (size_t bb = 0; bb < njb; bb++) {
            const size_t jb = njb - 1 - bb;
            const size_t j0 = jb * JB, j1 = std::min(Ms, j0 + JB), ld = j1;
            Cs.assign((j1 - j0) * ld, 0);
            for (size_t s = 0; s < n; s++) {
                const uint64_t b = rp[s], e = rp[s + 1];
                for (uint64_t x = b; x < e; x++) {
                    const uint32_t j = cu[x];
                    if (j < j0) continue;
                    if (j >= j1) break;
                    const int64_t vj = cv[x];
                    int64_t* c = &Cs[(j - j0) * ld];
                    for (uint64_t y = b; y < x; y++) c[cu[y]] += vj * cv[y];
                }
            }
            for (size_t j = j0; j < j1; j++) {
                const size_t sj = ds.sp_snp[j];
                const Orient& oj = O[j];
                for (size_t i = 0; i < j; i++) {
                    const size_t si = ds.sp_snp[i];
                    const Orient& oi = O[i];
                    const __int128 C = nn * oi.a * oj.a + (__int128)oi.a * oj.s * oj.sxp + (__int128)oj.a * oi.s * oi.sxp
                                     + (__int128)(oi.s * oj.s) * Cs[(j - j0) * ld + i];
                    T.at(si, sj) = r_from_C(nn, C, st[si], st[sj], dVs[i], dVs[j]);
                }
            }
        }
    }
}

// ------------------------------------------------------------------ exact integer path
// Exact integer LD for the slots of ds. Normal mode: T != nullptr, all pairs (plus sparse pairs).
// Blocked (--memory) mode: F != nullptr, only pairs with slot j >= jstart and slot i < ilimit, each
// column segment written to the file-backed triangle.
static void ld_integer_core(const DenseStore& ds, size_t M, const vector<SnpStats>& st, int nt,
                            const vector<vector<uint32_t>>& miss_lists, PackedTri* T, FileTri* F,
                            size_t jstart, size_t ilimit) {
    const size_t ns = ds.nslots, n = ds.n;
    const char* kname; ldk::DenseFn kern = ldk::pick_dense(&kname);
    if (T) info("Correlation path", string("exact integer (") + kname + "), " + std::to_string(ns) + " dense + " +
                                    std::to_string(ds.nsparse()) + " sparse SNPs");
    ilimit = std::min(ilimit, ns);
    uint64_t maxk = 1;
    for (size_t s = 0; s < ns; s++) maxk = std::max<uint64_t>(maxk, 2 * st[ds.slot2snp[s]].den);
    for (size_t u = 0; u < ds.nsparse(); u++) maxk = std::max<uint64_t>(maxk, 2 * st[ds.sp_snp[u]].den);
    vector<double> dVs(ds.nsparse());
    for (size_t u = 0; u < ds.nsparse(); u++) {
        const SnpStats& a = st[ds.sp_snp[u]];
        dVs[u] = double((__int128)n * a.sxx - (__int128)a.sx * a.sx);
    }
    const size_t maxq = std::max<size_t>(1, (size_t(INT32_MAX) / (maxk * maxk)) / 2);
    // missing data (exact, WP5): per-slot missing lists, per-sample lists of slots missing there
    bool has_missing = false;
    for (size_t s = 0; s < ns; s++) has_missing |= st[ds.slot2snp[s]].n_miss > 0;
    vector<const vector<uint32_t>*> ML(ns, nullptr);
    vector<uint64_t> mrp; vector<uint32_t> mcol; vector<int16_t> ones;
    ldk::SparseFn skern = ldk::pick_sparse();
    if (has_missing) {
        size_t maxm = 0;
        for (size_t s = 0; s < ns; s++) { ML[s] = &miss_lists[ds.slot2snp[s]]; maxm = std::max(maxm, ML[s]->size()); }
        ones.assign(std::max<size_t>(maxm, 1), 1);
        mrp.assign(n + 1, 0);
        for (size_t s = 0; s < ns; s++) for (uint32_t i : *ML[s]) mrp[i + 1]++;
        for (size_t i = 0; i < n; i++) mrp[i + 1] += mrp[i];
        mcol.resize(mrp[n]);
        vector<uint64_t> pos(mrp.begin(), mrp.end() - 1);
        for (size_t s = 0; s < ns; s++) for (uint32_t i : *ML[s]) mcol[pos[i]++] = (uint32_t)s;   // slots ascending per sample
    }
    const size_t QC = std::min(env_size("FASTLD_QC", 256), maxq);
    // j-block width: large for single-thread A reuse, but ~3 blocks per thread for load balance
    const size_t JBdef = std::max<size_t>(64, std::min<size_t>(512, (ns / (3 * size_t(nt)) + 15) / 16 * 16));
    const size_t JB = std::max<size_t>(16, env_size("FASTLD_JB", has_missing ? std::min<size_t>(JBdef, 256) : JBdef) / 16 * 16);
    const size_t IB = std::max<size_t>(3, env_size("FASTLD_IB", 48) / 3 * 3);   // i-panels per L2 block
    const int ORDER = (int)env_size("FASTLD_ORDER", 1);
    const size_t js0 = jstart / 16 * 16;
    const size_t nblk = ns > js0 ? (ns - js0 + JB - 1) / JB : 0;
    const __int128 nn = (__int128)n;
    vector<double> dV(ns);
    for (size_t s = 0; s < ns; s++) {
        const SnpStats& a = st[ds.slot2snp[s]];
        dV[s] = double(nn * a.sxx - (__int128)a.sx * a.sx);
    }
    const bool prof = getenv("FASTLD_PROFILE") != nullptr;
    // sparse pairs first: afterwards the dense blocks are the last users of the dense panels,
    // which are then released block by block (highest j first) while the triangle fills
    if (T) {
        auto t1 = Clock::now();
        ld_dense_sparse(ds, st, *T, nt, maxk, dV, dVs);
        auto t2 = Clock::now();
        ld_sparse_sparse(ds, st, *T, nt, dVs);
        if (prof) fprintf(stderr, "profile: dense x sparse %.3f s, sparse x sparse %.3f s\n",
                          std::chrono::duration<double>(t2 - t1).count(), secs(t2));
    }
    if (prof && T) fprintf(stderr, "profile: before dense blocks: triangle resident %zu MB, rss %ld MB\n", resident_mb(T->T, T->bytes), rss_mb());
    // panel p is read as an i-panel only by blocks b >= floor(16p/JB) (and as j-panel by its own
    // block): once all blocks >= f are done, panels >= f*JB/16 can be dropped (MADV_DONTNEED)
    const bool release = T && jstart == 0 && env_size("FASTLD_NOFREE", 0) == 0;
    vector<uint8_t> done(nblk, 0);
    size_t frontier = nblk;
    const size_t page = 4096, total_bytes = ((ns + 15) / 16) * ds.pstride * sizeof(int16_t);
    size_t freed_from = (total_bytes + page - 1) / page * page;
    auto release_done = [&](size_t blk) {
        if (!release) return;
#pragma omp critical(ldk_release)
        {
            done[blk] = 1;
            while (frontier > 0 && done[frontier - 1]) frontier--;
            const size_t p0 = frontier * JB / 16;
            size_t off = (p0 * ds.pstride * sizeof(int16_t) + page - 1) / page * page;
            if (off < freed_from) {
                madvise((char*)ds.G + off, freed_from - off, MADV_DONTNEED);
                freed_from = off;
            }
        }
    };
    double t_kern = 0, t_fin = 0, t_zero = 0;
#pragma omp parallel num_threads(nt) reduction(+ : t_kern, t_fin, t_zero)
    {
        vector<int64_t> Lb;
        vector<int32_t> X1, X2, NB;
        struct Def { size_t x, y; double r; };
        vector<Def> deferred;
        vector<double> colbuf;
#pragma omp for schedule(dynamic, 1)
        for (size_t bb = 0; bb < nblk; bb++) {
            double tp0 = prof ? omp_get_wtime() : 0;
            const size_t blk = nblk - 1 - bb;                 // largest blocks first
            const size_t J0 = js0 + blk * JB, J1 = std::min(ns, J0 + JB);
            const size_t J1r = (J1 + 7) / 8 * 8;              // j-subs of 8 (padding lanes are zero)
            const size_t nI = (J1 + 15) / 16, ldL = nI * 16;
            const size_t IL = std::min(J1, ilimit), nIi = (IL + 15) / 16;   // i-side: slots < IL
            if (prof) fprintf(stderr, "profile: block %zu start rss %ld MB\n", blk, rss_mb());
            Lb.assign((J1r - J0) * ldL, 0);
            if (prof) fprintf(stderr, "profile: block %zu Lb (%zu MB) rss %ld MB\n", blk, Lb.size() * 8 >> 20, rss_mb());
            double tp1 = prof ? omp_get_wtime() : 0;
            if (ORDER == 1) {
                // BLIS-like order: sample chunk (QC pairs) -> i-block (IB panels, in L2) -> j-sub of 8
                // SNPs (QC lines, in L1) -> i-groups of 3 panels -> kernel
                for (size_t q0 = 0; q0 < ds.npair; q0 += QC) {
                    const size_t nq = std::min(QC, ds.npair - q0);
                    for (size_t ib = 0; ib < nIi; ib += IB) {
                        const size_t ie = std::min(nIi, ib + IB);
                        for (size_t js = J0; js < J1r; js += 8) {
                            const int16_t* B = ds.panel(js / 16) + q0 * 32 + (js % 16) * 2;
                            for (size_t ig = ib; ig < ie; ig += 3) {
                                if (ig * 16 >= js + 8) break;     // this and later groups are all above j
                                const int IP = (int)std::min<size_t>(3, ie - ig);
                                kern(IP, ds.panel(ig) + q0 * 32, ds.pstride, B, nq, &Lb[(js - J0) * ldL + ig * 16], ldL);
                            }
                        }
                    }
                }
            } else {
                // i-group outer: sample chunk -> i-group of 3 panels (3*QC lines, L1 for small QC)
                // -> every j-sub of the block (from L2)
                for (size_t q0 = 0; q0 < ds.npair; q0 += QC) {
                    const size_t nq = std::min(QC, ds.npair - q0);
                    for (size_t ig = 0; ig < nIi; ig += 3) {
                        const int IP = (int)std::min<size_t>(3, nIi - ig);
                        const int16_t* A = ds.panel(ig) + q0 * 32;
                        for (size_t js = std::max(J0, ig * 16 / 8 * 8); js < J1r; js += 8) {
                            if (ig * 16 >= js + 8) continue;
                            kern(IP, A, ds.pstride, ds.panel(js / 16) + q0 * 32 + (js % 16) * 2, nq, &Lb[(js - J0) * ldL + ig * 16], ldL);
                        }
                    }
                }
            }
            double tp2 = prof ? omp_get_wtime() : 0;
            if (prof) fprintf(stderr, "profile: block %zu kernel done rss %ld MB\n", blk, rss_mb());
            // missing-data terms for this block:
            //   X1[j][i] = sum over s in M_j of k_i[s]   (A_ij = sx_i - X1)
            //   X2[i][j] = sum over s in M_i of k_j[s]   (A_ji = sx_j - X2)
            //   NB[j][i] = |M_i intersect M_j|
            const size_t pb0 = J0 / 16, npb = (J1 + 15) / 16 - pb0, ld2 = npb * 16;
            if (has_missing) {
                X1.assign((J1 - J0) * ldL, 0); X2.assign(J1 * ld2, 0); NB.assign((J1 - J0) * J1, 0);
                for (size_t j = std::max(J0, jstart); j < J1; j++)
                    if (!ML[j]->empty()) skern((int)nIi, ds.panel(0), ds.pstride, ML[j]->data(), ones.data(), ML[j]->size(), &X1[(j - J0) * ldL]);
                for (size_t i = 0; i < IL; i++)
                    if (!ML[i]->empty()) skern((int)npb, ds.panel(pb0), ds.pstride, ML[i]->data(), ones.data(), ML[i]->size(), &X2[i * ld2]);
                for (size_t smp = 0; smp < n; smp++) {
                    const uint64_t b0 = mrp[smp], b1 = mrp[smp + 1];
                    for (uint64_t x = b0; x < b1; x++) {
                        const uint32_t j = mcol[x];
                        if (j < J0) continue;
                        if (j >= J1) break;
                        int32_t* nb = &NB[(j - J0) * J1];
                        for (uint64_t y = b0; y < x; y++) nb[mcol[y]]++;
                    }
                }
            }
            // r for slot pairs i < j, j in this block
            auto r_pair = [&](size_t i, size_t j, const SnpStats& a, const SnpStats& b) {
                if (!a.n_miss && !b.n_miss) {
                    const __int128 num = nn * Lb[(j - J0) * ldL + i] - (__int128)a.sx * b.sx;
                    return double(num) / sqrt(dV[i] * dV[j]);                        // baseline formula
                }
                const __int128 Aij = a.sx - X1[(j - J0) * ldL + i], Aji = b.sx - X2[i * ld2 + (j - pb0 * 16)];
                const __int128 nij = (__int128)n - a.n_miss - b.n_miss + NB[(j - J0) * J1 + i];
                return r_missing(Lb[(j - J0) * ldL + i], Aij, Aji, nij, a, b, n);
            };
            if (F) {   // blocked mode: one column segment per j, Z positions slot2snp[0] .. slot2snp[i_last]
                const size_t z0 = ds.slot2snp[0];
                for (size_t j = std::max(J0, jstart); j < J1; j++) {
                    const size_t imax = std::min(j, ilimit);
                    if (!imax) continue;
                    const size_t sj = ds.slot2snp[j], cnt = ds.slot2snp[imax - 1] - z0 + 1;
                    colbuf.assign(cnt, 0.0);
                    for (size_t i = 0; i < imax; i++) colbuf[ds.slot2snp[i] - z0] = r_pair(i, j, st[ds.slot2snp[i]], st[sj]);
                    F->put(sj, z0, colbuf.data(), cnt);
                }
            } else {
                for (size_t i = 0; i + 1 < J1; i++) {
                    const size_t si = ds.slot2snp[i];
                    const SnpStats& a = st[si];
                    for (size_t j = std::max(J0, i + 1); j < J1; j++) {
                        const size_t sj = ds.slot2snp[j];
                        const double r = r_pair(i, j, a, st[sj]);
                        // slots are in Z order except for folded-back SNPs (appended): pairs whose column
                        // is not j's are deferred so the triangle fills strictly block by block
                        if (si < sj) T->at(si, sj) = r; else deferred.push_back({sj, si, r});
                    }
                }
            }
            if (prof) { double tp3 = omp_get_wtime(); t_zero += tp1 - tp0; t_kern += tp2 - tp1; t_fin += tp3 - tp2; }
            if (prof && T) {
                size_t pg = (T->bytes + 4095) / 4096; vector<unsigned char> mv(pg);
                mincore(T->T, T->bytes, mv.data());
                size_t res = 0, first = pg, last = 0;
                for (size_t q = 0; q < pg; q++) if (mv[q] & 1) { res++; first = std::min(first, q); last = q; }
                fprintf(stderr, "profile: block %zu done, rss %ld MB; triangle resident %zu MB (pages %zu..%zu of %zu)\n", blk, rss_mb(), res * 4096 >> 20, first, last, pg);
            }
            release_done(blk);
            if (prof) fprintf(stderr, "profile: after release rss %ld MB\n", rss_mb());
        }
#pragma omp barrier
        for (const Def& d : deferred) T->at(d.x, d.y) = d.r;   // all blocks done; distinct pairs per thread (normal mode only)
    }
    if (prof) fprintf(stderr, "profile: buffer setup %.3f s, multiply loop %.3f s, finalize (r) %.3f s  [QC=%zu JB=%zu IB=%zu]\n", t_zero, t_kern, t_fin, QC, JB, IB);
}

static std::unique_ptr<LDMatrix> ld_integer(const DenseStore& ds, size_t M, const vector<SnpStats>& st, int nt,
                                            const vector<vector<uint32_t>>& miss_lists) {
    auto T = std::make_unique<PackedTri>(M);
    ld_integer_core(ds, M, st, nt, miss_lists, T.get(), nullptr, 0, SIZE_MAX);
    return T;
}

// ------------------------------------------------------------------ floating-point path (BLAS dsyrk)
static std::unique_ptr<LDMatrix> ld_double(const DenseStore& ds, size_t M, const vector<SnpStats>& st,
                                           const vector<vector<uint32_t>>& miss_lists, const Opts& o) {
    const int nt = o.n_threads;
    const size_t n = ds.n;
    info("Correlation path", "floating point (BLAS dsyrk)");
    auto Rm = std::make_unique<FullMatrix>(M);
    vector<double>& R = Rm->R;
    vector<int32_t> snp2slot(M, -1), snp2sp(M, -1);
    for (size_t s = 0; s < ds.nslots; s++) snp2slot[ds.slot2snp[s]] = (int32_t)s;
    for (size_t u = 0; u < ds.nsparse(); u++) snp2sp[ds.sp_snp[u]] = (int32_t)u;
    vector<uint64_t> spcur(ds.sp_off.begin(), ds.sp_off.end() - (ds.sp_off.empty() ? 0 : 1));
    size_t S = o.chunk ? o.chunk : std::max<size_t>(256, std::min<size_t>(4096, (size_t(256) << 20) / (8 * std::max<size_t>(M, 1))));
    S = std::min(S + (S & 1), std::max<size_t>(n + (n & 1), 2));   // even, so chunks start on a sample pair
    vector<double> Z(S * M);
    vector<size_t> mpos(M, 0);   // cursor into each missing list
    for (size_t a = 0; a < n; a += S) {
        const size_t s = std::min(S, n - a);
#pragma omp parallel for num_threads(nt) schedule(static)
        for (size_t j = 0; j < M; j++) {
            double* z = &Z[j * s];
            const int32_t slot = snp2slot[j];
            const double mu = st[j].mean, is = st[j].inv_sd;
            if (snp2sp[j] >= 0) {   // sparse SNP: k = a + s*k' (no missing values by construction)
                const int32_t u = snp2sp[j];
                const Orient ou = orient(st[j], n);
                const double base = (double(ou.a) - mu) * is;
                for (size_t i = 0; i < s; i++) z[i] = base;
                uint64_t& c = spcur[u];
                const uint64_t lim = ds.sp_off[u + 1];
                while (c < lim && ds.sp_idx[c] < a + s) { z[ds.sp_idx[c] - a] = (double(ou.a + ou.s * ds.sp_val[c]) - mu) * is; c++; }
                continue;
            }
            if (slot < 0) { memset(z, 0, s * sizeof(double)); continue; }
            if (ds.float_mode) {
                const float* g = &ds.F[size_t(slot) * n + a];
                for (size_t i = 0; i < s; i++) z[i] = (double(g[i]) - mu) * is;
            } else {
                const int16_t* g = ds.panel(slot / 16) + (slot % 16) * 2 + (a / 2) * 32;
                for (size_t i = 0; i < s; i++) z[i] = (double(g[(i / 2) * 32 + (i & 1)]) - mu) * is;
            }
            const auto& ml = miss_lists[j];
            size_t& c = mpos[j];
            while (c < ml.size() && ml[c] < a + s) { z[ml[c] - a] = 0.0; c++; }   // mean imputation
        }
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, (int)M, (int)s, 1.0, Z.data(), (int)s, 1.0, R.data(), (int)M);
    }
#pragma omp parallel for num_threads(nt) schedule(dynamic, 16)
    for (size_t j = 0; j < M; j++) for (size_t i = j + 1; i < M; i++) R[j * M + i] = R[i * M + j];
    return Rm;
}

// ------------------------------------------------------------------ memory model and --memory
struct MemModel { double base, dense, tri, batch, lb, writer; };
static MemModel mem_model(size_t n, size_t M, int nt) {
    MemModel m;
    m.base = 12e6;
    m.dense = 4.0 * double((n + 1) / 2) * double((M + 15) / 16 * 16);
    m.tri = 8.0 * double(M) * double(M > 1 ? M - 1 : 0) / 2;
    m.batch = double(std::max<size_t>(64, std::min<size_t>(256, (size_t(16) << 20) / (5 * std::max<size_t>(n, 1))))) * 5.0 * double(n);
    const size_t JB = std::max<size_t>(64, std::min<size_t>(512, (M / (3 * size_t(nt)) + 15) / 16 * 16));
    m.lb = double(nt) * double(JB) * double(M) * 8.0;
    const double rows = double(std::max<size_t>(16, std::min<size_t>(256, (size_t(32) << 20) / (8 * std::max<size_t>(M, 1)))));
    m.writer = rows * double(M) * (8.0 + 16.0);
    return m;
}
// expected peak of the normal path (dense store released while the triangle fills)
static double peak_normal(const MemModel& m, size_t M) {
    const double decode = m.dense + m.batch;
    const double compute = std::max(m.dense, m.tri) + m.lb + std::min(m.tri, 512.0 * 8.0 * double(M));
    const double write = m.tri + m.writer;
    return m.base + std::max(decode, std::max(compute, write));
}

// decode the listed SNPs (Z indices) and append them to the dense store
static void decode_list(const Source& src, const vector<uint32_t>& list, size_t a, size_t b, DenseStore& ds,
                        vector<SnpStats>& st, int nt) {
    const size_t n = src.n_samples();
    const size_t BATCH = std::max<size_t>(64, std::min<size_t>(256, (size_t(16) << 20) / (5 * std::max<size_t>(n, 1))));
    vector<Var> vars(BATCH);
    vector<std::unique_ptr<ThreadCtx, void (*)(ThreadCtx*)>> ctx;
    for (int t = 0; t < nt; t++) ctx.push_back(src.make_ctx());
    Placer placer{ds, st, n, nt, 0.0};
    vector<size_t> js; vector<const int32_t*> ks;
    for (size_t b0 = a; b0 < b; b0 += BATCH) {
        const size_t b1 = std::min(b, b0 + BATCH);
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (size_t t = b0; t < b1; t++) src.decode(list[t], vars[t - b0], ctx[omp_get_thread_num()].get());
        js.clear(); ks.clear();
        for (size_t t = b0; t < b1; t++) { js.push_back(list[t]); ks.push_back(vars[t - b0].k.data()); }
        placer.place(js, ks);
    }
}

// --memory: dense integer path over SNP blocks; the LD matrix is kept in a temporary file
static std::unique_ptr<LDMatrix> compute_ld_blocked(const Source& src, size_t M, const Opts& o, vector<SnpStats>& st,
                                                    Timing& tm, double limit, const MemModel& mm) {
    const int nt = o.n_threads;
    const size_t n = src.n_samples();
    const double per_snp = 4.0 * double((n + 1) / 2);
    const size_t B = std::max<size_t>(8, env_size("FASTLD_MEM_B", 256));   // test override
    // compute phase: store (A+B slots) + decode batch + block buffers (nt * B * (A+B) * 8 * ~2);
    // write phase (store freed): writer buffers only. Solve the compute phase for A.
    const double fixed = mm.base + mm.batch;
    const double per_slot = per_snp + double(nt) * double(B) * 8.0 * 2.0;
    if (limit < mm.base + mm.writer && !getenv("FASTLD_MEM_A"))
        die("--memory " + std::to_string(o.memory_gb) + " GB cannot be met: writing the output needs " +
            std::to_string((mm.base + mm.writer) / 1e9) + " GB");
    const double room = limit - fixed;
    if (room < per_slot * (64 + B) && !getenv("FASTLD_MEM_A"))
        die("--memory " + std::to_string(o.memory_gb) + " GB cannot be met: at least " +
            std::to_string((fixed + per_slot * (64 + B)) / 1e9) + " GB are needed for " + std::to_string(n) + " samples");
    size_t A = std::max<size_t>(64, size_t(room / per_slot) - B);
    if (getenv("FASTLD_MEM_A")) A = std::max<size_t>(1, env_size("FASTLD_MEM_A", A));   // test override
    char msg[160];
    snprintf(msg, sizeof msg, "%.3f GB -> blocked: %zu SNPs per block, LD matrix in a temporary file", o.memory_gb, A);
    info("Memory limit", msg);

    auto t = Clock::now();
    DenseStore ds; BitStore bs; bool bits_mode = false;
    vector<vector<uint32_t>> miss_lists;
    decode_all(src, M, o, st, ds, bs, bits_mode, miss_lists, A);   // statistics for all SNPs, first block stored
    malloc_trim(0);
    tm.parse = secs(t);
    info("Parsing dosages", "done! (" + std::to_string(tm.parse) + " s)");
    t = Clock::now();
    vector<uint32_t> act;
    uint64_t nmiss = 0;
    for (size_t j = 0; j < M; j++) {
        if (st[j].na || st[j].mono) continue;
        act.push_back((uint32_t)j); nmiss += st[j].n_miss;
        const uint64_t mk = 2 * uint64_t(st[j].den);
        if (ds.float_mode || (size_t(INT32_MAX) / (mk * mk)) < 64)
            die("--memory applies to the exact integer paths; this region needs the floating-point path (probabilities with more "
                "than 10 bits), which needs about " + std::to_string((mm.base + 2 * mm.dense + 2 * mm.tri) / 1e9) + " GB");
    }
    const double miss_max = getenv("FASTLD_MISSMAX") ? atof(getenv("FASTLD_MISSMAX")) : 0.025;
    if (!act.empty() && double(nmiss) / (double(act.size()) * double(n)) > miss_max)
        die("--memory applies to the exact integer paths; this region's missing rate needs the floating-point path");
    const char* kname; ldk::pick_dense(&kname);
    info("Correlation path", string("exact integer (") + kname + "), blocked, " + std::to_string(act.size()) + " SNPs");
    auto F = std::make_unique<FileTri>(M, o.tmp_dir);
    const size_t Ma = act.size();
    size_t passes = 0;
    for (size_t a0 = 0; a0 < Ma; a0 += A) {
        const size_t a1 = std::min(Ma, a0 + A);
        if (a0 > 0) { ds.nslots = 0; ds.slot2snp.clear(); decode_list(src, act, a0, a1, ds, st, nt); passes++; }
        ld_integer_core(ds, M, st, nt, miss_lists, nullptr, F.get(), 0, a1 - a0);          // A x A
        for (size_t b0 = a1; b0 < Ma; b0 += B) {
            const size_t b1 = std::min(Ma, b0 + B);
            ds.nslots = a1 - a0; ds.slot2snp.resize(a1 - a0);
            decode_list(src, act, b0, b1, ds, st, nt);                                    // A x B
            ld_integer_core(ds, M, st, nt, miss_lists, nullptr, F.get(), a1 - a0, a1 - a0);
        }
    }
    tm.ld = secs(t);
    info("Computing correlations", "done! (" + std::to_string(tm.ld) + " s)");
    return F;
}

std::unique_ptr<LDMatrix> compute_ld(const Source& src, size_t M, const Opts& o, vector<SnpStats>& st, Timing& tm) {
    {
        const MemModel mm = mem_model(src.n_samples(), M, o.n_threads);
        const double need = peak_normal(mm, M);
        char msg[96]; snprintf(msg, sizeof msg, "%.3f GB (exact integer path; less for hard calls)", need / 1e9);
        info("Memory needed", msg);
        if (o.memory_gb > 0 && need > o.memory_gb * 1e9) return compute_ld_blocked(src, M, o, st, tm, o.memory_gb * 1e9, mm);
    }
    auto t = Clock::now();
    std::unique_ptr<LDMatrix> R;
    {
        DenseStore ds; BitStore bs; bool bits_mode = false;
        vector<vector<uint32_t>> miss_lists;
        decode_all(src, M, o, st, ds, bs, bits_mode, miss_lists);
        malloc_trim(0);   // return the freed decode buffers (glibc keeps them in the heap otherwise)
        if (getenv("FASTLD_PROFILE")) fprintf(stderr, "profile: after decode rss %ld MB\n", rss_mb());
        tm.parse = secs(t);
        info("Parsing dosages", "done! (" + std::to_string(tm.parse) + " s)");

        t = Clock::now();
        if (bits_mode) R = ld_bits(bs, M, st, o.n_threads);
        else {
            bool int_ok = !o.force_double && !ds.float_mode;
            uint64_t nmiss = 0, nact = ds.nslots + ds.nsparse();
            auto chk = [&](const SnpStats& a) {
                nmiss += a.n_miss;
                uint64_t mk = 2 * uint64_t(a.den);
                if ((size_t(INT32_MAX) / (mk * mk)) < 64) int_ok = false;      // > 10-bit probabilities
            };
            for (size_t s = 0; s < ds.nslots; s++) chk(st[ds.slot2snp[s]]);
            for (size_t u = 0; u < ds.nsparse(); u++) chk(st[ds.sp_snp[u]]);
            // missing data: exact integer path while the missing lists are short (crossover measured)
            const double miss_rate = nact ? double(nmiss) / (double(nact) * double(src.n_samples())) : 0.0;
            const double miss_max = getenv("FASTLD_MISSMAX") ? atof(getenv("FASTLD_MISSMAX")) : 0.025;
            if (nmiss && miss_rate > miss_max) int_ok = false;
            if (int_ok && nmiss && ds.nsparse()) fold_sparse(ds, st);           // missing data: dense panels only
            R = int_ok ? ld_integer(ds, M, st, o.n_threads, miss_lists) : ld_double(ds, M, st, miss_lists, o);
        }
    }
    tm.ld = secs(t);
    info("Computing correlations", "done! (" + std::to_string(tm.ld) + " s)");
    return R;
}

// --stats: genotype summary without LD
void print_stats(const Source& src, size_t M, int nt) {
    const size_t n = src.n_samples();
    vector<uint8_t> bits(M), hard(M);
    vector<uint64_t> nmiss(M), nzero(M), nobs(M);
    vector<double> maf(M);
#pragma omp parallel num_threads(nt)
    {
        Var v;
        auto ctx = src.make_ctx();
#pragma omp for schedule(dynamic, 8)
        for (size_t j = 0; j < M; j++) {
            src.decode(j, v, ctx.get());
            bits[j] = v.bits; hard[j] = v.hard; nmiss[j] = v.n_miss;
            const uint64_t no = n - v.n_miss; nobs[j] = no;
            const double twice = 2.0 * v.den;
            const bool flip = no && double(v.sx) > double(v.den) * double(no);   // mean above den: allele1 is minor
            const int32_t zero = flip ? int32_t(2 * v.den) : 0;
            uint64_t z = 0;
            for (size_t i = 0; i < n; i++) z += (!v.miss[i]) & (v.k[i] == zero);
            nzero[j] = z;
            double mean = no ? double(v.sx) / double(no) : 0.0;
            maf[j] = no ? std::min(mean, twice - mean) / twice : 0.0;
        }
    }
    uint8_t bmin = 255, bmax = 0; uint64_t hc = 0, miss = 0;
    uint64_t z[4] = {0, 0, 0, 0}, o[4] = {0, 0, 0, 0}, cnt[4] = {0, 0, 0, 0};
    for (size_t j = 0; j < M; j++) {
        bmin = std::min(bmin, bits[j]); bmax = std::max(bmax, bits[j]); hc += hard[j]; miss += nmiss[j];
        int b = maf[j] < 0.01 ? 1 : (maf[j] <= 0.05 ? 2 : 3);
        z[0] += nzero[j]; o[0] += nobs[j]; cnt[0]++;
        z[b] += nzero[j]; o[b] += nobs[j]; cnt[b]++;
    }
    auto pct = [](double a, double b) { char s[64]; if (b > 0) snprintf(s, 64, "%.4f", a / b); else snprintf(s, 64, "NA"); return string(s); };
    info("Bit depth min", M ? std::to_string(bmin) : "NA");
    info("Bit depth max", M ? std::to_string(bmax) : "NA");
    info("Missing rate", pct(double(miss), double(n) * double(M)));
    info("Hard-call SNP share", pct(double(hc), double(M)));
    info("Zero share (all)", pct(double(z[0]), double(o[0])));
    const char* lab[4] = {"", "Zero share MAF<0.01", "Zero share MAF.01-.05", "Zero share MAF>0.05"};
    for (int b = 1; b < 4; b++) info(lab[b], pct(double(z[b]), double(o[b])) + "  (" + std::to_string(cnt[b]) + " SNPs)");
}
