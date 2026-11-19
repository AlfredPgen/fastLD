// BGEN v1.2/1.3 (layout 2) source: bgenix index matching and exact numerator decoding.
#include <fcntl.h>
#include <libdeflate.h>
#include <sqlite3.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>

#include <unordered_map>

#include "common.hpp"

// opt-in profiling (FASTLD_PROFILE): seconds per decode stage, summed over threads
double g_prof_inflate = 0, g_prof_emit = 0, g_prof_finish = 0;
static const bool g_prof = getenv("FASTLD_PROFILE") != nullptr;
static inline double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

static void pread_all(int fd, void* buf, size_t n, uint64_t off, const string& fn) {
    uint8_t* p = (uint8_t*)buf;
    while (n) {
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0) die("short read in file '" + fn + "' (truncated or unreadable at byte " + std::to_string(off) + ")");
        p += r; n -= r; off += r;
    }
}

// 8-bit unphased/phased fast path, all samples included: k = twice - ca*a - b, 0 where missing.
// Returns non-zero if some observed probability is not 0 or den. Branch-free so it vectorises;
// cloned for AVX-512BW / AVX2 with run-time dispatch (the binary itself needs only x86-64-v2).
// 8-bit fast path, all samples included: k = twice - ca*a - b, 0 where missing; also the exact
// sums over observed samples (sx, sxx), the missing count and the counts of k == 0 / k == 2den.
// Returns non-zero if some observed probability is not 0 or den. Branch-free so it vectorises;
// cloned for x86-64-v4 / AVX2 with run-time dispatch (the binary itself needs only x86-64-v2).
struct Emit8 { int64_t sx, sxx; uint32_t nm, c0, c2, bad; };
__attribute__((target_clones("arch=x86-64-v4", "avx2", "default")))
static Emit8 emit8_all(const uint8_t* __restrict pr, const uint8_t* __restrict ploidy, size_t n, int32_t twice, int32_t ca,
                       int32_t den, int32_t* __restrict k, uint8_t* __restrict m) {
    uint32_t bad = 0, nm = 0, c0 = 0, c2 = 0;
    int64_t sx = 0, sxx = 0;
    for (size_t i = 0; i < n; i++) {
        const int32_t a = pr[2 * i], b = pr[2 * i + 1];
        const int32_t mi = ploidy[i] >> 7;
        const int32_t keep = mi - 1;                     // all ones if observed, 0 if missing
        const int32_t x = (twice - ca * a - b) & keep;
        k[i] = x;
        m[i] = (uint8_t)mi;
        nm += (uint32_t)mi;
        sx += x; sxx += x * x;                           // x <= 2*255: x*x fits int32
        c0 += (uint32_t)((x == 0) & (keep != 0));
        c2 += (uint32_t)(x == twice);
        bad |= (uint32_t)((((a != 0) & (a != den)) | ((b != 0) & (b != den))) & keep);
    }
    return {sx, sxx, nm, c0, c2, bad};
}

struct BgenCtx {
    vector<uint8_t> raw, unz;
    libdeflate_decompressor* dz = nullptr;
    ~BgenCtx() { if (dz) libdeflate_free_decompressor(dz); }
};

class BgenSource : public Source {
public:
    string fn;
    int fd = -1;
    uint64_t fsize = 0;
    uint32_t N = 0, M = 0, flags = 0, first_off = 0;
    vector<string> ids;
    vector<uint32_t> incl;          // included sample indices, empty = all
    size_t n_incl = 0;
    const vector<Snp>* snps = nullptr;

    ~BgenSource() { if (fd >= 0) close(fd); }
    size_t n_samples() const override { return n_incl; }
    size_t n_file_samples() const override { return N; }
    string describe() const override { return "BGEN"; }
    std::unique_ptr<ThreadCtx, void (*)(ThreadCtx*)> make_ctx() const override {
        BgenCtx* c = new BgenCtx; c->dz = libdeflate_alloc_decompressor();
        return {reinterpret_cast<ThreadCtx*>(c), [](ThreadCtx* p) { delete reinterpret_cast<BgenCtx*>(p); }};
    }
    void read_header();
    void decode(size_t j, Var& v, ThreadCtx* ctx) const override;
};

void BgenSource::read_header() {
    struct stat sb;
    if (fstat(fd, &sb) != 0) die("cannot stat BGEN file '" + fn + "'");
    fsize = (uint64_t)sb.st_size;
    if (fsize < 24) die("This is not a BGEN file: '" + fn + "' (file too short)");
    uint8_t b[24];
    pread_all(fd, b, 24, 0, fn);
    uint32_t LH;
    memcpy(&first_off, b, 4); memcpy(&LH, b + 4, 4); memcpy(&M, b + 8, 4); memcpy(&N, b + 12, 4);
    if (memcmp(b + 16, "bgen", 4) != 0 && memcmp(b + 16, "\0\0\0\0", 4) != 0) die("This is not a BGEN file: '" + fn + "'");
    if (LH < 20 || uint64_t(4) + LH > fsize || LH > first_off) die("BGEN file '" + fn + "' has an invalid header length " + std::to_string(LH));
    if (uint64_t(first_off) + 4 > fsize) die("BGEN file '" + fn + "' has an invalid first-variant offset");
    pread_all(fd, &flags, 4, 4 + LH - 4, fn);
    if (((flags >> 2) & 0xF) != 2) die("BGEN file '" + fn + "' must use layout 2 (BGEN v1.2/v1.3)");
    if ((flags & 3) == 3) die("BGEN file '" + fn + "' has an unknown compression type");
    if (flags >> 31) {  // embedded sample identifiers
        uint32_t LS, NS;
        if (uint64_t(4) + LH + 8 > uint64_t(first_off) + 4) die("BGEN file '" + fn + "': sample block outside header area");
        pread_all(fd, &LS, 4, 4 + LH, fn);
        pread_all(fd, &NS, 4, 4 + LH + 4, fn);
        if (LS < 8 || uint64_t(LH) + LS > first_off) die("BGEN file '" + fn + "' has an invalid sample block length");
        if (NS != N) die("BGEN file '" + fn + "': sample block lists " + std::to_string(NS) + " samples, header " + std::to_string(N));
        vector<char> s(LS);
        pread_all(fd, s.data(), LS, 4 + LH, fn);
        size_t p = 8;
        for (uint32_t i = 0; i < N; i++) {
            if (p + 2 > LS) die("BGEN file '" + fn + "': truncated sample identifier block");
            uint16_t L; memcpy(&L, &s[p], 2);
            if (p + 2 + L > LS) die("BGEN file '" + fn + "': truncated sample identifier block");
            ids.emplace_back(&s[p + 2], L); p += 2 + L;
        }
    }
}

// bounds-checked cursor inside one buffer
struct Cur {
    const uint8_t* p; const uint8_t* e; const char* what; const string* rs;
    void need(size_t n) const { if ((size_t)(e - p) < n) die(string("corrupt BGEN variant block (") + what + ") for SNP '" + *rs + "'"); }
    uint16_t u16() { need(2); uint16_t v; memcpy(&v, p, 2); p += 2; return v; }
    uint32_t u32() { need(4); uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
    void skip(size_t n) { need(n); p += n; }
};

void BgenSource::decode(size_t j, Var& v, ThreadCtx* tctx) const {
    BgenCtx& c = *reinterpret_cast<BgenCtx*>(tctx);
    const Snp& s = (*snps)[j];
    if (s.off < 0 || s.size <= 0 || uint64_t(s.off) + uint64_t(s.size) > fsize)
        die("variant block of SNP '" + s.rsid + "' lies outside BGEN file '" + fn + "' (truncated file or stale .bgi index)");
    const double tp0 = g_prof ? now_s() : 0;
    c.raw.resize(s.size);
    pread_all(fd, c.raw.data(), s.size, s.off, fn);
    Cur q{c.raw.data(), c.raw.data() + c.raw.size(), "header", &s.rsid};
    q.skip(q.u16()); q.skip(q.u16()); q.skip(q.u16()); q.skip(4);   // varid, rsid, chrom, pos
    uint16_t K = q.u16();
    if (K != 2) die("SNP '" + s.rsid + "' is not biallelic");
    for (int a = 0; a < K; a++) q.skip(q.u32());
    uint32_t C = q.u32();
    q.what = "genotype block";
    const uint8_t* d; size_t D;
    if ((flags & 3) == 0) { q.need(C); d = q.p; D = C; }
    else {
        if (C < 4) die("corrupt BGEN variant block for SNP '" + s.rsid + "'");
        D = q.u32(); q.need(C - 4);
        if (D > (size_t(1) << 34)) die("corrupt BGEN variant block for SNP '" + s.rsid + "'");
        c.unz.resize(D);
        if ((flags & 3) == 1) {
            size_t got;
            if (libdeflate_zlib_decompress(c.dz, q.p, C - 4, c.unz.data(), D, &got) != LIBDEFLATE_SUCCESS || got != D)
                die("zlib error for SNP '" + s.rsid + "' in BGEN file '" + fn + "'");
        } else {
            size_t r = ZSTD_decompress(c.unz.data(), D, q.p, C - 4);
            if (ZSTD_isError(r) || r != D) die("zstd error for SNP '" + s.rsid + "' in BGEN file '" + fn + "'");
        }
        d = c.unz.data();
    }
    const double tp1 = g_prof ? now_s() : 0;
    if (D < 10 + size_t(N)) die("corrupt BGEN genotype block for SNP '" + s.rsid + "'");
    uint32_t n32; memcpy(&n32, d, 4);
    uint16_t k2; memcpy(&k2, d + 4, 2);
    if (n32 != N) die("sample count mismatch in SNP '" + s.rsid + "'");
    if (k2 != 2) die("SNP '" + s.rsid + "' is not biallelic");
    if (d[6] != 2 || d[7] != 2) die("SNP '" + s.rsid + "': only diploid samples are supported");
    const uint8_t* ploidy = d + 8;
    const uint8_t phased = d[8 + N], B = d[9 + N];
    const uint8_t* pr = d + 10 + N;
    if (B == 0 || B > 32) die("SNP '" + s.rsid + "': invalid probability bit depth");
    if (B > 23) die("SNP '" + s.rsid + "': probability bit depth > 23 not supported");
    const size_t nbytes = (2 * size_t(N) * B + 7) / 8;
    if (D < 10 + size_t(N) + nbytes) die("corrupt BGEN genotype block for SNP '" + s.rsid + "' (probabilities truncated)");
    const uint32_t den = (uint32_t(1) << B) - 1;
    const int32_t twice = int32_t(2 * den);
    const int32_t ca = phased ? 1 : 2;  // unphased: k = 2den - 2P(11) - P(12); phased: k = 2den - h1 - h2

    const size_t n = n_incl;
    v.k.resize(n); v.miss.resize(n);
    int32_t* kk = v.k.data(); uint8_t* mm = v.miss.data();
    const bool all = incl.empty();
    const uint32_t* inc = incl.data();
    uint32_t bad = 0;  // OR of "probability not in {0, den}" over observed samples
    if (B == 8) {
        const int32_t dd = (int32_t)den;
        if (all) {
            const Emit8 r = emit8_all(pr, ploidy, n, twice, ca, dd, kk, mm);
            bad = r.bad;
            v.sx = r.sx; v.sxx = r.sxx; v.n_miss = r.nm; v.c0 = r.c0; v.c2 = r.c2; v.sums_done = true;
        } else {
            for (size_t o = 0; o < n; o++) {
                size_t i = inc[o];
                int32_t a = pr[2 * i], b = pr[2 * i + 1];
                uint8_t mi = ploidy[i] >> 7;
                kk[o] = mi ? 0 : twice - ca * a - b;
                mm[o] = mi;
                bad |= mi ? 0u : (uint32_t)(((a != 0) & (a != dd)) | ((b != 0) & (b != dd)));
            }
        }
    } else if (B == 16) {
        const int32_t dd = (int32_t)den;
        for (size_t o = 0; o < n; o++) {
            size_t i = all ? o : inc[o];
            uint16_t a16, b16; memcpy(&a16, pr + 4 * i, 2); memcpy(&b16, pr + 4 * i + 2, 2);
            int32_t a = a16, b = b16;
            uint8_t mi = ploidy[i] >> 7;
            kk[o] = mi ? 0 : twice - ca * a - b;
            mm[o] = mi;
            bad |= mi ? 0u : (uint32_t)(((a != 0) & (a != dd)) | ((b != 0) & (b != dd)));
        }
    } else {
        const uint64_t mask = (uint64_t(1) << B) - 1;
        auto rd = [&](uint64_t bit) {
            uint64_t w = 0; size_t by = bit / 8;
            memcpy(&w, pr + by, std::min<size_t>(8, nbytes - by));
            return int32_t((w >> (bit % 8)) & mask);
        };
        for (size_t o = 0; o < n; o++) {
            size_t i = all ? o : inc[o];
            int32_t a = rd(uint64_t(2 * i) * B), b = rd(uint64_t(2 * i + 1) * B);
            uint8_t mi = ploidy[i] >> 7;
            kk[o] = mi ? 0 : twice - ca * a - b;
            mm[o] = mi;
            bad |= mi ? 0u : (uint32_t)(((a != 0) & (a != (int32_t)den)) | ((b != 0) & (b != (int32_t)den)));
        }
    }
    if (!(B == 8 && all)) v.sums_done = false;
    v.den = den; v.bits = B; v.hard = (bad == 0);
    const double tp2 = g_prof ? now_s() : 0;
    finish_var(v);
    if (g_prof) {
        const double tp3 = now_s();
#pragma omp atomic
        g_prof_inflate += tp1 - tp0;
#pragma omp atomic
        g_prof_emit += tp2 - tp1;
#pragma omp atomic
        g_prof_finish += tp3 - tp2;
    }
}

static void match_bgi(const string& bgi, const string& bgen, const string& zfn, vector<Snp>& snps) {
    sqlite3* db;
    if (sqlite3_open_v2(bgi.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) die("cannot open BGI file '" + bgi + "'");
    sqlite3_stmt* st;
    const char* q = "SELECT file_start_position, size_in_bytes FROM Variant WHERE chromosome=?1 AND position=?2 AND rsid=?3 AND allele1=?4 AND allele2=?5";
    if (sqlite3_prepare_v2(db, q, -1, &st, nullptr) != SQLITE_OK) die("BGI file '" + bgi + "' is not a bgenix index");
    bool bad = false;
    for (auto& s : snps) {
        sqlite3_reset(st);
        sqlite3_bind_text(st, 1, s.chrom.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(st, 2, s.pos);
        sqlite3_bind_text(st, 3, s.rsid.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(st, 4, s.a1.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(st, 5, s.a2.c_str(), -1, SQLITE_STATIC);
        if (sqlite3_step(st) == SQLITE_ROW) { s.off = sqlite3_column_int64(st, 0); s.size = sqlite3_column_int64(st, 1); }
        else { fprintf(stderr, "Error : SNP with rsID '%s' in file '%s' could not be matched with SNP in BGEN file '%s'!\n", s.rsid.c_str(), zfn.c_str(), bgen.c_str()); bad = true; }
    }
    sqlite3_finalize(st); sqlite3_close(db);
    if (bad) { fputc('\n', stderr); exit(1); }
}

std::unique_ptr<Source> open_bgen(const std::map<string, string>& c, const string& master, int line,
                                  vector<Snp>& snps, size_t n_samples) {
    auto get = [&](const string& k) -> string { auto it = c.find(k); return it == c.end() ? "" : it->second; };
    auto src = std::make_unique<BgenSource>();
    src->fn = get("bgen");
    string bgi = get("bgi").empty() ? src->fn + ".bgi" : get("bgi");
    info("BGEN file", src->fn); info("BGEN BGI file", bgi);
    src->fd = open(src->fn.c_str(), O_RDONLY);
    if (src->fd < 0) die("cannot open BGEN file '" + src->fn + "'");
    src->read_header();
    src->n_incl = src->N;
    if (!get("incl").empty()) {
        vector<string> ids = !get("sample").empty() ? read_id_file_first_col(get("sample"), 2) : src->ids;
        if (ids.size() != src->N) die("sample identifiers do not match number of samples in BGEN file '" + src->fn + "'");
        std::unordered_map<string, int> want;
        for (auto& s : read_id_file_first_col(get("incl"), 0)) want[s] = 1;
        for (uint32_t i = 0; i < src->N; i++) if (want.count(ids[i])) src->incl.push_back(i);
        src->n_incl = src->incl.size();
        if (src->n_incl != n_samples) die("Master file '" + master + "': " + std::to_string(n_samples) + " samples specified in column 'n_samples' but " + std::to_string(src->n_incl) + " samples included via file '" + get("incl") + "'");
        if (src->n_incl == src->N) src->incl.clear();
    } else if (n_samples != src->N) {
        die("Master file '" + master + "' is missing an entry in line " + std::to_string(line) + " column 'incl', because " + std::to_string(n_samples) +
            " samples are specified in the master file but BGEN file includes " + std::to_string(src->N) + " samples!");
    }
    match_bgi(bgi, src->fn, get("z"), snps);
    src->snps = &snps;
    return src;
}
