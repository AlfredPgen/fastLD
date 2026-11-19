// PLINK 1 (.bed/.bim/.fam) and PLINK 2 (.pgen/.pvar/.psam) sources.
//
// SNP matching (Z file row -> genotype file variant): same position, same chromosome after removing
// a leading "chr" and leading zeros (23/24/25/26 read as X/Y/XY/MT), same UNORDERED allele pair; among several candidates the one
// whose ID equals the Z rsid is taken. Zero matches or several remaining matches are errors.
// The dosage counted is that of the Z file's allele2; variants listed the other way round are
// flipped (count = 2 - count).
//
// Numerators: hard calls (bed; pgen without a dosage track) are scaled as 8-bit BGEN hard calls
// (k = 255 * count, den = 255), so a hard-call data set gives byte-identical output in all three
// formats. pgen dosages are exact in 1/16384 units (k = 0..32768, den = 16384).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <unordered_map>

#include "common.hpp"
#include "pgenlib/pgenlib_read.h"

static string norm_chrom(string c) {
    if (c.size() > 3 && (c.compare(0, 3, "chr") == 0 || c.compare(0, 3, "CHR") == 0)) c = c.substr(3);
    size_t z = 0;
    while (z + 1 < c.size() && c[z] == '0') z++;
    c = c.substr(z);
    // PLINK numeric codes and common aliases of the non-autosomes
    if (c == "23" || c == "x") return "X";
    if (c == "24" || c == "y") return "Y";
    if (c == "25" || c == "xy") return "XY";
    if (c == "26" || c == "M" || c == "m" || c == "mt") return "MT";
    return c;
}

struct VarRec { string chrom, id, a1, a2; uint32_t pos; bool multi = false; };

// match Z SNPs to variants; sets s.off = variant index and s.flip
static void match_variants(vector<Snp>& snps, const vector<VarRec>& vars, const string& zfn, const string& gfn,
                           bool counted_is_a2) {
    std::unordered_map<string, vector<uint32_t>> by;
    for (uint32_t v = 0; v < vars.size(); v++) by[vars[v].chrom + ":" + std::to_string(vars[v].pos)].push_back(v);
    bool bad = false;
    for (auto& s : snps) {
        auto it = by.find(norm_chrom(s.chrom) + ":" + std::to_string(s.pos));
        vector<uint32_t> cand;
        if (it != by.end())
            for (uint32_t v : it->second) {
                const VarRec& r = vars[v];
                if ((r.a1 == s.a1 && r.a2 == s.a2) || (r.a1 == s.a2 && r.a2 == s.a1)) cand.push_back(v);
            }
        if (cand.size() > 1) {
            vector<uint32_t> byid;
            for (uint32_t v : cand) if (vars[v].id == s.rsid) byid.push_back(v);
            cand = byid;
        }
        if (cand.size() != 1) {
            fprintf(stderr, "Error : SNP with rsID '%s' in file '%s' (chromosome %s, position %u, alleles %s/%s) %s in '%s'!\n",
                    s.rsid.c_str(), zfn.c_str(), s.chrom.c_str(), s.pos, s.a1.c_str(), s.a2.c_str(),
                    cand.empty() ? "could not be matched with a variant" : "matches several variants", gfn.c_str());
            bad = true; continue;
        }
        const VarRec& r = vars[cand[0]];
        if (r.multi) { fprintf(stderr, "Error : SNP '%s' matches a multiallelic variant in '%s'\n", s.rsid.c_str(), gfn.c_str()); bad = true; continue; }
        s.off = cand[0];
        // the file counts allele (counted_is_a2 ? r.a2 : r.a1); flip unless that is the Z allele2
        const string& counted = counted_is_a2 ? r.a2 : r.a1;
        s.flip = counted != s.a2;
    }
    if (bad) { fputc('\n', stderr); exit(1); }
}

// incl: first token of each line matched to IID
static void select_samples(const vector<string>& iids, const std::map<string, string>& c, const string& master, int line,
                           size_t n_samples, const string& gfn, vector<uint32_t>& incl, size_t& n_incl) {
    auto it = c.find("incl");
    n_incl = iids.size();
    incl.clear();
    if (it != c.end() && !it->second.empty()) {
        std::unordered_map<string, int> want;
        for (auto& s : read_id_file_first_col(it->second, 0)) want[s] = 1;
        for (uint32_t i = 0; i < iids.size(); i++) if (want.count(iids[i])) incl.push_back(i);
        n_incl = incl.size();
        if (n_incl != n_samples) die("Master file '" + master + "': " + std::to_string(n_samples) + " samples specified in column 'n_samples' but " +
                                     std::to_string(n_incl) + " samples included via file '" + it->second + "'");
        if (n_incl == iids.size()) incl.clear();
    } else if (n_samples != iids.size()) {
        die("Master file '" + master + "' is missing an entry in line " + std::to_string(line) + " column 'incl', because " +
            std::to_string(n_samples) + " samples are specified in the master file but '" + gfn + "' includes " +
            std::to_string(iids.size()) + " samples!");
    }
}

// fill v from per-sample counts g (0,1,2; 3 = missing) of the counted allele, hard-call scale
static void var_from_counts(Var& v, const uint8_t* g, size_t n, bool flip) {
    v.k.resize(n); v.miss.resize(n);
    for (size_t i = 0; i < n; i++) {
        const uint8_t x = g[i];
        const bool m = x == 3;
        const int32_t c = flip ? 2 - x : x;
        v.k[i] = m ? 0 : 255 * c;
        v.miss[i] = m;
    }
    v.den = 255; v.bits = 8; v.hard = true; v.sums_done = false;
    finish_var(v);
}

// Byte lookup table for 2-bit packed genotypes (4 samples per byte): numerators (hard-call scale),
// missing flags and the exact per-byte contributions to sx, sxx, n_miss, c0 (k == 0), c2 (k == 510).
struct ByteLut {
    int32_t k[256][4]; uint8_t m[256][4];
    int32_t sx[256], sxx[256]; uint8_t nm[256], c0[256], c2[256];
    // count_of_code[c] = copies of the counted allele for 2-bit code c (3 = missing)
    void init(const uint8_t count_of_code[4], bool flip) {
        for (int b = 0; b < 256; b++) {
            sx[b] = sxx[b] = 0; nm[b] = c0[b] = c2[b] = 0;
            for (int t = 0; t < 4; t++) {
                const uint8_t x = count_of_code[(b >> (2 * t)) & 3];
                const bool mi = x == 3;
                const int32_t kk = mi ? 0 : 255 * (flip ? 2 - x : x);
                k[b][t] = kk; m[b][t] = mi;
                if (mi) nm[b]++; else { sx[b] += kk; sxx[b] += kk * kk; c0[b] += kk == 0; c2[b] += kk == 510; }
            }
        }
    }
};
static const ByteLut& lut_for(int kind, bool flip) {   // kind 0: bed codes, 1: pgen codes
    static ByteLut L[2][2];
    static bool done[2][2] = {{false, false}, {false, false}};
    static const uint8_t bed_cnt[4] = {2, 3, 1, 0}, pgen_cnt[4] = {0, 1, 2, 3};
#pragma omp critical(ldk_lut)
    if (!done[kind][flip]) { L[kind][flip].init(kind ? pgen_cnt : bed_cnt, flip); done[kind][flip] = true; }
    return L[kind][flip];
}
// decode 2-bit packed genotypes of N samples (all included) into v
static void var_from_packed(Var& v, const uint8_t* raw, size_t N, const ByteLut& L) {
    v.k.resize(N); v.miss.resize(N);
    int32_t* k = v.k.data(); uint8_t* m = v.miss.data();
    int64_t sx = 0, sxx = 0; uint32_t nm = 0, c0 = 0, c2 = 0;
    const size_t full = N / 4;
    for (size_t q = 0; q < full; q++) {
        const uint8_t b = raw[q];
        memcpy(k + 4 * q, L.k[b], 16); memcpy(m + 4 * q, L.m[b], 4);
        sx += L.sx[b]; sxx += L.sxx[b]; nm += L.nm[b]; c0 += L.c0[b]; c2 += L.c2[b];
    }
    for (size_t i = full * 4; i < N; i++) {
        const uint8_t b = raw[i >> 2], t = i & 3;
        k[i] = L.k[b][t]; m[i] = L.m[b][t];
        if (m[i]) nm++; else { sx += k[i]; sxx += int64_t(k[i]) * k[i]; c0 += k[i] == 0; c2 += k[i] == 510; }
    }
    v.sx = sx; v.sxx = sxx; v.n_miss = nm; v.c0 = c0; v.c2 = c2; v.sums_done = true;
    v.den = 255; v.bits = 8; v.hard = true;
    finish_var(v);
}

// ------------------------------------------------------------------ PLINK 1 bed
struct BedCtx { vector<uint8_t> raw, g; };
class BedSource : public Source {
public:
    string bed; int fd = -1; uint64_t fsize = 0;
    size_t N = 0, bpv = 0, n_incl = 0;
    vector<uint32_t> incl;
    const vector<Snp>* snps = nullptr;
    ~BedSource() { if (fd >= 0) close(fd); }
    size_t n_samples() const override { return n_incl; }
    size_t n_file_samples() const override { return N; }
    string describe() const override { return "PLINK 1 bed"; }
    std::unique_ptr<ThreadCtx, void (*)(ThreadCtx*)> make_ctx() const override {
        return {reinterpret_cast<ThreadCtx*>(new BedCtx), [](ThreadCtx* p) { delete reinterpret_cast<BedCtx*>(p); }};
    }
    void decode(size_t j, Var& v, ThreadCtx* tc) const override {
        BedCtx& c = *reinterpret_cast<BedCtx*>(tc);
        const Snp& s = (*snps)[j];
        const uint64_t off = 3 + uint64_t(s.off) * bpv;
        if (off + bpv > fsize) die("'" + bed + "' is truncated (variant of SNP '" + s.rsid + "')");
        c.raw.resize(bpv);
        size_t got = 0;
        while (got < bpv) {
            ssize_t r = pread(fd, c.raw.data() + got, bpv - got, off + got);
            if (r <= 0) die("short read in '" + bed + "'");
            got += r;
        }
        // bed codes vs bim: 00 hom first allele (bim col 5), 01 missing, 10 het, 11 hom second allele
        // -> count of the bim col-5 allele: 2, missing, 1, 0
        if (incl.empty()) { var_from_packed(v, c.raw.data(), N, lut_for(0, s.flip)); return; }
        static const uint8_t cnt[4] = {2, 3, 1, 0};
        c.g.resize(n_incl);
        if (incl.empty()) for (size_t i = 0; i < N; i++) c.g[i] = cnt[(c.raw[i >> 2] >> (2 * (i & 3))) & 3];
        else for (size_t o = 0; o < n_incl; o++) { const size_t i = incl[o]; c.g[o] = cnt[(c.raw[i >> 2] >> (2 * (i & 3))) & 3]; }
        var_from_counts(v, c.g.data(), n_incl, s.flip);
    }
};

static std::unique_ptr<Source> open_bed(const std::map<string, string>& c, const string& master, int line,
                                        vector<Snp>& snps, size_t n_samples) {
    auto get = [&](const string& k) -> string {
        auto it = c.find(k);
        if (it == c.end() || it->second.empty()) die("Master file '" + master + "' line " + std::to_string(line) + " needs column '" + k + "' for PLINK 1 input");
        return it->second;
    };
    auto src = std::make_unique<BedSource>();
    src->bed = get("bed");
    const string bim = get("bim"), fam = get("fam");
    info("PLINK bed file", src->bed);
    vector<VarRec> vars;
    {
        std::ifstream f(bim);
        if (!f) die("cannot open bim file '" + bim + "'");
        string l; int ln = 0;
        while (getline(f, l)) {
            ln++;
            auto t = split(trim_cr(l), " \t", true);
            if (t.empty()) continue;
            if (t.size() < 6) die("bim file '" + bim + "' line " + std::to_string(ln) + " has fewer than 6 columns");
            VarRec r; r.chrom = norm_chrom(t[0]); r.id = t[1]; r.pos = (uint32_t)strtoul(t[3].c_str(), nullptr, 10); r.a1 = t[4]; r.a2 = t[5];
            vars.push_back(r);
        }
    }
    vector<string> iids;
    {
        std::ifstream f(fam);
        if (!f) die("cannot open fam file '" + fam + "'");
        string l;
        while (getline(f, l)) { auto t = split(trim_cr(l), " \t", true); if (t.size() >= 2) iids.push_back(t[1]); }
    }
    src->N = iids.size();
    src->bpv = (src->N + 3) / 4;
    src->fd = open(src->bed.c_str(), O_RDONLY);
    if (src->fd < 0) die("cannot open bed file '" + src->bed + "'");
    struct stat sb; fstat(src->fd, &sb); src->fsize = sb.st_size;
    uint8_t magic[3] = {0, 0, 0};
    if (pread(src->fd, magic, 3, 0) != 3 || magic[0] != 0x6c || magic[1] != 0x1b) die("'" + src->bed + "' is not a PLINK 1 .bed file");
    if (magic[2] != 0x01) die("'" + src->bed + "' is sample-major; only SNP-major .bed files are supported");
    if (src->fsize != 3 + uint64_t(vars.size()) * src->bpv)
        die("'" + src->bed + "' has " + std::to_string(src->fsize) + " bytes, expected " + std::to_string(3 + uint64_t(vars.size()) * src->bpv) +
            " for " + std::to_string(vars.size()) + " variants x " + std::to_string(src->N) + " samples");
    select_samples(iids, c, master, line, n_samples, src->bed, src->incl, src->n_incl);
    match_variants(snps, vars, get("z"), bim, false);   // bed counts are of the bim column-5 allele
    src->snps = &snps;
    return src;
}

// ------------------------------------------------------------------ PLINK 2 pgen (pgenlib)
using namespace plink2;

struct PgenCtx {
    PgenReader pgr;
    unsigned char* alloc = nullptr;
    uintptr_t* genovec = nullptr; uintptr_t* dpresent = nullptr; uint16_t* dmain = nullptr;
    vector<uint8_t> g;
    vector<uint16_t> dz;   // dosage per sample (0xFFFF = none), file order
    ~PgenCtx() { PglErr e = kPglRetSuccess; CleanupPgr(&pgr, &e); aligned_free_cond(alloc); }
};

class PgenSource : public Source {
public:
    string pgen;
    PgenFileInfo pgfi;
    unsigned char* pgfi_alloc = nullptr;
    uint32_t max_vrec_width = 0;
    uintptr_t pgr_alloc_cacheline_ct = 0;
    size_t N = 0, n_incl = 0;
    bool dosages = false;
    vector<uint32_t> incl;
    const vector<Snp>* snps = nullptr;
    ~PgenSource() { PglErr e = kPglRetSuccess; CleanupPgfi(&pgfi, &e); aligned_free_cond(pgfi_alloc); }
    size_t n_samples() const override { return n_incl; }
    size_t n_file_samples() const override { return N; }
    string describe() const override { return "PLINK 2 pgen"; }
    std::unique_ptr<ThreadCtx, void (*)(ThreadCtx*)> make_ctx() const override {
        PgenCtx* c = new PgenCtx;
        PreinitPgr(&c->pgr);
        PgrSetFreadBuf(nullptr, &c->pgr);
        const uintptr_t vec_ct = DivUp(N, kBitsPerVec), gv_ct = DivUp(N, kNypsPerVec), dm_ct = DivUp(N, 2 * kInt32PerVec);
        const uintptr_t bytes = pgr_alloc_cacheline_ct * kCacheline + gv_ct * kBytesPerVec + vec_ct * kBytesPerVec + dm_ct * kBytesPerVec + 4 * kCacheline;
        if (cachealigned_malloc(bytes, &c->alloc)) die("out of memory (pgen reader)");
        if (PgrInit(pgen.c_str(), max_vrec_width, const_cast<PgenFileInfo*>(&pgfi), &c->pgr, c->alloc) != kPglRetSuccess)
            die("cannot open '" + pgen + "' for reading");
        unsigned char* p = c->alloc + pgr_alloc_cacheline_ct * kCacheline;
        c->genovec = (uintptr_t*)p; p += gv_ct * kBytesPerVec;
        c->dpresent = (uintptr_t*)p; p += vec_ct * kBytesPerVec;
        c->dmain = (uint16_t*)p;
        return {reinterpret_cast<ThreadCtx*>(c), [](ThreadCtx* q) { delete reinterpret_cast<PgenCtx*>(q); }};
    }
    void decode(size_t j, Var& v, ThreadCtx* tc) const override {
        PgenCtx& c = *reinterpret_cast<PgenCtx*>(tc);
        const Snp& s = (*snps)[j];
        PgrSampleSubsetIndex pssi;
        PgrClearSampleSubsetIndex(&c.pgr, &pssi);
        uint32_t dct = 0;
        if (PgrGetD(nullptr, pssi, (uint32_t)N, (uint32_t)s.off, &c.pgr, c.genovec, c.dpresent, c.dmain, &dct) != kPglRetSuccess)
            die("error reading variant of SNP '" + s.rsid + "' from '" + pgen + "'");
        auto code = [&](size_t i) { return uint8_t((c.genovec[i / kBitsPerWordD2] >> (2 * (i % kBitsPerWordD2))) & 3); };
        if (!dosages && incl.empty()) { var_from_packed(v, reinterpret_cast<const uint8_t*>(c.genovec), N, lut_for(1, s.flip)); return; }
        if (!dosages || dct == 0) {   // hard calls: ALT counts 0/1/2, 3 = missing
            c.g.resize(n_incl);
            if (incl.empty()) for (size_t i = 0; i < N; i++) c.g[i] = code(i);
            else for (size_t o = 0; o < n_incl; o++) c.g[o] = code(incl[o]);
            if (!dosages) { var_from_counts(v, c.g.data(), n_incl, s.flip); return; }
        }
        // dosages: exact numerators in 1/16384 units (32768 = 2 copies of ALT)
        vector<uint16_t>& dz = c.dz;
        dz.assign(N, 0xFFFF);
        {
            uint32_t t = 0;
            for (size_t i = 0; i < N && t < dct; i++)
                if ((c.dpresent[i / kBitsPerWord] >> (i % kBitsPerWord)) & 1) dz[i] = c.dmain[t++];
        }
        v.k.resize(n_incl); v.miss.resize(n_incl);
        bool hard = true;
        for (size_t o = 0; o < n_incl; o++) {
            const size_t i = incl.empty() ? o : incl[o];
            int32_t k; bool m = false;
            if (dz[i] != 0xFFFF) k = dz[i];
            else { const uint8_t x = code(i); if (x == 3) { m = true; k = 0; } else k = 16384 * x; }
            if (!m && s.flip) k = 32768 - k;
            if (!m && (k & 16383)) hard = false;
            v.k[o] = m ? 0 : k; v.miss[o] = m;
        }
        v.den = 16384; v.bits = 15; v.hard = hard; v.sums_done = false;
        finish_var(v);
    }
};

static std::unique_ptr<Source> open_pgen(const std::map<string, string>& c, const string& master, int line,
                                         vector<Snp>& snps, size_t n_samples) {
    auto get = [&](const string& k) -> string {
        auto it = c.find(k);
        if (it == c.end() || it->second.empty()) die("Master file '" + master + "' line " + std::to_string(line) + " needs column '" + k + "' for PLINK 2 input");
        return it->second;
    };
    auto src = std::make_unique<PgenSource>();
    src->pgen = get("pgen");
    const string pvar = get("pvar"), psam = get("psam");
    info("PLINK pgen file", src->pgen);
    if (pvar.size() > 4 && pvar.compare(pvar.size() - 4, 4, ".zst") == 0) die("compressed .pvar.zst files are not supported; decompress '" + pvar + "'");
    vector<VarRec> vars;
    {
        std::ifstream f(pvar);
        if (!f) die("cannot open pvar file '" + pvar + "'");
        string l; int ic = 0, ip = 1, ii = 2, ir = 3, ia = 4; bool hdr = false;
        while (getline(f, l)) {
            l = trim_cr(l);
            if (l.empty() || (l[0] == '#' && l.size() > 1 && l[1] == '#')) continue;
            auto t = split(l, " \t", true);
            if (!hdr && l[0] == '#') {   // #CHROM POS ID REF ALT ...
                hdr = true; t[0] = t[0].substr(1);
                for (int k = 0; k < (int)t.size(); k++) {
                    if (t[k] == "CHROM") ic = k; else if (t[k] == "POS") ip = k; else if (t[k] == "ID") ii = k;
                    else if (t[k] == "REF") ir = k; else if (t[k] == "ALT") ia = k;
                }
                continue;
            }
            if (!hdr) {   // headerless .pvar = .bim column order: CHROM ID CM POS ALT REF
                if (t.size() < 6) die("pvar file '" + pvar + "' has a line with fewer than 6 columns");
                VarRec r; r.chrom = norm_chrom(t[0]); r.id = t[1]; r.pos = (uint32_t)strtoul(t[3].c_str(), nullptr, 10); r.a1 = t[5]; r.a2 = t[4];
                r.multi = r.a2.find(',') != string::npos; vars.push_back(r); continue;
            }
            int mx = std::max(std::max(ic, ip), std::max(ii, std::max(ir, ia)));
            if ((int)t.size() <= mx) die("pvar file '" + pvar + "' has a short line");
            VarRec r; r.chrom = norm_chrom(t[ic]); r.id = t[ii]; r.pos = (uint32_t)strtoul(t[ip].c_str(), nullptr, 10);
            r.a1 = t[ir]; r.a2 = t[ia];   // a1 = REF, a2 = ALT (counted)
            r.multi = r.a2.find(',') != string::npos;
            vars.push_back(r);
        }
    }
    vector<string> iids;
    {
        std::ifstream f(psam);
        if (!f) die("cannot open psam file '" + psam + "'");
        string l; int iid_col = -1; bool first = true;
        while (getline(f, l)) {
            l = trim_cr(l);
            if (l.empty()) continue;
            auto t = split(l, " \t", true);
            if (first && l[0] == '#') {
                t[0] = t[0].substr(1);
                for (int k = 0; k < (int)t.size(); k++) if (t[k] == "IID") iid_col = k;
                if (iid_col < 0) die("psam file '" + psam + "' has no IID column");
                first = false; continue;
            }
            first = false;
            if (iid_col < 0) iid_col = t.size() >= 2 ? 1 : 0;   // headerless .psam = .fam
            if ((int)t.size() > iid_col) iids.push_back(t[iid_col]);
        }
    }
    // pgenlib file info
    PreinitPgfi(&src->pgfi);
    PgenHeaderCtrl header_ctrl;
    uintptr_t pgfi_alloc_cacheline_ct = 0;
    char errstr[kPglErrstrBufBlen];
    if (PgfiInitPhase1(src->pgen.c_str(), nullptr, (uint32_t)vars.size(), (uint32_t)iids.size(), &header_ctrl, &src->pgfi, &pgfi_alloc_cacheline_ct, errstr) != kPglRetSuccess)
        die(string("cannot open '") + src->pgen + "': " + (errstr + 7));
    if ((header_ctrl & 0x30) != 0 || src->pgfi.raw_variant_ct != vars.size()) {
        // allele_idx_offsets needed (multiallelic variants): build them from the pvar
        if (src->pgfi.raw_variant_ct != vars.size()) die("'" + src->pgen + "' and '" + pvar + "' disagree on the number of variants");
    }
    vector<uintptr_t> aio;
    bool any_multi = false;
    for (auto& r : vars) any_multi |= r.multi;
    if (any_multi) {
        if (cachealigned_malloc((vars.size() + 1) * sizeof(uintptr_t), &src->pgfi.allele_idx_offsets)) die("out of memory");
        uintptr_t* o = const_cast<uintptr_t*>(src->pgfi.allele_idx_offsets);
        o[0] = 0;
        for (size_t v = 0; v < vars.size(); v++) o[v + 1] = o[v] + 2 + std::count(vars[v].a2.begin(), vars[v].a2.end(), ',');
    }
    if ((header_ctrl & 0xc0) == 0xc0) {
        if (cachealigned_malloc(DivUp(vars.size(), kBitsPerWord) * kBytesPerWord, &src->pgfi.nonref_flags)) die("out of memory");
    }
    if (pgfi_alloc_cacheline_ct && cachealigned_malloc(pgfi_alloc_cacheline_ct * kCacheline, &src->pgfi_alloc)) die("out of memory");
    if (PgfiInitPhase2(header_ctrl, any_multi ? 1 : 0, 0, 0, 0, (uint32_t)vars.size(), &src->max_vrec_width, &src->pgfi, src->pgfi_alloc,
                       &src->pgr_alloc_cacheline_ct, errstr) != kPglRetSuccess)
        die(string("cannot read '") + src->pgen + "': " + (errstr + 7));
    src->N = src->pgfi.raw_sample_ct;
    if (src->N != iids.size()) die("'" + src->pgen + "' has " + std::to_string(src->N) + " samples, '" + psam + "' lists " + std::to_string(iids.size()));
    src->dosages = (src->pgfi.gflags & kfPgenGlobalDosagePresent) != 0;
    select_samples(iids, c, master, line, n_samples, src->pgen, src->incl, src->n_incl);
    match_variants(snps, vars, get("z"), pvar, true);   // pgen counts ALT
    src->snps = &snps;
    return src;
}

std::unique_ptr<Source> open_plink(const std::map<string, string>& c, const string& master, int line,
                                   vector<Snp>& snps, size_t n_samples, const string& fmt) {
    return fmt == "bed" ? open_bed(c, master, line, snps, n_samples) : open_pgen(c, master, line, snps, n_samples);
}
