// fastLD: shared declarations
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define FASTLD_VERSION "1.0.0"

using std::string;
using std::vector;
using Clock = std::chrono::steady_clock;

[[noreturn]] void die(const string& m);
void info(const char* k, const string& v);
void warn(const string& m);
inline double secs(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
vector<string> split(const string& s, const string& delims, bool skip_empty);
string trim_cr(string s);

// ------------------------------------------------------------------ options
struct Opts {
    string in_files, compression = "medium", dataset, format;
    bool write_text = false, write_bcor = false, stats = false;
    int n_threads = 1;
    int precision = 6;              // significant digits of --write-text
    double sample_miss = 0.1;
    double memory_gb = 0;           // 0 = unlimited
    size_t chunk = 0;               // samples per dsyrk chunk (0 = auto)
    bool force_double = false;
    string tmp_dir;                 // directory for the --memory temporary LD file (set by main)
};

// ------------------------------------------------------------------ SNPs
struct Snp {
    string rsid, chrom, a1, a2;     // as in the Z file
    uint32_t pos = 0;
    int64_t off = -1, size = 0;     // BGEN: variant block; bed/pgen: variant index in `off`
    bool flip = false;              // bed/pgen: genotype file counts Z allele1 -> count 2 - x
};

// One decoded variant for the included samples: exact integer dosage numerators
// k = 2*(2^B-1) - 2*P(A1A1) - P(A1A2) (count of Z allele2 times den), 0 where missing.
struct Var {
    vector<int32_t> k;
    vector<uint8_t> miss;           // 1 = missing
    vector<uint32_t> miss_idx;      // sorted missing sample indices
    uint32_t den = 0, n_miss = 0;
    uint8_t bits = 0;
    int64_t sx = 0, sxx = 0;        // over observed samples
    bool hard = false;              // every observed probability is 0 or the maximum
    uint32_t hard_d = 0;            // >0: all observed k in {0, d, 2d} (exact hard-call scale)
    uint32_t c0 = 0, c2 = 0;        // observed values equal to 0 / to 2*den
    bool sums_done = false;         // decoder already filled sx, sxx, n_miss, c0, c2
};

// Per-SNP summary kept for the whole run
struct SnpStats {
    double mean = 0, inv_sd = 0;
    int64_t sx = 0, sxx = 0;
    uint32_t n_miss = 0, den = 0, hard_d = 0;
    uint32_t nnz = 0;               // observed values != 0 after orienting to the minor allele
    uint8_t bits = 0;
    bool na = false, mono = false, hard = false;
    bool flip = false;              // minor allele = allele1 (mean numerator above den)
};

// A genotype source (BGEN, bed, pgen) with the Z-file SNPs matched and samples selected.
struct ThreadCtx;  // per-thread scratch, defined by each source
class Source {
public:
    virtual ~Source() {}
    virtual size_t n_samples() const = 0;                    // included samples
    virtual size_t n_file_samples() const = 0;
    virtual std::unique_ptr<ThreadCtx, void (*)(ThreadCtx*)> make_ctx() const = 0;
    virtual void decode(size_t j, Var& v, ThreadCtx* ctx) const = 0;  // thread-safe
    virtual string describe() const = 0;
};

void finish_var(Var& v);  // fills sx, sxx, n_miss, miss_idx, hard_d from k/miss

std::unique_ptr<Source> open_bgen(const std::map<string, string>& col, const string& master, int line,
                                  vector<Snp>& snps, size_t n_samples_expected);
std::unique_ptr<Source> open_plink(const std::map<string, string>& col, const string& master, int line,
                                   vector<Snp>& snps, size_t n_samples_expected, const string& fmt);

vector<Snp> read_z(const string& fn);
vector<string> read_id_file_first_col(const string& fn, int skip_lines);
