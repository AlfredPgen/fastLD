// fastLD: exact LD (SNP correlation) matrices from BGEN, PLINK 1 bed and PLINK 2 pgen files.
// Accepts LDstore v2 master files and command lines and writes the same output formats
// (text .ld, optionally gzipped, and BCOR v1.1).  See README.md.
//
// Conventions (same as LDstore v2.0, verified empirically):
//   * SNP order = Z-file order; unmatched SNP -> error
//   * n_samples must equal #samples in the genotype file, unless an incl file is given
//   * missing rate >= --sample-miss -> correlations "nan" (BCOR NA code);
//     otherwise missing dosages are mean-imputed
//   * zero-variance SNP -> "-nan" (nan has precedence); diagonal always 1
//   * BCOR value = floor((r+1) * 2^(8b-2)), NA = 0xD0.. pattern
#include <fstream>

#include <omp.h>

#include "common.hpp"
#include "engine.hpp"
#include "kernels.hpp"
#include "output.hpp"

extern "C" void openblas_set_num_threads(int);

static void usage() {
    puts("fastLD " FASTLD_VERSION ": exact LD matrices from BGEN / PLINK bed / PLINK pgen genotypes\n\n"
         "--in-files      master file (';'-separated), one dataset per line. Genotype columns:\n"
         "                  BGEN:    z;bgen;bgi;[sample;incl;]ld|bcor;n_samples\n"
         "                  PLINK 1: z;bed;bim;fam;[incl;]ld|bcor;n_samples\n"
         "                  PLINK 2: z;pgen;pvar;psam;[incl;]ld|bcor;n_samples\n"
         "--write-text    write SNP correlations to text file (column 'ld'; gzipped if the name ends in .gz)\n"
         "--write-bcor    write SNP correlations to BCOR v1.1 file (column 'bcor')\n"
         "--n-threads     number of threads (default 1)\n"
         "--compression   BCOR value size: ultra-low|low|medium|high (default medium)\n"
         "--sample-miss   missing-rate threshold in [0,1] (default 0.1)\n"
         "--dataset       datasets (1-based line numbers of master file) e.g. 1,2 or 1|2\n"
         "--memory        memory limit in GB for the exact integer paths (default: unlimited)\n"
         "--precision     significant digits in --write-text, 1..15 (default 6)\n"
         "--format        bgen|bed|pgen: optional check of the master-file columns\n"
         "--stats         print genotype summary statistics per dataset; computes no LD\n"
         "--version       print version and compiled-in kernels\n"
         "--read-only-bgen  accepted (always the case)\n"
         "--write-bdose, --read-bdose, --bdose-version  accepted and ignored (no bdose support)\n"
         "--chunk-samples samples per dsyrk chunk (tuning; default auto)\n"
         "--force-double  always use the floating-point (BLAS dsyrk) path");
}

static Opts parse_args(int argc, char** argv) {
    Opts o;
    auto need = [&](int& i) -> string {
        if (i + 1 >= argc) die(string("missing value for ") + argv[i]);
        return argv[++i];
    };
    auto num = [&](const string& flag, const string& s) {
        char* e; double v = strtod(s.c_str(), &e);
        if (s.empty() || *e) die("invalid value '" + s + "' for " + flag);
        return v;
    };
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); exit(0); }
        else if (a == "--version") {
            printf("fastLD %s\nkernels: %s\n", FASTLD_VERSION, ldk::compiled_kernels().c_str());
            exit(0);
        }
        else if (a == "--in-files") o.in_files = need(i);
        else if (a == "--write-text") o.write_text = true;
        else if (a == "--write-bcor") o.write_bcor = true;
        else if (a == "--stats") o.stats = true;
        else if (a == "--n-threads") o.n_threads = std::max(1, (int)num(a, need(i)));
        else if (a == "--compression") o.compression = need(i);
        else if (a == "--sample-miss") o.sample_miss = num(a, need(i));
        else if (a == "--dataset") o.dataset = need(i);
        else if (a == "--memory") o.memory_gb = num(a, need(i));
        else if (a == "--precision") o.precision = (int)num(a, need(i));
        else if (a == "--format") o.format = need(i);
        else if (a == "--chunk-samples") o.chunk = strtoull(need(i).c_str(), nullptr, 10);
        else if (a == "--force-double") o.force_double = true;
        else if (a == "--read-only-bgen") {}
        else if (a == "--write-bdose" || a == "--read-bdose")
            warn(a + " is accepted for compatibility and ignored: fastLD reads genotypes directly and writes no .bdose file");
        else if (a == "--bdose-version") { string v = need(i); warn("--bdose-version " + v + " is accepted for compatibility and ignored"); }
        else die("unknown/unsupported command-line argument '" + a + "' (fastLD only computes LD matrices: --write-text/--write-bcor)");
    }
    if (o.in_files.empty()) die("'--in-files' is required");
    if (!o.stats && !o.write_text && !o.write_bcor) die("specify --write-text and/or --write-bcor (or --stats)");
    if (o.sample_miss < 0 || o.sample_miss > 1) die("--sample-miss must be in [0,1]");
    if (o.precision < 1 || o.precision > 15) die("--precision must be between 1 and 15");
    if (o.memory_gb < 0) die("--memory must be positive");
    if (!o.format.empty() && o.format != "bgen" && o.format != "bed" && o.format != "pgen") die("--format must be bgen, bed or pgen");
    return o;
}

struct Dataset { std::map<string, string> col; int line; };

static vector<Dataset> read_master(const string& fn) {
    std::ifstream f(fn);
    if (!f) die("cannot open master file '" + fn + "'");
    string line;
    if (!getline(f, line)) die("master file '" + fn + "' is empty");
    auto hdr = split(trim_cr(line), ";", false);
    vector<Dataset> ds; int ln = 1;
    while (getline(f, line)) {
        ln++; line = trim_cr(line);
        if (line.empty()) continue;
        auto v = split(line, ";", false);
        if (v.size() != hdr.size()) die("master file '" + fn + "' line " + std::to_string(ln) + " has wrong number of columns");
        Dataset d; d.line = ln;
        for (size_t i = 0; i < hdr.size(); i++) d.col[hdr[i]] = v[i];
        ds.push_back(d);
    }
    return ds;
}

// which genotype format the master-file columns describe
static string detect_format(const std::map<string, string>& c, const string& master) {
    auto has = [&](const char* k) { auto it = c.find(k); return it != c.end() && !it->second.empty(); };
    vector<string> f;
    if (has("bgen")) f.push_back("bgen");
    if (has("bed") || has("bim") || has("fam")) f.push_back("bed");
    if (has("pgen") || has("pvar") || has("psam")) f.push_back("pgen");
    if (f.empty()) die("Master file '" + master + "' names no genotype file (columns bgen, bed/bim/fam or pgen/pvar/psam)");
    if (f.size() > 1) die("Master file '" + master + "' mixes genotype formats (" + f[0] + " and " + f[1] + ") in one line");
    return f[0];
}

int main(int argc, char** argv) {
    auto T0 = Clock::now();
    Opts o = parse_args(argc, argv);
    puts("\n|--------------------------------------|\n| fastLD " FASTLD_VERSION "                         |\n|  genotypes -> exact LD matrix        |\n|--------------------------------------|\n");
    const int nt = o.n_threads;
    openblas_set_num_threads(nt);
    omp_set_num_threads(nt);
    uint8_t dummy; bcor_bytes(o.compression, dummy);

    auto ds = read_master(o.in_files);
    vector<int> sel;
    if (o.dataset.empty() || o.dataset == "all") for (size_t i = 0; i < ds.size(); i++) sel.push_back(i);
    else for (auto& t : split(o.dataset, ",|", true)) {
        int k = atoi(t.c_str());
        if (k < 1 || k > (int)ds.size()) die("--dataset " + t + " not in master file");
        sel.push_back(k - 1);
    }
    info("dataset", o.dataset.empty() ? "all" : o.dataset);
    info("compression", o.compression);
    info("n-threads", std::to_string(nt));
    info("sample-miss", std::to_string(o.sample_miss));

    for (int di : sel) {
        auto& c = ds[di].col;
        auto get = [&](const string& k) -> string {
            auto it = c.find(k);
            if (it == c.end() || it->second.empty()) die("Master file '" + o.in_files + "' is missing column/entry '" + k + "' in line " + std::to_string(ds[di].line));
            return it->second;
        };
        auto t_ds = Clock::now();
        printf("\n---------\nDATASET %d\n---------\n", di + 1);
        const string fmt = detect_format(c, o.in_files);
        if (!o.format.empty() && o.format != fmt) die("--format " + o.format + " does not agree with the master-file columns (" + fmt + ")");
        string zfn = get("z");
        const string ns_str = get("n_samples");
        char* e; size_t n_samples = strtoull(ns_str.c_str(), &e, 10);
        if (*e || ns_str[0] == '-') die("Master file '" + o.in_files + "': invalid n_samples '" + ns_str + "'");
        info("Z file", zfn);
        if (!o.stats) {
            if (o.write_text) info("LD matrix", get("ld"));
            if (o.write_bcor) info("BCOR file", get("bcor"));
        }
        vector<Snp> snps = read_z(zfn);
        std::unique_ptr<Source> src = fmt == "bgen" ? open_bgen(c, o.in_files, ds[di].line, snps, n_samples)
                                                    : open_plink(c, o.in_files, ds[di].line, snps, n_samples, fmt);
        info("Number of samples", std::to_string(src->n_samples()));
        info("Number of SNPs", std::to_string(snps.size()));
        if (o.stats) { print_stats(*src, snps.size(), nt); continue; }

        Timing tm;
        vector<SnpStats> st;
        {
            const string outp = o.write_text ? get("ld") : get("bcor");
            const size_t sl = outp.find_last_of('/');
            o.tmp_dir = sl == string::npos ? "." : (sl == 0 ? "/" : outp.substr(0, sl));
        }
        std::unique_ptr<LDMatrix> R = compute_ld(*src, snps.size(), o, st, tm);

        auto t = Clock::now();
        if (o.write_text) write_text(get("ld"), *R, st, nt, o.precision);
        if (o.write_bcor) write_bcor(get("bcor"), *R, snps, st, (uint32_t)src->n_samples(), o.compression, nt);
        info("Writing correlations", "done! (" + std::to_string(secs(t)) + " s)");
        info("Dataset run time", std::to_string(secs(t_ds)) + " s");
    }
    printf("\n- Run time              : %.3f s\n\n", secs(T0));
    return 0;
}
