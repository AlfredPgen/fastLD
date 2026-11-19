// Text (.ld) and BCOR v1.1 writers.
#include <charconv>

#include <libdeflate.h>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include "output.hpp"

PackedTri::PackedTri(size_t m) : M(m) {
    bytes = std::max<size_t>(m > 1 ? m * (m - 1) / 2 * sizeof(double) : 0, 4096);
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("out of memory allocating the LD matrix (" + std::to_string(bytes >> 20) + " MB)");
    if (getenv("FASTLD_TRI_THP")) madvise(p, bytes, MADV_HUGEPAGE);
    T = (double*)p;
}
PackedTri::~PackedTri() { if (T) munmap(T, bytes); }

FileTri::FileTri(size_t m, const string& dir) : M(m) {
    string tmpl = (dir.empty() ? string(".") : dir) + "/.fastLD-ld-XXXXXX";
    vector<char> name(tmpl.begin(), tmpl.end()); name.push_back(0);
    fd = mkstemp(name.data());
    if (fd < 0) die("--memory: cannot create a temporary file in '" + dir + "' for the LD matrix");
    unlink(name.data());   // removed automatically when closed
    const off_t bytes = off_t(m > 1 ? m * (m - 1) / 2 : 0) * off_t(sizeof(double));
    if (bytes && ftruncate(fd, bytes) != 0) die("--memory: cannot size the temporary LD file (disk full?)");
}
FileTri::~FileTri() { if (fd >= 0) close(fd); }
static void pio(bool wr, int fd, void* p, size_t n, off_t off) {
    char* c = (char*)p;
    while (n) {
        ssize_t r = wr ? pwrite(fd, c, n, off) : pread(fd, c, n, off);
        if (r <= 0) die(wr ? "--memory: write error on the temporary LD file (disk full?)" : "--memory: read error on the temporary LD file");
        c += r; n -= r; off += r;
    }
}
void FileTri::put(size_t y, size_t x0, const double* v, size_t cnt) { pio(true, fd, (void*)v, cnt * sizeof(double), off_t(off(y) + x0) * 8); }
void FileTri::get(size_t y, size_t x0, double* v, size_t cnt) const { pio(false, fd, v, cnt * sizeof(double), off_t(off(y) + x0) * 8); }
void FileTri::rows(size_t a, size_t b, double* out) const {
    for (size_t i = a; i < b; i++) if (i) get(i, 0, out + (i - a) * M, i);
    vector<double> seg(b - a);
    for (size_t j = a + 1; j < M; j++) {
        const size_t ie = std::min(b, j);
        get(j, a, seg.data(), ie - a);
        for (size_t i = a; i < ie; i++) out[(i - a) * M + j] = seg[i - a];
    }
}
void FileTri::tail(size_t x, double* out) const { tails(x, x + 1, out); }
void FileTri::tails(size_t a, size_t b, double* out) const {
    vector<double> seg(b - a);
    for (size_t y = a + 1; y < M; y++) {
        const size_t xe = std::min(b, y);
        get(y, a, seg.data(), xe - a);
        for (size_t x = a; x < xe; x++) out[(x - a) * M + (y - x - 1)] = seg[x - a];
    }
}

int bcor_bytes(const string& c, uint8_t& code) {
    if (c == "low") { code = 0; return 2; }
    if (c == "medium") { code = 1; return 4; }
    if (c == "high") { code = 2; return 8; }
    if (c == "ultra-low") { code = 3; return 1; }
    die("unknown --compression '" + c + "'");
}

// printf("%g") (6 significant digits).  Fast path for 1e-4 <= |v| < 1 (the
// bulk of LD values); anything near a rounding tie, a decade boundary or out
// of range goes through std::to_chars, which is exact, so output is
// byte-identical to printf in all cases.
char* fmt_g(char* p, char* end, double v) {
    double a = fabs(v);
    if (a >= 1e-4 && a < 1.0) {
        int X; double sc;
        if (a >= 0.1) { X = 1; sc = 1e6; } else if (a >= 0.01) { X = 2; sc = 1e7; } else if (a >= 0.001) { X = 3; sc = 1e8; } else { X = 4; sc = 1e9; }
        double m = a * sc, q = floor(m), f = m - q;
        if (fabs(f - 0.5) > 1e-6) {
            uint32_t dgt = uint32_t(q) + (f > 0.5);
            if (dgt >= 100000 && dgt < 1000000) {
                if (v < 0) *p++ = '-';
                *p++ = '0'; *p++ = '.';
                for (int z = 1; z < X; z++) *p++ = '0';
                char t[6];
                for (int z = 5; z >= 0; z--) { t[z] = char('0' + dgt % 10); dgt /= 10; }
                int last = 5; while (t[last] == '0') last--;
                memcpy(p, t, last + 1); return p + last + 1;
            }
        }
    }
    return std::to_chars(p, end, v, std::chars_format::general, 6).ptr;
}

// Text output. A file name ending in ".gz" gives gzip output: every row is one gzip member
// (rows are compressed in parallel; concatenated members are a valid gzip file for zcat, R's
// gzfile() and data.table::fread, Python's gzip, ...).
void write_text(const string& fn, const LDMatrix& R, const vector<SnpStats>& st, int nt, int precision) {
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) die("cannot write '" + fn + "'");
    const size_t M = R.size();
    const bool gz = fn.size() > 3 && fn.compare(fn.size() - 3, 3, ".gz") == 0;
    int gz_level = 6;
    if (const char* lv = getenv("FASTLD_GZIP_LEVEL")) gz_level = std::max(1, std::min(12, atoi(lv)));
    vector<libdeflate_compressor*> comp(gz ? size_t(std::max(nt, 1)) : size_t(0), nullptr);
    for (auto& c : comp) if (!(c = libdeflate_alloc_compressor(gz_level))) die("out of memory (gzip compressor)");
    vector<string> zbuf(comp.size());
    const size_t ROWS = std::max<size_t>(16, std::min<size_t>(256, (size_t(32) << 20) / (8 * std::max<size_t>(M, 1))));
    const size_t width = 8 + size_t(precision) + 8;  // sign, "0.", digits, "e-XXX", separator
    vector<string> buf(std::min(ROWS, std::max<size_t>(M, 1)));
    vector<double> blk(std::min(ROWS, std::max<size_t>(M, 1)) * M);
    for (size_t a = 0; a < M; a += ROWS) {
        size_t b = std::min(M, a + ROWS);
        R.rows(a, b, blk.data());
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (size_t i = a; i < b; i++) {
            const double* row = &blk[(i - a) * M];
            string& s = buf[i - a];
            s.resize(M * width + 2);
            char* p = s.data(); char* e = p + s.size();
            for (size_t j = 0; j < M; j++) {
                if (i == j) { *p++ = '1'; }
                else if (st[i].na || st[j].na) { memcpy(p, "nan", 3); p += 3; }
                else if (st[i].mono || st[j].mono) { memcpy(p, "-nan", 4); p += 4; }
                else if (precision == 6) p = fmt_g(p, e, row[j]);
                else p = std::to_chars(p, e, row[j], std::chars_format::general, precision).ptr;
                *p++ = ' ';
            }
            *p++ = '\n';
            s.resize(p - s.data());
            if (gz) {
                const size_t th = size_t(omp_get_thread_num());
                string& z = zbuf[th];
                z.resize(libdeflate_gzip_compress_bound(comp[th], s.size()));
                const size_t zn = libdeflate_gzip_compress(comp[th], s.data(), s.size(), z.data(), z.size());
                if (!zn) die("gzip compression failed for '" + fn + "'");
                s.assign(z.data(), zn);
            }
        }
        for (size_t i = a; i < b; i++)
            if (fwrite(buf[i - a].data(), 1, buf[i - a].size(), f) != buf[i - a].size()) die("write error on '" + fn + "'");
    }
    for (auto c : comp) libdeflate_free_compressor(c);
    if (fclose(f) != 0) die("write error on '" + fn + "'");
}

template <class T>
static void encode_tail(const double* tl, size_t x, size_t M, const vector<SnpStats>& st, T* out, int nbytes) {
    const double scale = ldexp(1.0, 8 * nbytes - 2);
    const T NA = T(T(0xD0) << (8 * (nbytes - 1)));
    const bool xbad = st[x].na || st[x].mono;
    for (size_t y = x + 1; y < M; y++) {
        T v;
        if (xbad || st[y].na || st[y].mono) v = NA;
        else {
            double r = tl[y - x - 1];
            r = r > 1.0 ? 1.0 : (r < -1.0 ? -1.0 : r);
            v = T(floor((r + 1.0) * scale));
        }
        *out++ = v;
    }
}

void write_bcor(const string& fn, const LDMatrix& R, const vector<Snp>& snps, const vector<SnpStats>& st,
                uint32_t n_samples, const string& comp, int nt) {
    uint8_t code; int nb = bcor_bytes(comp, code);
    const size_t M = R.size();
    string meta;
    for (size_t i = 0; i < M; i++) {
        const Snp& s = snps[i];
        uint32_t L = 20 + s.rsid.size() + s.chrom.size() + s.a1.size() + s.a2.size(), idx = i, pos = s.pos;
        uint16_t lr = s.rsid.size(), lc = s.chrom.size(); uint32_t l1 = s.a1.size(), l2 = s.a2.size();
        meta.append((char*)&L, 4); meta.append((char*)&idx, 4);
        meta.append((char*)&lr, 2); meta += s.rsid; meta.append((char*)&pos, 4);
        meta.append((char*)&lc, 2); meta += s.chrom;
        meta.append((char*)&l1, 4); meta += s.a1; meta.append((char*)&l2, 4); meta += s.a2;
    }
    uint64_t corr_off = 32 + meta.size();
    uint64_t npairs = uint64_t(M) * (M ? M - 1 : 0) / 2;
    uint64_t fsize = corr_off + npairs * nb;
    uint32_t ns = n_samples, nm = M;
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) die("cannot write '" + fn + "'");
    fwrite("bcor1.1", 1, 7, f); fwrite(&fsize, 8, 1, f); fwrite(&ns, 4, 1, f); fwrite(&nm, 4, 1, f);
    fwrite(&code, 1, 1, f); fwrite(&corr_off, 8, 1, f);
    fwrite(meta.data(), 1, meta.size(), f);
    // columns x = 0..M-2, entries y = x+1..M-1 (strictly lower triangle, column-major)
    const size_t COLS = std::max<size_t>(16, std::min<size_t>(512, (size_t(32) << 20) / (8 * std::max<size_t>(M, 1))));
    vector<vector<uint8_t>> buf(COLS);
    vector<double> blk(COLS * std::max<size_t>(M, 1));
    for (size_t a = 0; a + 1 < M; a += COLS) {
        size_t b = std::min(M - 1, a + COLS);
        R.tails(a, b, blk.data());
#pragma omp parallel for num_threads(nt) schedule(dynamic, 4)
        for (size_t x = a; x < b; x++) {
            const double* row = &blk[(x - a) * M];
            auto& v = buf[x - a]; v.resize((M - x - 1) * nb);
            switch (nb) {
                case 1: encode_tail(row, x, M, st, (uint8_t*)v.data(), 1); break;
                case 2: encode_tail(row, x, M, st, (uint16_t*)v.data(), 2); break;
                case 4: encode_tail(row, x, M, st, (uint32_t*)v.data(), 4); break;
                default: encode_tail(row, x, M, st, (uint64_t*)v.data(), 8); break;
            }
        }
        for (size_t x = a; x < b; x++)
            if (fwrite(buf[x - a].data(), 1, buf[x - a].size(), f) != buf[x - a].size()) die("write error on '" + fn + "'");
    }
    if (fclose(f) != 0) die("write error on '" + fn + "'");
}
