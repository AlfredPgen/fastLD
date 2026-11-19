#pragma once
#include "common.hpp"

// Read access to a symmetric LD matrix, row by row (out[j] for all j != i).
class LDMatrix {
public:
    virtual ~LDMatrix() {}
    virtual size_t size() const = 0;
    virtual void row(size_t i, double* out) const = 0;
    // rows a..b-1 into out[(i-a)*M + j] (values for j == i unspecified)
    virtual void rows(size_t a, size_t b, double* out) const { for (size_t i = a; i < b; i++) row(i, out + (i - a) * size()); }
    // r(x, y) for y = x+1..M-1 into out[y - x - 1]
    virtual void tail(size_t x, double* out) const = 0;
    // tails of x = a..b-1 into out[(x-a)*M + y - x - 1]
    virtual void tails(size_t a, size_t b, double* out) const { for (size_t x = a; x < b; x++) tail(x, out + (x - a) * size()); }
};

// full column-major M x M (symmetric) matrix
class FullMatrix : public LDMatrix {
public:
    vector<double> R; size_t M;
    explicit FullMatrix(size_t m) : R(m * m, 0.0), M(m) {}
    size_t size() const override { return M; }
    void row(size_t i, double* out) const override { memcpy(out, &R[i * M], M * sizeof(double)); }
    void tail(size_t x, double* out) const override { if (x + 1 < M) memcpy(out, &R[x * M + x + 1], (M - x - 1) * sizeof(double)); }
};

// Packed triangle stored per larger index: pair (x, y), x < y, at y*(y-1)/2 + x, i.e. "column" y
// holds r(0..y-1, y) contiguously. Backed by an anonymous zero-filled mapping, so pages are
// only resident once written (the integer paths fill it j-block by j-block).
class PackedTri : public LDMatrix {
public:
    double* T = nullptr; size_t M = 0, bytes = 0;
    explicit PackedTri(size_t m);
    ~PackedTri();
    PackedTri(const PackedTri&) = delete;
    PackedTri& operator=(const PackedTri&) = delete;
    static size_t off(size_t y) { return y * (y - 1) / 2; }
    double& at(size_t x, size_t y) { return T[off(y) + x]; }   // x < y
    size_t size() const override { return M; }
    void row(size_t i, double* out) const override {
        if (i) memcpy(out, T + off(i), i * sizeof(double));
        for (size_t j = i + 1; j < M; j++) out[j] = T[off(j) + i];
    }
    void rows(size_t a, size_t b, double* out) const override {
        for (size_t i = a; i < b; i++) if (i) memcpy(out + (i - a) * M, T + off(i), i * sizeof(double));
        // right of the diagonal: column j holds (i, j) for i < j contiguously -> tiled transpose
        for (size_t j0 = a + 1; j0 < M; j0 += 64) {
            const size_t j1 = std::min(M, j0 + 64);
            for (size_t i0 = a; i0 < b; i0 += 64) {
                const size_t i1 = std::min(b, i0 + 64);
                for (size_t j = j0; j < j1; j++) {
                    const size_t ie = std::min(i1, j);
                    const double* c = T + off(j);
                    for (size_t i = i0; i < ie; i++) out[(i - a) * M + j] = c[i];
                }
            }
        }
    }
    void tail(size_t x, double* out) const override { for (size_t y = x + 1; y < M; y++) out[y - x - 1] = T[off(y) + x]; }
    // tails of x = a..b-1 into out[(x-a)*M + y - x - 1]
    void tails(size_t a, size_t b, double* out) const override {
        for (size_t y0 = a + 1; y0 < M; y0 += 64) {
            const size_t y1 = std::min(M, y0 + 64);
            for (size_t x0 = a; x0 < b; x0 += 64) {
                const size_t x1 = std::min(b, x0 + 64);
                for (size_t y = y0; y < y1; y++) {
                    const size_t xe = std::min(x1, y);
                    const double* c = T + off(y);
                    for (size_t x = x0; x < xe; x++) out[(x - a) * M + (y - x - 1)] = c[x];
                }
            }
        }
    }
};

// Same layout as PackedTri, kept in an unlinked temporary file (--memory blocked mode): written
// column segment by column segment with pwrite, read by the writers with pread. File pages live
// in the page cache (reclaimable), not in the process's resident set.
class FileTri : public LDMatrix {
public:
    int fd = -1; size_t M = 0;
    FileTri(size_t m, const string& dir);
    ~FileTri();
    FileTri(const FileTri&) = delete;
    FileTri& operator=(const FileTri&) = delete;
    static size_t off(size_t y) { return y * (y - 1) / 2; }
    void put(size_t y, size_t x0, const double* v, size_t cnt);         // r(x0 .. x0+cnt-1, y), x0+cnt <= y
    void get(size_t y, size_t x0, double* v, size_t cnt) const;
    size_t size() const override { return M; }
    void row(size_t i, double* out) const override { rows(i, i + 1, out); }
    void rows(size_t a, size_t b, double* out) const override;
    void tail(size_t x, double* out) const override;
    void tails(size_t a, size_t b, double* out) const override;
};

int bcor_bytes(const string& c, uint8_t& code);
void write_text(const string& fn, const LDMatrix& R, const vector<SnpStats>& st, int nt, int precision);
void write_bcor(const string& fn, const LDMatrix& R, const vector<Snp>& snps, const vector<SnpStats>& st,
                uint32_t n_samples, const string& comp, int nt);
char* fmt_g(char* p, char* end, double v);
