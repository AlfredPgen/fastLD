# fastLD

Fast, exact LD (SNP correlation) matrices for fine-mapping, computed straight from **BGEN**,
**PLINK 1 bed** or **PLINK 2 pgen** files for the SNPs listed in a Z file. Output is a full dense
matrix as text (optionally gzipped) or as a compact BCOR file, ready for SuSiE, FINEMAP and similar
tools.

* **Exact.** Correlations are computed from exact integer sums wherever the data allow (8-bit
  dosages, hard calls, low missingness). There every printed value is the correctly rounded true
  value, identical for any thread count and on any CPU.
* **Fast on one thread.** SIMD kernels for imputed dosages, a sparse path for low-frequency
  variants and a bit-plane path for hard calls. See [Benchmarks](#benchmarks).
* **Low memory.** Dosages are held in 2 bytes per genotype (2 bits for hard calls) and the matrix
  as a packed triangle; `--memory` caps usage for very large regions.
* **Drop-in.** Accepts LDstore v2 master files and command lines unchanged and writes BCOR v1.1.

## Install

Download `fastLD_linux_x86_64` from the [releases page](https://github.com/AlfredPgen/fastLD/releases)
and make it executable:

```bash
chmod +x fastLD_linux_x86_64
./fastLD_linux_x86_64 --version
```

It is a static binary for x86-64 Linux with no dependencies. It runs on any x86-64 CPU and picks
AVX2 or AVX-512 kernels at run time.

Build from source (Debian/Ubuntu):

```bash
sudo apt install g++ make libopenblas-dev libdeflate-dev libzstd-dev libsqlite3-dev
make            # ./fastLD (dynamic)
make static     # ./fastLD (static; also needs libgfortran-13-dev)
```

## Quick start

A Z file lists the SNPs of the region, in the order wanted in the matrix (extra columns are
ignored):

```
rsid chromosome position allele1 allele2
rs1 01 1000123 A G
rs2 01 1000456 C T
```

A master file says where the data are and where the output goes (`;`-separated, one region per
line):

```
z;bgen;bgi;ld;n_samples
region1.z;chr1.bgen;chr1.bgen.bgi;region1.ld.gz;487409
```

```bash
fastLD --in-files master --write-text --n-threads 1
```

## Master file

Columns are found by name; unknown columns are ignored. The genotype format follows from the
columns. `--format bgen|bed|pgen` optionally checks it.

| Input | Columns |
|---|---|
| BGEN v1.2/1.3 with a bgenix index | `z;bgen;bgi;[sample;incl;]ld\|bcor;n_samples` |
| PLINK 1 | `z;bed;bim;fam;[incl;]ld\|bcor;n_samples` |
| PLINK 2 | `z;pgen;pvar;psam;[incl;]ld\|bcor;n_samples` |

* `ld` is the text output, `bcor` the BCOR output. If the `ld` name ends in `.gz` the text is
  gzipped.
* `n_samples` must equal the number of samples in the genotype file. To use a subset, give `incl`
  (a file with one sample ID per line; BGEN: matched to the first column of the `.sample` file or
  to the IDs stored in the BGEN; PLINK: matched to IID) and set `n_samples` to the subset size.

**SNP matching and allele order.** The LD sign always refers to the Z file's `allele2`, so the
matrix lines up with summary statistics whose effect allele is `allele2`.

* BGEN: exact match of rsid, chromosome, position, allele1 and allele2 in the `.bgi` index.
* bed/pgen: same position, same chromosome (a leading `chr` and leading zeros are ignored; 23/24/25/26
  are read as X/Y/XY/MT) and the same pair of alleles in either order. Variants stored the other
  way round are flipped. If several variants fit, the one whose ID equals the Z-file rsid is used.
* A SNP that cannot be matched, or that matches more than one variant, stops the run with a
  message naming it.

## Command line

| Flag | |
|---|---|
| `--in-files FILE` | master file |
| `--write-text` | write the text matrix (column `ld`) |
| `--write-bcor` | write a BCOR v1.1 file (column `bcor`) |
| `--compression ultra-low\|low\|medium\|high` | BCOR value size: 1, 2, 4 or 8 bytes (default medium) |
| `--precision D` | significant digits in the text output, 1 to 15 (default 6) |
| `--n-threads N` | default 1; output is identical for any N |
| `--sample-miss X` | SNPs with a missing rate of X or more get `nan` (default 0.1) |
| `--memory GB` | memory limit; larger regions are processed in SNP blocks with identical output |
| `--dataset 1,2` | which master-file lines to run (default all) |
| `--format bgen\|bed\|pgen` | check that the master-file columns describe this format |
| `--stats` | print bit depth, missing rate, share of hard-call SNPs and of zero dosages; no LD |
| `--version` | version and the kernels usable on this CPU |

`--read-only-bgen` is accepted. `--write-bdose`, `--read-bdose` and `--bdose-version` are accepted
and ignored with a warning.

## Output

**Text.** An M x M matrix, values separated by spaces, 6 significant digits, diagonal `1`. Pairs
with a SNP above `--sample-miss` print as `nan`, pairs with a monomorphic SNP as `-nan`.
Missing genotypes below the threshold are mean-imputed (means and variances over each SNP's observed
samples; a missing value contributes zero after centring).

**Gzipped text.** Name the output `something.ld.gz`. The decompressed content is byte-identical to
the plain text. `zcat`, R (`data.table::fread`, `read.table`, `gzfile`) and Python read it directly.

**BCOR.** The strict lower triangle as fixed-size integers, `floor((r + 1) * 2^(8b-2))` for `b`
bytes per value, plus the SNP list. This is the format FINEMAP reads.

Sizes and write times for a 4,000-SNP region (16 million values), one thread:

| Output | File size | Write time | Precision |
|---|---|---|---|
| text, 6 digits | 182 MB | 0.6 s | 6 significant digits |
| text, gzipped (`.ld.gz`) | 61 MB | 4.2 s | same as text |
| BCOR medium | 32 MB | 0.07 s | 1e-9 |
| BCOR low | 16 MB | as medium | 6e-5 |

Gzipped text is the simple choice when the reader expects a text matrix. BCOR medium is the
smallest lossless-in-practice option: it is more precise than 6-digit text and half the size of the
gzipped file.

A full dense matrix is always written. Thresholded or banded matrices are deliberately not
offered: dropping entries breaks the consistency between LD and z-scores that fine-mapping relies
on.

## Loading into R

An R numeric matrix takes 8 bytes per value whatever the file looks like, so the file format does
not change memory use once loaded. It does change file size, load time and the peak during loading
(`fread` followed by `as.matrix` holds about two copies of the matrix).

`R/read_ld.R` (base R, no packages) reads BCOR files directly into a matrix:

```r
source("read_ld.R")
ld  <- read_bcor("region1.bcor")     # ld$R: M x M matrix, ld$snps: data.frame, ld$n_samples
fit <- susieR::susie_rss(z, ld$R, n = ld$n_samples)

R <- read_ld_text("region1.ld.gz")   # text output, plain or gzipped
```

Loading an 8,000-SNP matrix (R 4.3.3, one thread):

| Method | File size | Load time | Peak R memory |
|---|---|---|---|
| text + `data.table::fread` + `as.matrix` | 688 MB | 3.3 s | 1,310 MB |
| BCOR medium + `read_bcor` | 128 MB | 2.8 s | 805 MB |

## Benchmarks

One thread, `--write-text`, wall time including start-up. Laptop CPU: Intel Core i7-11370H
(AVX-512 VNNI). fastLD: median of 3 runs; LDstore v2.0: one run, same command line. Synthetic 8-bit
zlib BGEN data, 4,000 SNPs.

| Data | Samples | LDstore v2.0 | fastLD | Speed-up | Peak memory (LDstore / fastLD) |
|---|---|---|---|---|---|
| dense dosages, MAF 0.02 to 0.5 | 30,000 | 64.3 s | 5.2 s | 12x | 1,067 / 261 MB |
| imputed-like, MAF 0.001 to 0.5 | 50,000 | 108.5 s | 5.2 s | 21x | 1,677 / 371 MB |
| hard calls | 50,000 | 104.9 s | 3.0 s | 35x | 1,677 / 119 MB |

fastLD time by phase (seconds, typical run):

| Data | Read and decode | Correlations | Write text |
|---|---|---|---|
| dense dosages | 1.0 | 3.5 | 0.6 |
| imputed-like | 1.1 | 3.1 | 0.6 |
| hard calls | 0.5 | 1.5 | 0.5 |

With 4 threads the three runs take 2.3 s, 2.4 s and 1.3 s. The same hard-call data read from bed
or pgen take the same time as from BGEN and give byte-identical output.

Speed depends on the CPU: the AVX2 kernels are roughly 1.5 to 2 times slower than the AVX-512 ones,
and 16-bit BGEN files use the slower floating-point path. Raw logs and the results from the
development machine (a Sapphire Rapids VM) are in `benchmarks/results/`.

Compared with LDstore's output on the same data, 99.2% to 100% of the 6-digit values are identical
and the rest differ by one unit in the last digit (see [Compatibility](#compatibility-with-ldstore)).

## How it works

Each SNP is decoded to exact integer dosage numerators. Sums and cross products are accumulated as
integers and the correlation is evaluated once at the end in double precision.

* **Dosages (8-bit BGEN and similar):** 16-bit integers in 16-SNP panels, an AVX-512 VNNI
  outer-product kernel with AVX2 and scalar fallbacks, and cache blocking.
* **Low-frequency variants:** SNPs whose minor-allele dosage is zero for at least 90% of samples
  are stored as sparse lists, so their pairs cost the number of carriers and not the number of
  samples.
* **Hard calls** (bed, hard-call pgen, BGEN with 0/1 probabilities at any bit depth): two bit
  planes per SNP and popcount kernels (AVX-512 VPOPCNTDQ, AVX2, scalar). Missing genotypes are
  handled exactly with observed-sample masks.
* **Missing values in dosage data:** exact, by correcting the integer sums with per-SNP lists of
  missing samples (up to 2.5% missing on average).
* **Floating-point fallback** (OpenBLAS `dsyrk`): probabilities with 12 bits or more (for example
  16-bit BGEN from qctool or plink2), pgen dosages, or heavy missingness.

## Tests

| Target | What | Needs |
|---|---|---|
| `make test` | every text value equals the correctly rounded exact value (big-integer reference) on dense, imputed, hard-call, missing-data and 16-bit sets, for all kernels; gzip output; error handling | python3, numpy, scipy |
| `make test-plink` | BGEN, bed and pgen give byte-identical output, also with swapped alleles, Z subsets and sample subsets; pgen dosages against a numpy reference | network (downloads plink2), `pip install Pgenlib` |
| `make test-r` | `read_bcor` against the text output at all compression levels, `NA`, r = 1 and r = -1 | R |
| `make test-orig` | comparison with LDstore v2.0 (downloaded at run time) | network |
| `make asan` | the self-contained tests under AddressSanitizer and UBSan | |

`tests/golden.sh OLD_BIN NEW_BIN` checks that two builds give byte-identical output; it is the
regression test between releases.

## Compatibility with LDstore

fastLD accepts LDstore v2 master files and its LD-related command-line flags, and writes the same
text layout and BCOR v1.1 files, so it can replace `ldstore_v2.0_x86_64 --in-files master
--write-text|--write-bcor --read-only-bgen` without changes. It is an independent implementation
written from the public file-format descriptions and contains no LDstore code. Because fastLD is
exact, a small share of 6-digit values (under 1% in our tests) differs from LDstore's in the last
digit. `.bdose` files, `--bcor-to-text` and the other LDstore functions are not provided.

LDstore and the BCOR format are described in: Benner C. et al. (2017) Prospects of fine-mapping
trait-associated genomic regions by using summary statistics from genome-wide association studies.
*Am J Hum Genet* 101:539-551.

## Limitations

* x86-64 Linux only.
* Biallelic, diploid variants only. BGEN layout 1 (v1.1) and compressed `.pvar.zst` files are not
  supported.
* `--memory` applies to the exact integer paths. It keeps the matrix in a temporary file in the
  output directory, which is removed automatically.

## Tuning

Normally not needed. `FASTLD_KERNEL=avx2|scalar` forces a slower kernel, `FASTLD_GZIP_LEVEL=1..12`
sets the gzip level (default 6), `FASTLD_PROFILE=1` prints phase timings and memory use to stderr.

## Licence

MIT, see `LICENSE`. The PLINK 2 reader in `src/pgenlib/` is from plink-ng and is LGPL-3.0-or-later.
Libraries linked into the release binary are listed in `THIRD_PARTY.md`.
