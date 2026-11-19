#!/usr/bin/env python3
"""Synthetic BGEN v1.2 (layout 2, zlib) + bgenix .bgi + .sample + .z + LDstore master file.

usage: make_bgen.py PREFIX N_SAMPLES N_SNPS [seed]

environment:
  PROFILE=dense     (default) MAF ~ U(0.02, 0.5); probabilities = (1-eps)*onehot + eps*Dirichlet(1,1,1),
                    eps ~ Beta(1,9) per SNP  (the original generator; almost no exact zeros)
  PROFILE=imputed   MAF density ~ 1/MAF on [0.001, 0.5]; probabilities = (1-e)*onehot(g) + e*HWE(maf),
                    e ~ Beta(1,19) per sample and SNP
  PROFILE=hardcall  MAF ~ U(0.02, 0.5); probabilities exactly one-hot
  PROFILE=hardgcd   as hardcall, but heterozygotes get probabilities (m, den-2m, m)/den with random
                    integer m: numerators are still exactly {0, den, 2den}, probabilities are not 0/1
  BITS=8|16         probability bit depth (default 8)
  MISSALL=rate      every SNP: each sample missing with probability `rate`
  MISS=rate         threshold tests: every 7th SNP 15% missing, the others `rate`
  MISSFIX=snp:n,... exact number of missing samples for given SNPs (0-based)
  MONO=snp,...      monomorphic SNPs (all hom allele1)
  FLIP=frac         this fraction of SNPs has allele2 as the MAJOR allele (genotypes and probabilities mirrored)
  DUPPREV=j,...     SNP j is an exact copy of SNP j-1 (r = 1)
  NEGPREV=j,...     SNP j mirrors SNP j-1 (P(AA) <-> P(BB), r = -1)
  NPY=1             force writing PREFIX.k.npy / PREFIX.miss.npy (default: only when M <= 300)

Genotypes: two haplotypes per sample from an AR(1) latent Gaussian along the chromosome
(rho = 0.95), thresholded at the MAF quantile; g = number of allele2 copies.
PREFIX.k.npy: int64 (N, M) dosage numerators k = 2*(2^B-1) - 2*P(AA) - P(AB) (0 where missing);
PREFIX.miss.npy: bool (N, M) missing mask.  Streaming in SNP chunks (memory ~ 64*N).
"""
import os, sys, struct, zlib, sqlite3, numpy as np
from scipy.stats import norm

pre, N, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
PROFILE = os.environ.get("PROFILE", "dense")
BITS = int(os.environ.get("BITS", "8")); DEN = 2**BITS - 1
MISS = float(os.environ.get("MISS", "0"))
MISSALL = float(os.environ.get("MISSALL", "0"))
MISSFIX = {int(a): int(b) for a, b in (x.split(":") for x in os.environ.get("MISSFIX", "").split(",") if x)}
MONO = {int(x) for x in os.environ.get("MONO", "").split(",") if x}
DUPPREV = {int(x) for x in os.environ.get("DUPPREV", "").split(",") if x}
NEGPREV = {int(x) for x in os.environ.get("NEGPREV", "").split(",") if x}
SAVE_NPY = M <= 300 or os.environ.get("NPY") == "1"
assert PROFILE in ("dense", "imputed", "hardcall", "hardgcd"), PROFILE

rho = 0.95
if PROFILE == "imputed":
    maf = np.exp(rng.uniform(np.log(0.001), np.log(0.5), M))   # density proportional to 1/MAF
else:
    maf = rng.uniform(0.02, 0.5, M)
thr = norm.ppf(maf)
FLIP = float(os.environ.get("FLIP", "0"))
flipmask = rng.random(M) < FLIP if FLIP > 0 else np.zeros(M, bool)

CH = 64
state = [rng.standard_normal(N), rng.standard_normal(N)]
def next_chunk(j0, j1):
    m = j1 - j0; G = np.zeros((m, N), np.int8)
    for h in range(2):
        z = state[h]
        for t in range(m):
            if j0 + t > 0: z = rho * z + rng.standard_normal(N) * np.sqrt(1 - rho**2)
            G[t] += z < thr[j0 + t]
        state[h] = z
    P = np.zeros((m, N, 3)); np.put_along_axis(P, G[..., None].astype(int), 1.0, axis=2)
    if PROFILE == "dense":
        eps = rng.beta(1, 9, (m, 1, 1))
        P = (1 - eps) * P + eps * rng.dirichlet([1, 1, 1], (m, N))
    elif PROFILE == "imputed":
        p = maf[j0:j1, None]
        hwe = np.stack([(1 - p) ** 2, 2 * p * (1 - p), p ** 2], -1)          # (m, 1, 3)
        e = rng.beta(1, 19, (m, N, 1))
        P = (1 - e) * P + e * hwe
    if PROFILE == "hardgcd":
        het = G == 1
        mm = rng.integers(0, int(0.3 * DEN), size=G.shape) / DEN
        P[het] = np.stack([mm[het], 1 - 2 * mm[het], mm[het]], -1)
    fm = flipmask[j0:j1]
    if fm.any(): P[fm] = P[fm][..., ::-1]          # mirror: allele2 becomes the major allele
    return P

def quant(p):  # BGEN rounding: quantized probabilities sum to 2^B-1
    x = np.round(p * DEN, 9)   # exact multiples of 1/DEN stay exact
    f = np.floor(x); r = (DEN - f.sum(-1)).astype(int)
    order = np.argsort(-(x - f), axis=-1)
    rank = np.argsort(order, axis=-1); f += (rank < r[..., None])
    return f.astype(np.uint8 if BITS == 8 else np.uint16)

samples = [f"S{i+1}" for i in range(N)]
sblock = b"".join(struct.pack("<H", len(s)) + s.encode() for s in samples)
sblock = struct.pack("<II", len(sblock) + 8, N) + sblock
hdr = struct.pack("<I", 20) + struct.pack("<II", M, N) + b"bgen" + struct.pack("<I", 1 | (2 << 2) | (1 << 31))
off = len(hdr) + len(sblock)
if os.path.exists(pre + ".bgen.bgi"): os.remove(pre + ".bgen.bgi")
db = sqlite3.connect(pre + ".bgen.bgi")
db.execute("CREATE TABLE Metadata (filename TEXT, file_size INT, last_write_time INT, first_1000_bytes BLOB, index_creation_time INT)")
db.execute("CREATE TABLE Variant (chromosome TEXT, position INT, rsid TEXT, number_of_alleles INT, allele1 TEXT, allele2 TEXT, file_start_position INT, size_in_bytes INT, PRIMARY KEY (chromosome, position, rsid, allele1, allele2, file_start_position)) WITHOUT ROWID")
pos = np.sort(rng.choice(np.arange(1_000_000, 1_000_000 + 50 * M), M, replace=False))
rows, zl = [], ["rsid chromosome position allele1 allele2"]
if SAVE_NPY:
    Kall = np.zeros((N, M), np.int64); Mall = np.zeros((N, M), bool)
with open(pre + ".bgen", "wb") as f:
    f.write(struct.pack("<I", off)); f.write(hdr); f.write(sblock)
    for j in range(M):
        if j % CH == 0: Q = quant(next_chunk(j, min(M, j + CH))); j0 = j
        rs = f"rs{j+1}"; a1, a2 = "A", "G"
        start = f.tell()
        v = struct.pack("<H", len(rs)) + rs.encode() + struct.pack("<H", len(rs)) + rs.encode()
        v += struct.pack("<H", 2) + b"01" + struct.pack("<IH", int(pos[j]), 2)
        v += struct.pack("<I", 1) + a1.encode() + struct.pack("<I", 1) + a2.encode()
        ploid = np.full(N, 2, np.uint8); qq = Q[j - j0][:, :2].copy()      # P(AA), P(AG)
        mm = np.zeros(N, bool)
        if MISS > 0: mm |= rng.random(N) < (MISS if j % 7 else 0.15)
        if MISSALL > 0: mm |= rng.random(N) < MISSALL
        if j in MISSFIX: mm[rng.choice(N, MISSFIX[j], replace=False)] = True
        if j in MONO: qq[:] = (DEN, 0)
        if j in DUPPREV: qq = prev_q.copy(); mm = prev_m.copy()
        if j in NEGPREV: qq = np.stack([DEN - prev_q[:, 0] - prev_q[:, 1], prev_q[:, 1]], -1).astype(qq.dtype); mm = prev_m.copy()
        ploid[mm] |= 0x80; qq[mm] = 0
        prev_q, prev_m = qq.copy(), mm.copy()
        if SAVE_NPY:
            k = 2 * DEN - 2 * qq[:, 0].astype(np.int64) - qq[:, 1].astype(np.int64)
            k[mm] = 0; Kall[:, j] = k; Mall[:, j] = mm
        body = struct.pack("<IHBB", N, 2, 2, 2) + ploid.tobytes() + struct.pack("<BB", 0, BITS) + qq.tobytes()
        c = zlib.compress(body, 6)
        v += struct.pack("<II", len(c) + 4, len(body)) + c
        f.write(v)
        rows.append(("01", int(pos[j]), rs, 2, a1, a2, start, len(v)))
        zl.append(f"{rs} 01 {pos[j]} {a1} {a2}")
db.executemany("INSERT INTO Variant VALUES (?,?,?,?,?,?,?,?)", rows)
first = open(pre + ".bgen", "rb").read(1000)
db.execute("INSERT INTO Metadata VALUES (?,?,?,?,?)", (os.path.basename(pre) + ".bgen", os.path.getsize(pre + ".bgen"), int(os.path.getmtime(pre + ".bgen")), first, 0))
db.commit(); db.close()
if SAVE_NPY:
    np.save(pre + ".k.npy", Kall); np.save(pre + ".miss.npy", Mall)
open(pre + ".z", "w").write("\n".join(zl) + "\n")
open(pre + ".sample", "w").write("ID_1 ID_2 missing\n0 0 0\n" + "".join(f"{s} {s} 0\n" for s in samples))
open(pre + ".master", "w").write("z;bgen;bgi;sample;bdose;bcor;ld;n_samples\n"
    f"{pre}.z;{pre}.bgen;{pre}.bgen.bgi;{pre}.sample;{pre}.bdose;{pre}.bcor;{pre}.ld;{N}\n")
print(f"wrote {pre}.bgen  N={N} M={M} profile={PROFILE} bits={BITS}")
