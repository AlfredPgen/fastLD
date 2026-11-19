#!/usr/bin/env python3
"""Exact LD reference from the generator's dosage numerators (PREFIX.k.npy, PREFIX.miss.npy) and
the share of text tokens in each .ld file that equal the correctly rounded exact value.
Only numpy is needed. Use on data sets with at most 300 SNPs.

Definition (what fastLD and LDstore compute): per SNP i, mean and sum of squares over its
observed samples; a missing value contributes zero after centring. With integer numerators k:
  n_i, sx_i, sxx_i over observed samples of i; C_ij = sum k_i k_j over samples observed in both;
  A_ij = sum k_i over samples observed in both; n_ij = #samples observed in both;
  num = n_i n_j C_ij - n_i sx_j A_ij - n_j sx_i A_ji + sx_i sx_j n_ij,  V_i = n_i sxx_i - sx_i^2,
  r = num / sqrt(n_i n_j V_i V_j)       (exact integers, 50-digit decimal square root)
Missing rate >= --sample-miss -> 'nan' row/column; V_i == 0 -> '-nan' (nan wins); diagonal '1'.

usage: exact_check.py PREFIX A.ld [B.ld ...] [--sample-miss 0.1] [--precision 6] [--snps i,j,...]
  --snps: the .ld files cover only these 0-based SNP indices, in this order (Z-file subsets)
exit status 1 unless every file is 100% correct.
"""
import argparse, decimal, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("prefix"); ap.add_argument("files", nargs="+")
ap.add_argument("--sample-miss", type=float, default=0.1)
ap.add_argument("--precision", type=int, default=6)
ap.add_argument("--snps")
a = ap.parse_args()
decimal.getcontext().prec = 50
D = decimal.Decimal

K = np.load(a.prefix + ".k.npy").astype(np.int64)
Mi = np.load(a.prefix + ".miss.npy").astype(bool)
if a.snps:
    idx = [int(x) for x in a.snps.split(",")]
    K, Mi = K[:, idx], Mi[:, idx]
N, M = K.shape
if M > 300: sys.exit("exact_check.py: at most 300 SNPs")
O = (~Mi).astype(np.int64)
K = K * O                                        # missing -> 0
n = O.sum(0); sx = K.sum(0); sxx = (K * K).sum(0)
C = K.T @ K                                      # sums over samples observed in both (missing are 0)
A = K.T @ O                                      # A[i, j] = sum of k_i over samples where j observed (i observed implied by k_i=0)
nij = O.T @ O
V = [int(n[i]) * int(sxx[i]) - int(sx[i]) ** 2 for i in range(M)]
na = [N == 0 or (float(N - int(n[i])) / float(N)) >= a.sample_miss for i in range(M)]

ref = np.empty((M, M), dtype=object)
for i in range(M):
    ref[i, i] = "1"
    for j in range(i + 1, M):
        if na[i] or na[j]: t = "nan"
        elif V[i] == 0 or V[j] == 0: t = "-nan"
        else:
            ni, nj = int(n[i]), int(n[j])
            num = ni * nj * int(C[i, j]) - ni * int(sx[j]) * int(A[i, j]) - nj * int(sx[i]) * int(A[j, i]) + int(sx[i]) * int(sx[j]) * int(nij[i, j])
            r = D(num) / (D(ni) * D(nj) * D(V[i]) * D(V[j])).sqrt()
            t = float(format(r, f".{a.precision}g"))   # correctly rounded decimal -> its value
        ref[i, j] = ref[j, i] = t

ok_all = True
for f in a.files:
    tok = open(f).read().split()
    if len(tok) != M * M:
        print(f"[exact] {f}: FAIL, {len(tok)} tokens, expected {M*M}"); ok_all = False; continue
    good = 0
    bad_examples = []
    for (i, j), t in zip(np.ndindex(M, M), tok):
        r = ref[i, j]
        if isinstance(r, str): ok = (t == r)
        else: ok = t not in ("nan", "-nan", "1") and float(t) == r or (t == "1" and r == 1.0)
        good += ok
        if not ok and len(bad_examples) < 3: bad_examples.append((i, j, t, r))
    share = good / (M * M)
    print(f"[exact] {f}: {share*100:.4f}% of {M*M} tokens equal the correctly rounded exact value"
          + ("" if share == 1 else f"  e.g. {bad_examples}"))
    ok_all &= share == 1
sys.exit(0 if ok_all else 1)
