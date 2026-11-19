#!/usr/bin/env python3
"""Compare fastLD output with original LDstore output.
usage: compare.py text A.ld B.ld | compare.py bcor A.bcor B.bcor
Prints identity statistics and max |r difference|; exit 1 on structural mismatch."""
import sys, struct, numpy as np

def text(a, b):
    ta = open(a).read(); tb = open(b).read()
    la, lb = ta.split("\n"), tb.split("\n")
    if len(la) != len(lb): print("FAIL different #lines"); sys.exit(1)
    xa, xb = ta.split(), tb.split()
    if len(xa) != len(xb): print("FAIL different #tokens"); sys.exit(1)
    xa, xb = np.array(xa), np.array(xb)
    same = xa == xb
    na = np.isin(xa, ["nan", "-nan"]); nb = np.isin(xb, ["nan", "-nan"])
    if (na != nb).any(): print("FAIL nan pattern differs"); sys.exit(1)
    fa = xa[~na].astype(float); fb = xb[~nb].astype(float)
    d = np.abs(fa - fb).max() if fa.size else 0.0
    print(f"text  tokens={xa.size}  identical={same.mean()*100:.5f}%  differing={int((~same).sum())}  "
          f"byte-identical-file={ta == tb}  max|dr|={d:.2e}")

def bcor(a, b):
    A, B = open(a, "rb").read(), open(b, "rb").read()
    ha, hb = struct.unpack_from("<7sQIIBQ", A), struct.unpack_from("<7sQIIBQ", B)
    if ha != hb: print("FAIL header", ha, hb); sys.exit(1)
    off = ha[5]
    if A[:off] != B[:off]: print("FAIL meta block differs"); sys.exit(1)
    t = {0: np.uint16, 1: np.uint32, 2: np.uint64, 3: np.uint8}[ha[4]]
    va, vb = np.frombuffer(A, t, offset=off), np.frombuffer(B, t, offset=off)
    if va.size != vb.size: print("FAIL size"); sys.exit(1)
    same = va == vb
    nbytes = np.dtype(t).itemsize; NA = 0xD0 << (8 * (nbytes - 1))
    if ((va == NA) != (vb == NA)).any(): print("FAIL NA pattern differs"); sys.exit(1)
    m = va != NA
    diff = np.abs(va[m].astype(np.float64) - vb[m].astype(np.float64)) / 2.0 ** (8 * nbytes - 2) if m.any() else np.zeros(1)
    units = np.abs(va[m].astype(np.int64 if nbytes < 8 else np.float64) - vb[m].astype(np.int64 if nbytes < 8 else np.float64))
    print(f"bcor  values={va.size}  identical={same.mean()*100:.5f}%  header+meta identical=True  "
          f"byte-identical-file={A == B}  max|dr|={diff.max():.2e}  max diff in units={units.max() if units.size else 0:.0f}")

{"text": text, "bcor": bcor}[sys.argv[1]](sys.argv[2], sys.argv[3])
