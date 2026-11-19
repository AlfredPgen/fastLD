#!/usr/bin/env python3
"""Benchmark original LDstore vs fastLD with IDENTICAL command lines.
usage: bench.py ORIG_BIN FAST_BIN MASTER [reps] [threads,..] [out.json]
Runs: --write-text and --write-bcor (medium), each at every thread count."""
import json, os, subprocess, sys, time, numpy as np
orig, fast, master = sys.argv[1:4]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
threads = [int(t) for t in (sys.argv[5] if len(sys.argv) > 5 else "1,4").split(",")]
out = sys.argv[6] if len(sys.argv) > 6 else master + ".bench.json"

def run(cmd):
    t = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, status, ru = os.wait4(p.pid, 0)
    dt = time.perf_counter() - t
    err = p.stderr.read().decode(errors="replace")
    if status != 0 or "Error" in err: raise SystemExit(f"FAILED: {' '.join(cmd)}\n{err[-1500:]}")
    return dt, ru.ru_maxrss / 1024

res = []
for mode in ["--write-text", "--write-bcor"]:
    for t in threads:
        row = {"mode": mode, "threads": t}
        for name, b in [("ldstore", orig), ("fast", fast)]:
            ts, mem = [], []
            for _ in range(reps):
                dt, m = run([b, "--in-files", master, mode, "--read-only-bgen", "--n-threads", str(t)])
                ts.append(dt); mem.append(m)
            row[name] = dict(mean=float(np.mean(ts)), sd=float(np.std(ts, ddof=1)) if reps > 1 else 0.0, peak_mb=max(mem), runs=ts)
        row["speedup"] = row["ldstore"]["mean"] / row["fast"]["mean"]
        res.append(row)
        print(f"{mode:13s} threads={t}:  LDstore {row['ldstore']['mean']:8.2f} ± {row['ldstore']['sd']:.2f} s ({row['ldstore']['peak_mb']:.0f} MB)"
              f"   fastLD {row['fast']['mean']:7.2f} ± {row['fast']['sd']:.2f} s ({row['fast']['peak_mb']:.0f} MB)   speedup {row['speedup']:.1f}x", flush=True)
json.dump(res, open(out, "w"), indent=1)
