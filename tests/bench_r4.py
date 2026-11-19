#!/usr/bin/env python3
"""Benchmark protocol R4: one thread, wall time, N runs per binary, binaries alternated in the
same session. Records wall time and the three phases (parse, correlations, write) parsed from
the program output, peak RSS (wait4), CPU model; prints mean +- sd.

usage: bench_r4.py OUT.json MASTER NAME=BIN [NAME=BIN ...] [--runs 3] [--mode text|bcor]
                   [--threads 1] [--env NAME:VAR=VAL ...] [--extra 'NAME:--flag value' ...]
"""
import argparse, json, os, re, statistics as S, subprocess, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("master"); ap.add_argument("bins", nargs="+")
ap.add_argument("--runs", type=int, default=3); ap.add_argument("--mode", default="text")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--env", action="append", default=[]); ap.add_argument("--extra", action="append", default=[])
a = ap.parse_args()
bins = dict(b.split("=", 1) for b in a.bins)
envs, extra = {}, {}
for e in a.env:
    n, kv = e.split(":", 1); k, v = kv.split("=", 1); envs.setdefault(n, {})[k] = v
for e in a.extra:
    n, f = e.split(":", 1); extra[n] = f.split()
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
mode = ["--write-text"] if a.mode == "text" else ["--write-bcor", "--compression", "medium"]
pat = {"parse": r"Parsing dosages\s*: done! \(([\d.]+) s\)",
       "corr": r"Computing correlations\s*: done! \(([\d.]+) s\)",
       "write": r"Writing correlations\s*: done! \(([\d.]+) s\)"}
# untimed warm-up run (page cache), first binary
_wb = next(iter(bins.values()))
subprocess.run([_wb, "--in-files", a.master, *mode, "--n-threads", str(a.threads)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
res = {n: {"wall": [], "parse": [], "corr": [], "write": [], "rss_mb": [], "path": None, "cmd": None} for n in bins}
for r in range(a.runs):
    for n, b in bins.items():
        cmd = [b, "--in-files", a.master, *mode, "--read-only-bgen", "--n-threads", str(a.threads), *extra.get(n, [])]
        env = dict(os.environ, **envs.get(n, {}))
        with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
            t = time.perf_counter()
            p = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env)
            _, status, ru = os.wait4(p.pid, 0)
            wall = time.perf_counter() - t
            fo.seek(0); fe.seek(0); out = fo.read().decode(); err = fe.read().decode()
        if status != 0: raise SystemExit(f"{n} failed ({status}): {err[-2000:]}")
        R = res[n]; R["cmd"] = " ".join(cmd); R["env"] = envs.get(n, {})
        R["wall"].append(wall); R["rss_mb"].append(ru.ru_maxrss / 1024)
        for k, rx in pat.items():
            m = re.search(rx, out); R[k].append(float(m.group(1)) if m else None)
        m = re.search(r"Correlation path\s*: (.*)", out); R["path"] = m.group(1).strip() if m else None
        print(f"run {r+1} {n:12s} wall {wall:8.2f} s  parse {R['parse'][-1]}  corr {R['corr'][-1]}  write {R['write'][-1]}"
              f"  rss {R['rss_mb'][-1]:.0f} MB  [{R['path']}]", flush=True)

def ms(x):
    x = [v for v in x if v is not None]
    if not x: return None
    return {"mean": S.mean(x), "sd": S.stdev(x) if len(x) > 1 else 0.0}
summary = {"cpu": cpu, "master": a.master, "threads": a.threads, "mode": a.mode, "runs": a.runs,
           "date": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}}
for n, R in res.items():
    summary["results"][n] = {k: ms(R[k]) for k in ("wall", "parse", "corr", "write")}
    summary["results"][n].update(rss_mb_max=max(R["rss_mb"]), path=R["path"], raw=R)
json.dump(summary, open(a.out, "w"), indent=1)
print(f"\nCPU: {cpu}   threads={a.threads}  mode={a.mode}  runs={a.runs}")
for n, s in summary["results"].items():
    f = lambda d: f"{d['mean']:.2f} ± {d['sd']:.2f}" if d else "-"
    print(f"{n:12s} wall {f(s['wall'])}  parse {f(s['parse'])}  corr {f(s['corr'])}  write {f(s['write'])}"
          f"  peakRSS {s['rss_mb_max']:.0f} MB  [{s['path']}]")
