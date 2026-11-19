#!/usr/bin/env bash
# Golden test (exactness contract R2): NEW_BIN must produce byte-identical text and BCOR files
# (all four compression levels) to BASELINE_BIN, for 1 and 4 threads and every kernel available
# on this CPU, on 8-bit BGEN data without missing values (the exact integer path).
# usage: tests/golden.sh BASELINE_BIN NEW_BIN        (WORKDIR=... to choose the scratch dir)
set -euo pipefail
BASE=$(readlink -f "$1"); NEW=$(readlink -f "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORKDIR:-/tmp/fastLD-golden}; mkdir -p "$W"; cd "$W"
GEN="python3 $HERE/make_bgen.py"

kernels=("")
grep -q avx2 /proc/cpuinfo && kernels+=("avx2")
kernels+=("scalar")

gen() { # name N M seed [ENV...]
  local n=$1 N=$2 M=$3 s=$4; shift 4
  [ -f $n.master ] || env "$@" $GEN $W/$n $N $M $s >/dev/null
}

# datasets -------------------------------------------------------------------------------
gen small 1000 200 1
gen sub 1000 200 1
[ -f sub_full.z ] || cp sub.z sub_full.z
{ echo "rsid chromosome position allele1 allele2"; for r in 50 2 9 120 7 199; do grep "^rs$r " sub_full.z; done; } > sub.z
gen incl 1000 200 1
python3 -c "import random; random.seed(1); open('$W/s.incl','w').write(''.join(f'S{i}\n' for i in sorted(random.sample(range(1,1001),300))))"
if ! grep -q incl incl.master; then
  python3 - "$W/incl.master" "$W/s.incl" <<'E'
import sys; f, inc = sys.argv[1:3]; L = open(f).read().split("\n")
h = L[0].split(";"); v = L[1].split(";"); h.insert(4, "incl"); v.insert(4, inc); v[-1] = "300"
open(f, "w").write(";".join(h) + "\n" + ";".join(v) + "\n")
E
fi
gen gD1 2000 400 11 PROFILE=dense
gen gD2 2000 400 12 PROFILE=imputed
gen gD3h 2000 400 13 PROFILE=hardcall
gen gD2f 1999 400 14 PROFILE=imputed FLIP=0.5
gen gD1o 1999 400 15 PROFILE=dense FLIP=0.3
gen gD3g 2000 400 17 PROFILE=hardgcd FLIP=0.3

# runs -------------------------------------------------------------------------------------
pass=0; fail=0
master_for() { # dataset tool -> master with outputs redirected
  local n=$1 t=$2
  sed -e "s#;$W/$n.ld;#;$W/out.$t.ld;#" -e "s#;$W/$n.bcor;#;$W/out.$t.bcor;#" $n.master > $n.$t.master
  echo $n.$t.master
}
for n in small sub incl gD1 gD2 gD3h gD2f gD1o gD3g; do
  mb=$(master_for $n base); mn=$(master_for $n new)
  for th in 1 4; do
    for k in "${kernels[@]}"; do
      for mode in text ultra-low low medium high; do
        if [ $mode = text ]; then args=(--write-text); f=ld; else args=(--write-bcor --compression $mode); f=bcor; fi
        rm -f out.base.$f out.new.$f
        FASTLD_KERNEL=$k "$BASE" --in-files $mb "${args[@]}" --read-only-bgen --n-threads $th >/dev/null 2>&1 || true
        FASTLD_KERNEL=$k "$NEW"  --in-files $mn "${args[@]}" --read-only-bgen --n-threads $th >/dev/null 2>&1 || true
        if [ -s out.base.$f ] && cmp -s out.base.$f out.new.$f; then pass=$((pass+1))
        else fail=$((fail+1)); echo "FAIL: $n threads=$th kernel=${k:-default} $mode"; fi
      done
    done
  done
done
# engine variants of the new binary (sparse fold-back, aggressive sparse threshold) vs baseline
for n in gD2 gD2f; do
  mb=$(master_for $n base); mn=$(master_for $n new)
  for var in FASTLD_SPARSE_MIN=100000 FASTLD_SPARSE=0.5 FASTLD_SPARSE=0; do
    for th in 1 4; do
      for mode in text medium; do
        if [ $mode = text ]; then args=(--write-text); f=ld; else args=(--write-bcor --compression $mode); f=bcor; fi
        rm -f out.base.$f out.new.$f
        "$BASE" --in-files $mb "${args[@]}" --n-threads $th >/dev/null 2>&1 || true
        env $var "$NEW" --in-files $mn "${args[@]}" --n-threads $th >/dev/null 2>&1 || true
        if [ -s out.base.$f ] && cmp -s out.base.$f out.new.$f; then pass=$((pass+1))
        else fail=$((fail+1)); echo "FAIL: $n threads=$th $var $mode"; fi
      done
    done
  done
done
# --memory blocked mode (small forced blocks) vs baseline
for n in gD1o gD2f gD3h; do
  mb=$(master_for $n base); mn=$(master_for $n new)
  for th in 1 4; do
    for mode in text medium; do
      if [ $mode = text ]; then args=(--write-text); f=ld; else args=(--write-bcor --compression $mode); f=bcor; fi
      rm -f out.base.$f out.new.$f
      "$BASE" --in-files $mb "${args[@]}" --n-threads $th >/dev/null 2>&1 || true
      FASTLD_MEM_A=70 FASTLD_MEM_B=23 "$NEW" --in-files $mn "${args[@]}" --n-threads $th --memory 0.000001 >/dev/null 2>&1 || true
      if [ -s out.base.$f ] && cmp -s out.base.$f out.new.$f; then pass=$((pass+1))
      else fail=$((fail+1)); echo "FAIL: $n threads=$th --memory blocked $mode"; fi
    done
  done
done
echo "golden: $pass passed, $fail failed (kernels: default ${kernels[*]:1})"
[ $fail -eq 0 ]
