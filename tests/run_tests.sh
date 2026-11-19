#!/usr/bin/env bash
# Tier 2 ("make test-orig"): fastLD vs. the original LDstore v2.0 on synthetic BGEN data
# (structure, NA pattern, BCOR header/meta identical; values identical up to LDstore's own rounding).
# usage: tests/run_tests.sh [path/to/original/ldstore]   (downloaded automatically if absent)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
FAST=$ROOT/fastLD
W=${WORKDIR:-/tmp/fastLD-tests}; mkdir -p "$W"; cd "$W"
ORIG=${1:-}
if [ -z "$ORIG" ]; then
  [ -x ldstore_v2.0_x86_64/ldstore_v2.0_x86_64 ] || { curl -sL http://www.christianbenner.com/ldstore_v2.0_x86_64.tgz | tar xz; }
  ORIG=$W/ldstore_v2.0_x86_64/ldstore_v2.0_x86_64
fi
GEN="python3 $HERE/make_bgen.py"; CMP="python3 $HERE/compare.py"

mk() { # name N M seed [ENV...]
  local n=$1 N=$2 M=$3 s=$4; shift 4
  [ -f $n.bgen ] || env "$@" $GEN $W/$n $N $M $s >/dev/null
  for tool in orig fast; do
    sed -e "s#$W/$n.ld#$W/$n.$tool.ld#" -e "s#$W/$n.bcor#$W/$n.$tool.bcor#" $n.master > $n.$tool.master
  done
}
run() { # name compression extra-args...
  local n=$1 c=$2; shift 2
  "$ORIG" --in-files $n.orig.master --write-text --read-only-bgen "$@" >/dev/null 2>&1
  "$FAST" --in-files $n.fast.master --write-text --read-only-bgen "$@" >/dev/null
  echo -n "[$n text $*] "; $CMP text $n.orig.ld $n.fast.ld
  "$ORIG" --in-files $n.orig.master --write-bcor --read-only-bgen --compression $c "$@" >/dev/null 2>&1
  "$FAST" --in-files $n.fast.master --write-bcor --read-only-bgen --compression $c "$@" >/dev/null
  echo -n "[$n bcor $c $*] "; $CMP bcor $n.orig.bcor $n.fast.bcor
}

mk small 1000 200 1
for c in ultra-low low medium high; do run small $c; done
mk miss 800 120 5 MISS=0.02
run miss medium; run miss high --sample-miss 0.05
mk edge 1000 10 7 MISSFIX=1:99,2:100,3:101,4:1,5:999 MONO=6
run edge medium
# Z-file subset in non-BGEN order
mk sub 1000 200 1
[ -f sub_full.z ] || cp sub.z sub_full.z
printf "rsid chromosome position allele1 allele2\n" > sub.z; for r in 50 2 9 120 7 199; do grep "^rs$r " sub_full.z >> sub.z; done
run sub medium
# incl file (sample subset)
mk incl 1000 200 1
python3 -c "import random; random.seed(1); open('$W/s.incl','w').write(''.join(f'S{i}\n' for i in sorted(random.sample(range(1,1001),300))))"
for tool in orig fast; do
  python3 - $W/incl.$tool.master <<'E'
import sys; f=sys.argv[1]; L=open(f).read().split("\n")
h=L[0].split(";"); v=L[1].split(";"); h.insert(4,"incl"); v.insert(4,f.rsplit("/",1)[0]+"/s.incl"); v[-1]="300"
open(f,"w").write(";".join(h)+"\n"+";".join(v)+"\n")
E
done
run incl medium
# 16-bit probabilities
mk b16 2000 300 11 BITS=16
run b16 medium
# multithreaded run must give identical output
"$FAST" --in-files miss.fast.master --write-text --n-threads 4 >/dev/null; cp miss.fast.ld miss.t4.ld
"$FAST" --in-files miss.fast.master --write-text --n-threads 1 >/dev/null; cmp miss.t4.ld miss.fast.ld && echo "[threads] 1 vs 4 threads: byte-identical"
# exact-integer path vs floating-point path on identical input
"$FAST" --in-files small.fast.master --write-text >/dev/null; cp small.fast.ld small.int.ld
"$FAST" --in-files small.fast.master --write-text --force-double >/dev/null
echo -n "[int vs double path] "; $CMP text small.int.ld small.fast.ld
# both paths vs EXACT correlations (big-integer arithmetic on the dosage numerators); the original
# LDstore is shown for comparison (its own rounding error makes it < 100%)
python3 $HERE/exact_check.py $W/small small.int.ld small.fast.ld
python3 $HERE/exact_check.py $W/small small.orig.ld || true
echo "ALL TESTS RAN"
