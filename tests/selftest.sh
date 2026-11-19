#!/usr/bin/env bash
# Self-contained test tier ("make test"): needs python3 + numpy + scipy.
#   1. golden test: byte-identity with a reference binary (an earlier release) on the exact
#      integer path; skipped when no reference binary is given
#   2. exact reference: 100% correctly rounded text values on small sets
#   3. gzip text output
#   4. error handling (truncated / corrupt input, bad master files)
# usage: tests/selftest.sh REFERENCE_BIN|- NEW_BIN
set -uo pipefail
BASE=$(readlink -f "$1"); NEW=$(readlink -f "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORKDIR:-/tmp/fastLD-selftest}; mkdir -p "$W"
GEN="python3 $HERE/make_bgen.py"; EX="python3 $HERE/exact_check.py"
fails=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fails=$((fails+1)); }

echo "== golden (byte identity with a reference binary)"
if [ ! -x "$BASE" ]; then echo "SKIP  no reference binary (pass one as first argument or BASELINE=... to make)"
elif WORKDIR=$W/golden bash "$HERE/golden.sh" "$BASE" "$NEW" > $W/golden.log 2>&1; then ok "$(tail -1 $W/golden.log)"; else cat $W/golden.log; bad golden; fi

echo "== exact reference (every text value correctly rounded)"
cd "$W"
exact_case() { # name N M seed extra-args... -- ENV...
  local n=$1 N=$2 M=$3 s=$4; shift 4
  local args=() envs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do args+=("$1"); shift; done
  [ $# -gt 0 ] && shift; envs=("$@")
  [ -f $n.master ] || env "${envs[@]}" $GEN $W/$n $N $M $s >/dev/null
  for th in 1 4; do
    "$NEW" --in-files $n.master --write-text --n-threads $th "${args[@]}" >/dev/null 2>$W/err.txt || { bad "$n threads=$th: $(cat $W/err.txt)"; continue; }
    local sm=0.1; for ((q=0; q<${#args[@]}; q++)); do [ "${args[q]}" = "--sample-miss" ] && sm=${args[q+1]}; done
    local pr=6; for ((q=0; q<${#args[@]}; q++)); do [ "${args[q]}" = "--precision" ] && pr=${args[q+1]}; done
    if out=$($EX $W/$n $W/$n.ld --sample-miss $sm --precision $pr); then ok "$n threads=$th ${args[*]}: ${out#*: }"; else echo "$out"; bad "$n threads=$th"; fi
  done
}
exact_case ex_small 1000 200 1 --
exact_case ex_miss 800 120 5 -- MISS=0.02
exact_case ex_miss05 800 120 5 --sample-miss 0.05 -- MISS=0.02
exact_case ex_edge 1000 10 7 -- MISSFIX=1:99,2:100,3:101,4:1,5:999 MONO=6
exact_case ex_b16 2000 300 11 -- BITS=16
exact_case ex_imputed 2000 300 12 -- PROFILE=imputed
exact_case ex_impflip 1999 300 16 -- PROFILE=imputed FLIP=0.5
exact_case ex_hard 2000 300 13 -- PROFILE=hardcall
exact_case ex_hardm 2000 300 13 -- PROFILE=hardcall MISSALL=0.005
exact_case ex_hard16 2000 300 13 -- PROFILE=hardcall BITS=16
exact_case ex_dmiss 2000 300 11 -- MISSALL=0.002
exact_case ex_imiss 2000 300 12 -- PROFILE=imputed MISSALL=0.002
exact_case ex_prec 1000 200 1 --precision 10 --
exact_case ex_hardgcd 2000 300 17 -- PROFILE=hardgcd FLIP=0.3
echo "-- non-default kernels"
for k in avx2 scalar; do
  for n in ex_hardm ex_impflip ex_dmiss ex_hard16; do
    FASTLD_KERNEL=$k "$NEW" --in-files $n.master --write-text --n-threads 2 >/dev/null 2>$W/err.txt || { bad "$n kernel=$k"; continue; }
    if out=$($EX $W/$n $W/$n.ld); then ok "$n kernel=$k: ${out#*: }"; else echo "$out"; bad "$n kernel=$k"; fi
  done
done

echo "== gzip text output (ld file name ending in .gz)"
sed -e "s#;$W/ex_small.ld;#;$W/ex_small_gz.ld.gz;#" ex_small.master > ex_small_gz.master
for th in 1 4; do
  rm -f ex_small.ld ex_small_gz.ld.gz
  "$NEW" --in-files ex_small.master --write-text --n-threads $th >/dev/null 2>$W/err.txt
  "$NEW" --in-files ex_small_gz.master --write-text --n-threads $th >/dev/null 2>>$W/err.txt
  if gzip -t ex_small_gz.ld.gz 2>/dev/null && zcat ex_small_gz.ld.gz | cmp -s - ex_small.ld &&
     python3 -c "import gzip,sys; sys.exit(gzip.open('$W/ex_small_gz.ld.gz','rb').read() != open('$W/ex_small.ld','rb').read())"; then
    ok "gzip threads=$th: identical after decompression ($(stat -c %s ex_small.ld) -> $(stat -c %s ex_small_gz.ld.gz) bytes)"
  else bad "gzip threads=$th: $(cat $W/err.txt)"; fi
done

echo "== error handling"
[ -f ex_small.master ] || $GEN $W/ex_small 1000 200 1 >/dev/null
head -c 100000 ex_small.bgen > trunc.bgen
sed -e "s#$W/ex_small.bgen;#$W/trunc.bgen;#" -e "s#$W/ex_small.bgen.bgi#$W/ex_small.bgen.bgi#" ex_small.master > trunc.master
if "$NEW" --in-files trunc.master --write-text > /dev/null 2> err.txt; then bad "truncated BGEN accepted"
elif grep -q "truncated\|outside BGEN file" err.txt; then ok "truncated BGEN: $(head -1 err.txt)"; else bad "truncated BGEN: unclear error: $(cat err.txt)"; fi
head -c 30 ex_small.bgen > trunc2.bgen; sed -e "s#$W/ex_small.bgen;#$W/trunc2.bgen;#" ex_small.master > trunc2.master
if "$NEW" --in-files trunc2.master --write-text > /dev/null 2> err.txt; then bad "truncated header accepted"; else ok "truncated header: $(head -1 err.txt)"; fi
python3 - "$W" <<'E'
import sys; w = sys.argv[1]; b = bytearray(open(f"{w}/ex_small.bgen", "rb").read()); b[4] = 0xFF; b[5] = 0xFF
open(f"{w}/badlh.bgen", "wb").write(b)
E
sed -e "s#$W/ex_small.bgen;#$W/badlh.bgen;#" ex_small.master > badlh.master
if "$NEW" --in-files badlh.master --write-text > /dev/null 2> err.txt; then bad "bad header length accepted"; else ok "bad header length: $(head -1 err.txt)"; fi
if "$NEW" --in-files ex_small.master --write-text --write-bdose > /dev/null 2> err.txt && grep -q "ignored" err.txt; then ok "ignored flag warns: $(head -1 err.txt)"; else bad "--write-bdose warning"; fi
printf 'z;bgen;bed;ld;n_samples\na.z;a.bgen;a.bed;a.ld;10\n' > mixed.master
if "$NEW" --in-files mixed.master --write-text > /dev/null 2> err.txt && false; then :; elif grep -q "mixes" err.txt; then ok "mixed formats rejected"; else bad "mixed formats: $(cat err.txt)"; fi
if "$NEW" --in-files ex_small.master --write-text --format pgen > /dev/null 2> err.txt; then bad "--format mismatch accepted"; else ok "--format mismatch rejected"; fi
"$NEW" --in-files ex_small.master --stats > stats.txt 2>&1 && grep -q "Zero share" stats.txt && ok "--stats" || bad "--stats"

echo
if [ $fails -eq 0 ]; then echo "SELFTEST: all passed"; else echo "SELFTEST: $fails failed"; fi
[ $fails -eq 0 ]
