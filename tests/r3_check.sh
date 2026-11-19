#!/usr/bin/env bash
# Rule R3 for data that take the floating-point path in the baseline and an exact path now:
#  (a) 100% of text values equal the correctly rounded exact value (exact_check.py)
#  (b) share of text tokens identical to the ORIGINAL LDstore >= the baseline binary's share
# usage: tests/r3_check.sh ORIGINAL_LDSTORE BASELINE_BIN NEW_BIN
set -uo pipefail
ORIG=$(readlink -f "$1"); BASE=$(readlink -f "$2"); NEW=$(readlink -f "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORKDIR:-/tmp/fastLD-r3}; mkdir -p "$W"; cd "$W"
GEN="python3 $HERE/make_bgen.py"
fails=0
share() { python3 "$HERE/compare.py" text "$1" "$2" | grep -oP "identical=\K[0-9.]+"; }
check() { # name N M seed ENV...
  local n=$1 N=$2 M=$3 s=$4; shift 4
  [ -f $n.master ] || env "$@" $GEN $W/$n $N $M $s >/dev/null
  for t in orig base new; do sed -e "s#;$W/$n.ld;#;$W/$n.$t.ld;#" $n.master > $n.$t.master; done
  "$ORIG" --in-files $n.orig.master --write-text --read-only-bgen >/dev/null 2>&1
  "$BASE" --in-files $n.base.master --write-text >/dev/null 2>&1
  "$NEW"  --in-files $n.new.master --write-text >/dev/null 2>&1
  local path; path=$("$NEW" --in-files $n.new.master --write-text 2>/dev/null | grep -oP "Correlation path\s*: \K.*")
  local sb sn ex
  sb=$(share $n.orig.ld $n.base.ld); sn=$(share $n.orig.ld $n.new.ld)
  ex=$(python3 "$HERE/exact_check.py" $W/$n $n.new.ld | grep -oP "\] .*?: \K[0-9.]+%")
  local verdict=PASS
  python3 -c "import sys; sys.exit(0 if float('$sn') >= float('$sb') else 1)" || verdict=FAIL
  [ "$ex" = "100.0000%" ] || verdict=FAIL
  [ $verdict = PASS ] || fails=$((fails+1))
  printf "%s  %-10s identical to original: baseline %8s%%  new %8s%%   new exact: %s   [%s]\n" $verdict $n "$sb" "$sn" "$ex" "$path"
}
check D3m_s 2000 300 13 PROFILE=hardcall MISSALL=0.005
check D3h16_s 2000 300 13 PROFILE=hardcall BITS=16
check miss 800 120 5 MISS=0.02
check edge 1000 10 7 MISSFIX=1:99,2:100,3:101,4:1,5:999 MONO=6
check D1m_s 2000 300 11 MISSALL=0.002
check D2m_s 2000 300 12 PROFILE=imputed MISSALL=0.002
echo "R3: $fails failed"
[ $fails -eq 0 ]
