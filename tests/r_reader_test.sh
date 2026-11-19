#!/usr/bin/env bash
# WP7.2: R/read_ld.R read_bcor() vs --write-text --precision 15, all compression levels, data with
# NA SNPs, a monomorphic SNP and pairs with r = 1 and r = -1.  usage: tests/r_reader_test.sh NEW_BIN
set -euo pipefail
NEW=$(readlink -f "$1"); HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORKDIR:-/tmp/fastLD-rtest}; mkdir -p "$W"; cd "$W"
[ -f rt.master ] || MISSFIX=3:150,8:400 MONO=12 DUPPREV=21,40 NEGPREV=22,61 python3 "$HERE/make_bgen.py" $W/rt 1000 80 31 >/dev/null
"$NEW" --in-files rt.master --write-text --precision 15 >/dev/null
for lev in ultra-low low medium high; do
  sed "s#;$W/rt.bcor;#;$W/rt.$lev.bcor;#" rt.master > rt.$lev.master
  "$NEW" --in-files rt.$lev.master --write-bcor --compression $lev >/dev/null
done
Rscript "$HERE/test_read_ld.R" $W/rt
