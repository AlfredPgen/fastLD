#!/usr/bin/env bash
# WP8 tests: PLINK 1 (bed) and PLINK 2 (pgen) input.
#  1. hard-call data (with and without missing genotypes) written as BGEN, converted to bed and pgen
#     by plink2 (downloaded here); BGEN, bed and pgen must give byte-identical text and BCOR output,
#     also with alleles swapped in the bim/pvar (ref-last import) and for a reordered Z subset;
#  2. pgen with dosages: --write-text --precision 15 vs a float64 numpy reference read with the
#     pgenlib Python package (pip install pgenlib): every value within 1e-9.
# usage: tests/plink_test.sh NEW_BIN [PLINK2_BIN]
set -uo pipefail
NEW=$(readlink -f "$1"); HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORKDIR:-/tmp/fastLD-plink}; mkdir -p "$W"; cd "$W"
P2=${2:-}
if [ -z "$P2" ]; then
  if [ ! -x plink2 ]; then
    curl -sL -o p2.zip "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_20241206.zip" &&
      python3 -c "import zipfile; zipfile.ZipFile('p2.zip').extractall('.')" && chmod +x plink2
  fi
  P2=$W/plink2
fi
"$P2" --version >/dev/null || { echo "cannot run plink2"; exit 1; }
GEN="python3 $HERE/make_bgen.py"
fails=0; pass=0
ok() { pass=$((pass+1)); echo "PASS  $*"; }
bad() { fails=$((fails+1)); echo "FAIL  $*"; }

masters() { # prefix tag(ref-first|ref-last)  -> writes $p.$tag.bed.master, $p.$tag.pgen.master
  local p=$1 tag=$2 n
  n=$(($(wc -l < $p.sample) - 2))
  "$P2" --bgen $p.bgen $tag --sample $p.sample --make-bed --out $p.$tag.b >/dev/null 2>&1
  "$P2" --bgen $p.bgen $tag --sample $p.sample --make-pgen --out $p.$tag.p >/dev/null 2>&1
  printf "z;bed;bim;fam;ld;bcor;n_samples\n$W/$p.z;$W/$p.$tag.b.bed;$W/$p.$tag.b.bim;$W/$p.$tag.b.fam;$W/out.bed.ld;$W/out.bed.bcor;$n\n" > $p.$tag.bed.master
  printf "z;pgen;pvar;psam;ld;bcor;n_samples\n$W/$p.z;$W/$p.$tag.p.pgen;$W/$p.$tag.p.pvar;$W/$p.$tag.p.psam;$W/out.pgen.ld;$W/out.pgen.bcor;$n\n" > $p.$tag.pgen.master
}
same3() { # prefix tag zfile label
  local p=$1 tag=$2 z=$3 label=$4
  sed -e "s#;$W/$p.ld;#;$W/out.bgen.ld;#" -e "s#;$W/$p.bcor;#;$W/out.bgen.bcor;#" -e "s#^$W/$p.z;#$z;#" $p.master > m.bgen
  sed -e "s#^$W/$p.z;#$z;#" $p.$tag.bed.master > m.bed
  sed -e "s#^$W/$p.z;#$z;#" $p.$tag.pgen.master > m.pgen
  for mode in text medium high; do
    if [ $mode = text ]; then args=(--write-text); f=ld; else args=(--write-bcor --compression $mode); f=bcor; fi
    for fmt in bgen bed pgen; do rm -f out.$fmt.$f; "$NEW" --in-files m.$fmt "${args[@]}" > log.$fmt 2>&1 || { bad "$label $fmt: $(tail -2 log.$fmt)"; continue 2; }; done
    if cmp -s out.bgen.$f out.bed.$f && cmp -s out.bgen.$f out.pgen.$f; then ok "$label $mode: BGEN == bed == pgen  [$(grep -oP 'Correlation path\s*: \K.*' log.bed)]"
    else bad "$label $mode: outputs differ"; fi
  done
}

for spec in "hc 2000 300 51 PROFILE=hardcall" "hcm 2000 300 52 PROFILE=hardcall MISSALL=0.01"; do
  set -- $spec; p=$1
  [ -f $p.master ] || env "${@:5}" $GEN $W/$p $2 $3 $4 >/dev/null
  # reordered subset of the Z file
  { head -1 $p.z; tail -n +2 $p.z | awk 'NR % 3 == 0' | sort -r; } > $p.sub.z
  for tag in ref-first ref-last; do
    masters $p $tag
    same3 $p $tag $W/$p.z "$p $tag full"
    same3 $p $tag $W/$p.sub.z "$p $tag reordered-subset"
  done
done

# incl: sample subset matched on IID (first token per line), same result for all three formats
p=hcm
python3 -c "import random; random.seed(3); open('$W/sub.incl','w').write(''.join(f'S{i} extra-token\\n' for i in sorted(random.sample(range(1,2001),700))))"
for fmt in bgen bed pgen; do
  if [ $fmt = bgen ]; then src=$p.master; else src=$p.ref-last.$fmt.master; fi
  python3 - $src $W/sub.incl $W/out.$fmt.ld > m.incl.$fmt <<'E2'
import sys; f, inc = sys.argv[1:3]; L = open(f).read().split("\n")
h = L[0].split(";"); v = L[1].split(";")
h.append("incl"); v.append(inc); v[h.index("n_samples")] = "700"
if "ld" in h: v[h.index("ld")] = sys.argv[3]
print(";".join(h)); print(";".join(v))
E2
  rm -f out.$fmt.ld; "$NEW" --in-files m.incl.$fmt --write-text > log.incl.$fmt 2>&1 || bad "incl $fmt: $(tail -1 log.incl.$fmt)"
done
if cmp -s out.bgen.ld out.bed.ld && cmp -s out.bgen.ld out.pgen.ld && grep -q "Number of samples     : 700" log.incl.bed; then ok "incl subset (700 samples): BGEN == bed == pgen"; else bad "incl subset"; fi

# pgen with dosages vs numpy reference
p=dos
[ -f $p.master ] || PROFILE=dense MISSALL=0.003 $GEN $W/$p 1500 200 53 >/dev/null
for tag in ref-first ref-last; do
  masters $p $tag
  "$NEW" --in-files $p.$tag.pgen.master --write-text --precision 15 > log.dos 2>&1
  if python3 - "$W/$p.$tag.p.pgen" "$W/$p.$tag.p.pvar" "$W/$p.z" "$W/out.pgen.ld" <<'E'
import sys, numpy as np, pgenlib
pg, pvar, zf, ld = sys.argv[1:5]
v = [l.split() for l in open(pvar) if not l.startswith("#")]
z = [l.split() for l in open(zf)][1:]
key = {(r[0].lstrip("0"), r[1]): i for i, r in enumerate(v)}
rd = pgenlib.PgenReader(pg.encode())
n = rd.get_raw_sample_ct()
X = np.empty((n, len(z)))
buf = np.empty(n, np.float64)
for j, s in enumerate(z):
    i = key[(s[1].lstrip("0"), s[2])]
    rd.read_dosages(i, buf)                    # ALT dosage, -9 = missing
    d = buf.copy(); m = d == -9
    if v[i][4] != s[4]: d = 2 - d              # Z allele2 must be the counted allele
    d[m] = np.nan; X[:, j] = d
miss = np.isnan(X)
mu = np.nanmean(X, 0); Z = np.where(miss, 0.0, X - mu)       # mean imputation
ss = (Z * Z).sum(0); R = (Z.T @ Z) / np.sqrt(np.outer(ss, ss))
np.fill_diagonal(R, 1.0)
nanrate = miss.mean(0); na = nanrate >= 0.1
T = np.array([[np.nan if t in ("nan", "-nan") else float(t) for t in l.split()] for l in open(ld)])
f = ~(na[:, None] | na[None, :])
d = np.abs(T[f] - R[f]).max()
print(f"max |r - numpy float64 reference| = {d:.2e}  ({f.sum()} values)")
sys.exit(0 if d < 1e-9 else 1)
E
  then ok "pgen dosages $tag within 1e-9 [$(grep -oP 'Correlation path\s*: \K.*' log.dos)]"; else bad "pgen dosages $tag"; fi
done

echo "PLINK tests: $pass passed, $fails failed"
[ $fails -eq 0 ]
