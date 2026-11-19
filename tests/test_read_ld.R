# usage: Rscript tests/test_read_ld.R PREFIX   (PREFIX.ld written with --write-text --precision 15,
#        PREFIX.<level>.bcor for ultra-low, low, medium, high)
args <- commandArgs(trailingOnly = TRUE)
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "..", "R", "read_ld.R"))
pre <- args[1]
txt <- read_ld_text(paste0(pre, ".ld"))
step <- c("ultra-low" = 2^-6, "low" = 2^-14, "medium" = 2^-30, "high" = 1e-15)
ok <- TRUE
for (lev in names(step)) {
  b <- read_bcor(paste0(pre, ".", lev, ".bcor"))
  R <- b$R
  na_same <- all(is.na(R) == is.na(txt)) && any(is.na(txt))
  f <- !is.na(txt)
  d <- txt[f] - R[f]                         # floor encoding: BCOR value <= text value
  within <- all(abs(d) <= step[[lev]] + 1e-15)
  below <- all(R[f] <= txt[f] + 1e-15)
  diag1 <- all(diag(R) == 1)
  n_one <- sum(txt[f] == 1 & row(txt)[f] != col(txt)[f]); n_mone <- sum(txt[f] == -1)
  ones_ok <- all(R[f][txt[f] == 1] == 1) && all(R[f][txt[f] == -1] == -1)
  cat(sprintf("%-9s M=%d  NA pattern %s  max|diff| %.3g (step %.3g)  floor %s  diag %s  r=1 pairs %d, r=-1 pairs %d exact %s  snps %d\n",
              lev, nrow(R), na_same, max(abs(d)), step[[lev]], below, diag1, n_one, n_mone, ones_ok, nrow(b$snps)))
  ok <- ok && na_same && within && below && diag1 && ones_ok && nrow(b$snps) == nrow(R)
}
cat(if (ok) "R READER: PASS\n" else "R READER: FAIL\n")
quit(status = if (ok) 0 else 1)
