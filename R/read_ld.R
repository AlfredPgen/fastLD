# Read LD matrices written by fastLD / LDstore v2.0 into R. Base R only, no compiled code.
#
#   ld <- read_bcor("region.bcor")   # list(R = M x M numeric matrix, snps = data.frame, n_samples = integer)
#   R  <- read_ld_text("region.ld")  # M x M numeric matrix from --write-text output
#
# BCOR v1.1 layout (see write_bcor / encode_tail in src/output.cpp):
#   header  : "bcor1.1" | uint64 file size | uint32 n_samples | uint32 n_snps | uint8 compression
#             | uint64 offset of the correlation block
#   meta    : per SNP  uint32 L | uint32 index | uint16 len + rsid | uint32 position
#             | uint16 len + chromosome | uint32 len + allele1 | uint32 len + allele2
#   values  : strictly lower triangle, column by column: for x = 1..M-1 the values r(x, y), y = x+1..M,
#             each an unsigned integer v of b bytes with r = v / 2^(8b-2) - 1 (encoded with floor),
#             NA = 0xD0 followed by zero bytes. compression 3 = 1 byte, 0 = 2, 1 = 4, 2 = 8 bytes.
#
# R has no unsigned 32-bit integer: the bit pattern 0x80000000 (exactly r = 1 at "medium") reads as
# NA_integer_. 4- and 8-byte values are read as signed 32-bit words and reinterpreted exactly
# (NA_integer_ -> 2^31, negative -> + 2^32); 8-byte values are combined from exactly representable
# terms (error below 2e-16).
#
# Diagonal: 1. Pairs stored as NA (SNPs above --sample-miss, or monomorphic SNPs) become NaN, which is
# what --write-text prints ("nan" / "-nan"). Peak memory is about one M x M matrix plus ~100 MB.

read_bcor <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  u16 <- function(n = 1) readBin(con, "integer", n = n, size = 2, signed = FALSE, endian = "little")
  u32 <- function() { h <- u16(2); h[1] + h[2] * 65536 }
  u64 <- function() { h <- u16(4); h[1] + h[2] * 2^16 + h[3] * 2^32 + h[4] * 2^48 }
  str_n <- function(n) if (n == 0) "" else rawToChar(readBin(con, "raw", n = n))

  magic <- rawToChar(readBin(con, "raw", n = 7))
  if (magic != "bcor1.1") stop("'", path, "' is not a BCOR v1.1 file")
  fsize <- u64(); n_samples <- u32(); M <- u32()
  compression <- readBin(con, "integer", n = 1, size = 1, signed = FALSE)
  corr_off <- u64()
  nb <- switch(as.character(compression), "0" = 2L, "1" = 4L, "2" = 8L, "3" = 1L,
               stop("unknown BCOR compression code ", compression))

  rsid <- character(M); chrom <- character(M); a1 <- character(M); a2 <- character(M)
  pos <- numeric(M)
  for (i in seq_len(M)) {
    u32(); u32()                                  # record length, index
    rsid[i] <- str_n(u16()); pos[i] <- u32()
    chrom[i] <- str_n(u16())
    a1[i] <- str_n(u32()); a2[i] <- str_n(u32())
  }
  snps <- data.frame(rsid = rsid, position = pos, chromosome = chrom, allele1 = a1, allele2 = a2,
                     stringsAsFactors = FALSE)
  seek(con, corr_off)

  scale <- 2^(8 * nb - 2)
  # unsigned 32-bit words as doubles. readBin has no unsigned 32-bit type: read them as signed
  # (a plain copy) and reinterpret. The bit pattern 0x80000000 -- exactly r = 1 at "medium" --
  # comes back as NA_integer_ and is mapped to 2^31; negative values get 2^32 added. Exact.
  u32vec <- function(k) {
    x <- readBin(con, "integer", n = k, size = 4, endian = "little")
    v <- as.double(x)
    na <- is.na(x)
    v[na] <- 2^31
    neg <- which(!na & x < 0)
    v[neg] <- v[neg] + 2^32
    v
  }
  decode <- function(k) {          # next k values; numeric r with NaN for NA codes
    if (nb == 1L) {
      v <- readBin(con, "integer", n = k, size = 1, signed = FALSE)
      r <- v / scale - 1; r[v == 0xD0] <- NaN
    } else if (nb == 2L) {
      v <- readBin(con, "integer", n = k, size = 2, signed = FALSE, endian = "little")
      r <- v / scale - 1; r[v == 0xD000] <- NaN
    } else if (nb == 4L) {
      r <- u32vec(k) / scale - 1                     # exact
      r[r == 0xD0000000 / scale - 1] <- NaN
    } else {
      w <- u32vec(2 * k)                             # little endian: low word, high word
      lo <- w[c(TRUE, FALSE)]; hi <- w[c(FALSE, TRUE)]
      r <- (hi / 2^30 - 1) + lo / 2^62               # v / 2^62 - 1 from exactly representable terms
      r[hi == 0xD0000000 & lo == 0] <- NaN
    }
    if (length(r) != k) stop("'", path, "' is truncated")
    r
  }

  R <- matrix(0, M, M, dimnames = list(rsid, rsid))
  CH <- 2e6                        # values per readBin call (bounds the temporaries, ~100 MB)
  big <- as.double(M) * M > .Machine$integer.max
  x <- 1; nch <- 0
  while (x < M) {
    # columns x..x2 hold sum_{c=x}^{x2} (M - c) values: the lower triangle, column-major
    cnt <- M - x
    x2 <- x
    while (x2 < M - 1 && cnt + (M - x2 - 1) <= CH) { x2 <- x2 + 1; cnt <- cnt + (M - x2) }
    v <- decode(cnt)
    cs <- x:x2
    k <- M - cs
    idx <- sequence(k) + rep(if (big) (as.double(cs) - 1) * M + cs else (cs - 1L) * M + cs, k)
    R[idx] <- v                                  # contiguous runs, one per column
    rm(v, idx, k)
    if ((nch <- nch + 1) %% 8 == 0) invisible(gc(verbose = FALSE))   # keep temporaries from piling up
    x <- x2 + 1
  }
  # mirror into the upper triangle tile by tile (cache-friendly transposes)
  TS <- 512
  starts <- seq(1, M, by = TS)
  for (bj in seq_along(starts)) {
    J <- starts[bj]:min(M, starts[bj] + TS - 1)
    Bk <- R[J, J]
    up <- upper.tri(Bk)
    Bk[up] <- t(Bk)[up]
    R[J, J] <- Bk
    if (bj < length(starts)) for (bi in (bj + 1):length(starts)) {
      I <- starts[bi]:min(M, starts[bi] + TS - 1)
      R[J, I] <- t(R[I, J])
      if ((nch <- nch + 1) %% 32 == 0) invisible(gc(verbose = FALSE))
    }
  }
  rm(Bk, up)
  R[(seq_len(M) - 1) * (M + 1) + 1] <- 1         # diagonal
  invisible(gc(verbose = FALSE))
  list(R = R, snps = snps, n_samples = n_samples)
}

# Text output (--write-text): space separated, a trailing separator on every row, "nan"/"-nan" for NA.
# Base R reader (slow for large M); data.table::fread is much faster if available.
read_ld_text <- function(path) {
  first <- readLines(path, n = 1)
  M <- length(strsplit(trimws(first), " +")[[1]])
  v <- scan(path, what = character(), quiet = TRUE)
  if (length(v) != M * M) stop("'", path, "': expected ", M * M, " values, found ", length(v))
  r <- suppressWarnings(as.numeric(v))
  r[v %in% c("nan", "-nan")] <- NaN
  matrix(r, M, M, byrow = TRUE)
}
