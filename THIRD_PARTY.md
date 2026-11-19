# Third-party software

fastLD's own sources are MIT-licensed (see `LICENSE`). The release binary is statically linked
and contains the libraries below. Versions are those of the Ubuntu 24.04 packages used by the
build workflow (`.github/workflows/build.yml`).

| Library | Used for | Licence |
|---|---|---|
| pgenlib (plink-ng), vendored in `src/pgenlib/`, unmodified | reading PLINK 2 `.pgen` | LGPL-3.0-or-later |
| OpenBLAS | floating-point path (`dsyrk`) | BSD-3-Clause |
| libdeflate | zlib decompression of BGEN, gzip output | MIT |
| Zstandard | zstd decompression of BGEN | BSD-3-Clause |
| SQLite | reading bgenix `.bgi` indexes | public domain |
| GCC runtime (libstdc++, libgomp, libgcc, libgfortran) | runtime | GPL-3.0 with the GCC Runtime Library Exception |
| libquadmath | runtime (via libgfortran) | LGPL-2.1-or-later |
| GNU C Library | runtime | LGPL-2.1-or-later |

For the LGPL components the complete source of fastLD is in this repository, so the binary can be
rebuilt and relinked against modified versions of those libraries with `make static`.
