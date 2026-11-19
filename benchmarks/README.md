# Benchmark results

* `results/laptop_i7-11370H_1thread.log`: fastLD 1.0.0 release binary against LDstore v2.0 on an
  Intel Core i7-11370H, one thread (the numbers in the main README). Datasets are generated with
  `tests/make_bgen.py`: D1 = dense, 30,000 x 4,000, seed 11; D2 = `PROFILE=imputed`, 50,000 x 4,000,
  seed 12; D3h = `PROFILE=hardcall`, 50,000 x 4,000, seed 13.
* `results/final_*.json` and `results/baseline_*.json`: development-machine measurements (4-vCPU
  Sapphire Rapids VM, 3 runs each, alternated). "baseline" is the first internal version, before the
  kernel rewrite, the sparse and bit-plane paths and the packed LD triangle; "final" is the code
  released as 1.0.0. `D3m` = hard calls with 0.5% missing, `D1m` = dense with 0.2% missing,
  `D4` = dense 16-bit, `_t4` = 4 threads, `_bed`/`_pgen` = the same data read from PLINK files.
* `results/original_ldstore_D1.json`: LDstore v2.0 on D1 on the development machine.
* `results/r_load_M8000.json`: loading an 8,000-SNP matrix into R, text + `fread` against BCOR +
  `read_bcor`.

`tests/bench.py` and `tests/bench_r4.py` are the scripts that produced the JSON files.
