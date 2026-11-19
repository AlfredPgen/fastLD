# fastLD — build:   make            (dynamic, uses system OpenBLAS/zlib/zstd/sqlite3)
#                         make static     (single portable binary)
# Debian/Ubuntu deps:     apt install g++ make libopenblas-dev libdeflate-dev libzstd-dev libsqlite3-dev
CXX      ?= g++
CXXFLAGS ?= -O3 -march=x86-64-v2 -mtune=generic -std=c++17 -fopenmp -Wall -Wno-unused-result -Wno-psabi
BLAS_INC ?= $(shell pkg-config --cflags openblas 2>/dev/null || echo -I/usr/include/x86_64-linux-gnu/openblas-pthread)
LIBS     ?= -lopenblas -lzstd -ldeflate -lsqlite3 -lpthread -ldl -lm

all: fastLD

PGENLIB = src/pgenlib/pgenlib_read.cc src/pgenlib/pgenlib_misc.cc src/pgenlib/plink2_base.cc src/pgenlib/plink2_bits.cc src/pgenlib/plink2_simd.cc
SRCS = src/main.cpp src/util.cpp src/bgen.cpp src/plink.cpp src/engine.cpp src/output.cpp $(PGENLIB)
HDRS = $(wildcard src/*.hpp) $(wildcard src/pgenlib/*.h)

fastLD: $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(BLAS_INC) -o $@ $(SRCS) $(LIBS)

# static build: OpenBLAS is DYNAMIC_ARCH in Debian/Ubuntu, so the binary picks
# optimal kernels (Haswell/SkylakeX/Zen...) at run time on any x86-64 machine.
static: $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(BLAS_INC) -static -o fastLD $(SRCS) \
	  -Wl,--start-group -lopenblas -lgfortran -lquadmath -lzstd -ldeflate -lsqlite3 -lpthread -ldl -lm -Wl,--end-group

# optional reference binary (an earlier release) for the byte-identity regression test
BASELINE ?= ./fastLD-reference

# tier 1: self-contained (exact reference, gzip, error handling; golden test if BASELINE exists)
test: fastLD
	bash tests/selftest.sh $(BASELINE) ./fastLD

# tier 2: compare with the original LDstore v2.0 (downloaded at run time): structure and values
# (tests/run_tests.sh) and rule R3 for data that the baseline computed in floating point
test-orig: fastLD
	bash tests/run_tests.sh
	WORKDIR=/tmp/fastLD-r3 bash tests/r3_check.sh /tmp/fastLD-tests/ldstore_v2.0_x86_64/ldstore_v2.0_x86_64 $(BASELINE) ./fastLD

# PLINK 1/2 input (downloads plink2; needs the pgenlib Python package for the dosage reference)
test-plink: fastLD
	bash tests/plink_test.sh ./fastLD

# R reader (needs R; data.table only for the load benchmark)
test-r: fastLD
	bash tests/r_reader_test.sh ./fastLD

# sanitizers: build fastLD-asan and run the self-contained tests with it
asan: $(SRCS) $(HDRS)
	$(CXX) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	  -std=c++17 -fopenmp -march=x86-64-v2 $(BLAS_INC) -o fastLD-asan $(SRCS) $(LIBS)
	ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 bash tests/selftest.sh $(BASELINE) ./fastLD-asan

clean:
	rm -f fastLD fastLD-asan

.PHONY: all static test test-orig test-plink test-r asan clean
