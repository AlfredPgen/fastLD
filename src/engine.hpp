#pragma once
#include "common.hpp"
#include "output.hpp"

struct Timing { double parse = 0, ld = 0; };

std::unique_ptr<LDMatrix> compute_ld(const Source& src, size_t M, const Opts& o, vector<SnpStats>& st, Timing& tm);
void print_stats(const Source& src, size_t M, int nt);
