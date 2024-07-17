#ifndef _FGPM_PK_BINS_HPP_
#define _FGPM_PK_BINS_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"

template <class T>
void launch_bin_power(const T* d_grid, double2* d_bins, double k_min,
                      double k_delta, int n_k_bins, double rl, MPIDist dist,
                      int numBlocks, int blockSize);

#endif