#ifndef _FGPM_PK_BINS_HPP_
#define _FGPM_PK_BINS_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"

/**
 * @brief Launches a CUDA kernel to bin the power spectrum.
 *
 * This function template launches a CUDA kernel to bin the power spectrum of a
 * grid of complex numbers. It calculates the magnitude of the wave numbers,
 * determines the appropriate bin, and accumulates the power spectrum values
 * into the specified bins.
 *
 * @tparam T The type of the elements in the grid (e.g., complexDoubleDevice,
 * complexFloatDevice).
 * @param d_grid Pointer to the device memory where the input grid of complex
 * numbers is stored.
 * @param d_bins Pointer to the device memory where the binned power spectrum
 * values will be stored.
 * @param k_min The minimum wave number.
 * @param k_delta The bin width for the wave numbers.
 * @param n_k_bins The number of bins for the wave numbers.
 * @param rl The size of the simulation box.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_bin_power(const T* d_grid, double2* d_bins, double k_min,
                      double k_delta, int n_k_bins, double rl, MPIDist dist,
                      int numBlocks, int blockSize);

#endif