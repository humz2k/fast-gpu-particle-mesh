#ifndef _FGPM_POWER_SPECTRUM_INTERPOLATION_HPP_
#define _FGPM_POWER_SPECTRUM_INTERPOLATION_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"

/**
 * @brief Launches a CUDA kernel to interpolate the power spectrum.
 *
 * This function launches a CUDA kernel to interpolate the power spectrum values
 * for a given grid. It sets up the kernel configuration and passes the
 * necessary parameters to the kernel.
 *
 * @tparam T The type of the elements in the output array.
 * @param out Pointer to the output array where interpolated values will be
 * stored.
 * @param in Pointer to the input array containing power spectrum values.
 * @param k_delta The spacing between k bins in the power spectrum.
 * @param k_min The minimum k value in the power spectrum.
 * @param rl The size of the simulation box.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_interp_pk(T* out, double* in, double k_delta, double k_min,
                      double rl, const MPIDist dist, int numBlocks,
                      int blockSize);

#endif