#ifndef _FGPM_RNG_INITIALIZER_HPP_
#define _FGPM_RNG_INITIALIZER_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"
#include "power_spectrum.hpp"

/**
 * @brief Launches a GPU kernel to generate real random numbers.
 *
 * This function template launches a GPU kernel to fill a grid with real random
 * numbers. It takes a pointer to the device grid, a seed for random number
 * generation, an offset, the size of the grid, and the GPU execution
 * configuration parameters (number of blocks and block size).
 *
 * @tparam T The type of the elements in the grid (e.g., complexDoubleDevice,
 * complexFloatDevice).
 * @param d_grid Pointer to the device memory where random numbers will be
 * stored.
 * @param seed The seed for the random number generator.
 * @param offset An offset to apply to the random number sequence.
 * @param size The number of random numbers to generate.
 * @param numBlocks The number of blocks to use in the GPU kernel launch.
 * @param blockSize The size of each block to use in the GPU kernel launch.
 */
template <class T>
void launch_generate_real_random(T* d_grid, int seed, const MPIDist dist,
                                 int numBlocks, int blockSize);

/**
 * @brief Launches a CUDA kernel to scale grid amplitudes by the power spectrum.
 *
 * This function launches a CUDA kernel that scales the amplitudes of a grid
 * based on the provided power spectrum. It sets up the kernel configuration
 * and passes the necessary parameters to the kernel.
 *
 * @tparam T The type of the elements in the grid.
 * @param grid Pointer to the grid array whose amplitudes will be scaled.
 * @param initial_pk Reference to the PowerSpectrum object containing initial
 * power spectrum values.
 * @param rl The size of the simulation box.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_scale_amplitudes_by_power_spectrum(T* grid,
                                               const PowerSpectrum& initial_pk,
                                               double rl, const MPIDist dist,
                                               int numBlocks, int blockSize);

#endif