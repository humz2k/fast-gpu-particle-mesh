#ifndef _FGPM_RNG_INITIALIZER_HPP_
#define _FGPM_RNG_INITIALIZER_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"

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

#endif