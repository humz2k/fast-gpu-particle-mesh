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

/**
 * @brief Launches a CUDA kernel to transform the density field.
 *
 * This function launches a CUDA kernel to transform the density field into
 * three separate components along the x, y, and z axes. The transformation
 * includes scaling and interpolation operations.
 *
 * @tparam T The type of the elements in the grid.
 * @param d_grid Pointer to the device memory where the original density field
 * is stored.
 * @param d_x Pointer to the device memory where the transformed density field
 * along the x-axis will be stored.
 * @param d_y Pointer to the device memory where the transformed density field
 * along the y-axis will be stored.
 * @param d_z Pointer to the device memory where the transformed density field
 * along the z-axis will be stored.
 * @param delta The D+ value used in the transformation.
 * @param rl The size of the simulation box.
 * @param a The scale factor.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_transform_density_field(T* d_grid, T* d_x, T* d_y, T* d_z,
                                    double delta, double rl, double a,
                                    MPIDist dist, int numBlocks, int blockSize);

/**
 * @brief Launches a CUDA kernel to combine density vectors.
 *
 * This function launches a CUDA kernel that combines the density field vectors
 * from the x, y, and z components into a single gradient vector field. The
 * resulting vector field is stored in the provided device memory.
 *
 * @tparam T The type of the elements in the grid.
 * @param d_grad Pointer to the device memory where the resulting gradient
 * vectors will be stored.
 * @param d_x Pointer to the device memory where the density field along the
 * x-axis is stored.
 * @param d_y Pointer to the device memory where the density field along the
 * y-axis is stored.
 * @param d_z Pointer to the device memory where the density field along the
 * z-axis is stored.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_combine_density_vectors(float3* d_grad, T* d_x, T* d_y, T* d_z,
                                    MPIDist dist, int numBlocks, int blockSize);

/**
 * @brief Launches a CUDA kernel to place particles in the simulation grid.
 *
 * This function launches a CUDA kernel that calculates the positions and
 * velocities of particles based on the gradient field and other parameters.
 *
 * @param d_pos Pointer to the device memory where the particle positions will be stored.
 * @param d_vel Pointer to the device memory where the particle velocities will be stored.
 * @param d_grad Pointer to the device memory where the gradient field is stored.
 * @param delta The delta value used in the transformation.
 * @param dot_delta The dot product of delta and some other parameter.
 * @param rl The size of the simulation box.
 * @param a The scale factor.
 * @param deltaT The time step size.
 * @param fscal The scaling factor.
 * @param ng The size of the grid.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
void launch_place_particles(float3* d_pos, float3* d_vel, const float3* d_grad,
                            double delta, double dot_delta, double rl, double a,
                            double deltaT, double fscal, int ng, MPIDist dist,
                            int numBlocks, int blockSize);

#endif