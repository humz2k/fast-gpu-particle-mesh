#ifndef _FGPM_SOLVER_HPP_
#define _FGPM_SOLVER_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"

/**
 * @brief Launches a CUDA kernel to solve the gradient in k-space.
 *
 * This function template launches a CUDA kernel that computes the gradient of
 * the grid in k-space. It calculates the Green's function, applies it to the
 * grid, and computes the x, y, and z components of the gradient.
 *
 * @tparam T The type of the elements in the grid (e.g., complexDoubleDevice,
 * complexFloatDevice).
 * @param grid Pointer to the device memory where the input grid is stored.
 * @param d_x Pointer to the device memory where the x-component of the gradient
 * will be stored.
 * @param d_y Pointer to the device memory where the y-component of the gradient
 * will be stored.
 * @param d_z Pointer to the device memory where the z-component of the gradient
 * will be stored.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_kspace_solve_gradient(const T* grid, T* d_x, T* d_y, T* d_z,
                                  MPIDist dist, int numBlocks, int blockSize);

/**
 * @brief Launches a CUDA kernel to combine gradient vectors.
 *
 * This function template launches a CUDA kernel that combines the x, y, and z
 * components of the gradient into a single float3 vector.
 *
 * @tparam T The type of the elements in the grid (e.g., complexDoubleDevice,
 * complexFloatDevice).
 * @param d_grad Pointer to the device memory where the combined gradient
 * vectors will be stored.
 * @param d_x Pointer to the device memory where the x-component of the gradient
 * is stored.
 * @param d_y Pointer to the device memory where the y-component of the gradient
 * is stored.
 * @param d_z Pointer to the device memory where the z-component of the gradient
 * is stored.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_combine_vectors(float3* d_grad, const T* d_x, const T* d_y,
                            const T* d_z, MPIDist dist, int numBlocks,
                            int blockSize);

#endif