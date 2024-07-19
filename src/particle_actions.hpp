#ifndef _FGPM_PARTICLE_ACTIONS_HPP_
#define _FGPM_PARTICLE_ACTIONS_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"

/**
 * @brief Launches a CUDA kernel for Cloud-In-Cell (CIC) assignment.
 *
 * This function template launches a CUDA kernel that performs the CIC
 * assignment of particle masses to a grid. The grid is updated based on the
 * particle positions and masses.
 *
 * @tparam T The type of the elements in the grid (e.g., complexDoubleDevice,
 * complexFloatDevice).
 * @param d_grid Pointer to the device memory where the grid is stored.
 * @param d_pos Pointer to the device memory where the particle positions are
 * stored.
 * @param n_particles The number of particles.
 * @param mass The mass of each particle.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
template <class T>
void launch_CIC_kernel(T* d_grid, const float3* d_pos, int n_particles,
                       float mass, MPIDist dist, int numBlocks, int blockSize);

/**
 * @brief Launches a CUDA kernel to update particle positions.
 *
 * This function launches a CUDA kernel that updates the positions of particles
 * based on their velocities and a given prefactor.
 *
 * @param d_pos Pointer to the device memory where the particle positions are
 * stored.
 * @param d_vel Pointer to the device memory where the particle velocities are
 * stored.
 * @param prefactor The prefactor used in updating the positions.
 * @param ng The size of the grid.
 * @param nlocal The number of local particles to process.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
void launch_update_positions_kernel(float3* d_pos, const float3* d_vel,
                                    float prefactor, float ng, int nlocal,
                                    int numBlocks, int blockSize);

/**
 * @brief Launches a CUDA kernel to update particle velocities.
 *
 * This function launches a CUDA kernel that updates the velocities of particles
 * based on the gradient of the potential field, a time step, and a scaling
 * factor.
 *
 * @param d_vel Pointer to the device memory where the particle velocities are
 * stored.
 * @param d_pos Pointer to the device memory where the particle positions are
 * stored.
 * @param d_grad Pointer to the device memory where the gradient field is
 * stored.
 * @param deltaT The time step size.
 * @param fscal The scaling factor.
 * @param nlocal The number of local particles to process.
 * @param dist The MPIDist object containing distribution and grid information.
 * @param numBlocks The number of blocks to use in the CUDA kernel launch.
 * @param blockSize The size of each block to use in the CUDA kernel launch.
 */
void launch_update_velocities_kernel(float3* d_vel, const float3* d_pos,
                                     const float3* d_grad, double deltaT,
                                     double fscal, int nlocal, MPIDist dist,
                                     int numBlocks, int blockSize);

#endif