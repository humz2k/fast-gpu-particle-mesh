#ifndef _FGPM_PARTICLE_ACTIONS_HPP_
#define _FGPM_PARTICLE_ACTIONS_HPP_

#include "gpu.hpp"
#include "mpi_distribution.hpp"

template <class T>
void launch_CIC_kernel(T* d_grid, const float3* d_pos, int n_particles,
                       float mass, MPIDist dist, int numBlocks, int blockSize);

void launch_update_positions_kernel(float3* d_pos, const float3* d_vel,
                                    float prefactor, float ng, int nlocal,
                                    int numBlocks, int blockSize);

void launch_update_velocities_kernel(float3* d_vel, const float3* d_pos,
                                     const float3* d_grad, double deltaT,
                                     double fscal, int nlocal, MPIDist dist,
                                     int numBlocks, int blockSize);

#endif