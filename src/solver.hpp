#ifndef _FGPM_SOLVER_HPP_
#define _FGPM_SOLVER_HPP_

#include "mpi_distribution.hpp"
#include "gpu.hpp"

template<class T>
void launch_kspace_solve_gradient(const T* grid, T* d_x, T* d_y, T* d_z, MPIDist dist, int numBlocks, int blockSize);

template <class T>
void launch_combine_vectors(float3* d_grad, const T* d_x, const T* d_y, const T* d_z,
                                    MPIDist dist, int numBlocks,
                                    int blockSize);

#endif