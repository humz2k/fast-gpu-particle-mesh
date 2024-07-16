#ifndef _FGPM_MPI_DISTRIBUTIONS_HPP_
#define _FGPM_MPI_DISTRIBUTIONS_HPP_

#include "gpu.hpp"

struct MPIDist {
    int m_rank;
    int m_ng;
    int3 m_rank_dims;
    int3 m_rank_coords;
    int3 m_local_grid_size;

    MPIDist(int ng, int rank, int3 rank_dims, int3 rank_coords,
            int3 local_grid_size);

    MPIDist(int ng);

    __host__ __device__ int3 global_coords(int local_idx);
    __host__ __device__ int3 local_coords(int local_idx);
    __host__ __device__ int global_idx(int local_idx);
};

#endif