#ifndef _FGPM_MPI_DISTRIBUTIONS_HPP_
#define _FGPM_MPI_DISTRIBUTIONS_HPP_

#include "gpu.hpp"

struct MPIDist {
    int m_rank = 0;
    int m_ng;
    int3 m_rank_dims = (int3){1, 1, 1};
    int3 m_rank_coords = (int3){0, 0, 0};
    int3 m_local_grid_size;

    MPIDist(int ng, int rank, int3 rank_dims, int3 rank_coords,
            int3 local_grid_size);

    MPIDist(int ng);

    __host__ __device__ int3 idx2global(int idx);
    __host__ __device__ int3 idx2local(int idx);
};

#endif