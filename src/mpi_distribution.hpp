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
            int3 local_grid_size)
        : m_rank(rank), m_ng(ng), m_rank_dims(rank_dims),
          m_rank_coords(rank_coords), m_local_grid_size(local_grid_size) {}

    MPIDist(int ng)
        : m_rank(0), m_ng(ng), m_rank_dims{1, 1, 1}, m_rank_coords{0, 0, 0},
          m_local_grid_size{ng, ng, ng} {}

    __forceinline__ __host__ __device__ int3 local_coords(int local_idx) {
        return make_int3(
            (local_idx / m_local_grid_size.z) / m_local_grid_size.y,
            (local_idx / m_local_grid_size.z) % m_local_grid_size.y,
            local_idx % m_local_grid_size.z);
    }

    __forceinline__ __host__ __device__ int3 global_coords(int local_idx) {
        return local_coords(local_idx) + (m_rank_coords * m_local_grid_size);
    }

    __forceinline__ __host__ __device__ int global_idx(int local_idx) {
        int3 global = global_coords(local_idx);
        return global.x * m_ng * m_ng + global.y * m_ng + global.z;
    }
};

#endif