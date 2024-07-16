#include "mpi_distribution.hpp"

MPIDist::MPIDist(int ng)
    : m_ng(ng), m_rank(0), m_rank_dims{1, 1, 1}, m_rank_coords{0, 0, 0},
      m_local_grid_size{ng, ng, ng} {}

MPIDist::MPIDist(int ng, int rank, int3 rank_dims, int3 rank_coords,
                 int3 local_grid_size)
    : m_ng(ng), m_rank(rank), m_rank_dims(rank_dims),
      m_rank_coords(rank_coords), m_local_grid_size(local_grid_size) {}

__host__ __device__ int3 MPIDist::local_coords(int local_idx) {
    return make_int3((local_idx / m_local_grid_size.z) / m_local_grid_size.y,
                     (local_idx / m_local_grid_size.z) % m_local_grid_size.y,
                     local_idx % m_local_grid_size.z);
}

__host__ __device__ int3 MPIDist::global_coords(int local_idx) {
    return local_coords(local_idx) + (m_rank_coords * m_local_grid_size);
}

__host__ __device__ int MPIDist::global_idx(int local_idx) {
    int3 global = global_coords(local_idx);
    return global.x * m_ng * m_ng + global.y * m_ng + global.z;
}