#include "mpi_distribution.hpp"

MPIDist::MPIDist(int ng) : m_ng(ng), m_local_grid_size{ng,ng,ng}{

}

MPIDist::MPIDist(int ng, int rank, int3 rank_dims, int3 rank_coords, int3 local_grid_size) : m_ng(ng), m_rank(rank), m_rank_dims(rank_dims), m_rank_coords(rank_coords), m_local_grid_size(local_grid_size){

}

__host__ __device__ int3 MPIDist::idx2local(int idx){
    return make_int3(
        (idx/m_local_grid_size.z)/m_local_grid_size.y,
        (idx/m_local_grid_size.z)%m_local_grid_size.y,
        idx%m_local_grid_size.z);
}

__host__ __device__ int3 MPIDist::idx2global(int idx){
    return idx2local(idx) + (m_rank_coords * m_local_grid_size);
}