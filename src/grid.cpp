#include "grid.hpp"
#include "allocators.hpp"

Grid::Grid(const Params& params){}
Grid::Grid(){}
Grid::~Grid(){}

SimpleGrid::SimpleGrid(const Params& params) : m_params(params){
    int ng = m_params.ng();
    int size = ng*ng*ng;

    gpu_allocator.alloc(&m_d_grad,size * sizeof(float4));
    gpu_allocator.alloc(&m_d_greens,size * sizeof(float));
    gpu_allocator.alloc(&m_d_grid,size * sizeof(cufftDoubleComplex));
    gpu_allocator.alloc(&m_d_x,size * sizeof(cufftDoubleComplex));
    gpu_allocator.alloc(&m_d_y,size * sizeof(cufftDoubleComplex));
    gpu_allocator.alloc(&m_d_z,size * sizeof(cufftDoubleComplex));
}

SimpleGrid::~SimpleGrid(){
    gpu_allocator.free(m_d_grad);
    gpu_allocator.free(m_d_greens);
    gpu_allocator.free(m_d_grid);
    gpu_allocator.free(m_d_x);
    gpu_allocator.free(m_d_y);
    gpu_allocator.free(m_d_z);
}

void SimpleGrid::solve_gradient(){

}

void SimpleGrid::solve(){

}

void SimpleGrid::CIC(const Particles& particles){

}