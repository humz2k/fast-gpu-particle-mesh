#include "grid.hpp"
#include "allocators.hpp"
#include "common.hpp"
#include "rng_initializer.hpp"

SimpleGrid::SimpleGrid(const Params& params, int ng)
    : m_ng(ng), m_params(params) {
    m_size = m_ng * m_ng * m_ng;

    gpu_allocator.alloc(&m_d_grad, m_size * sizeof(float4));
    gpu_allocator.alloc(&m_d_greens, m_size * sizeof(float));
    gpu_allocator.alloc(&m_d_grid, m_size * sizeof(cufftDoubleComplex));
    gpu_allocator.alloc(&m_d_x, m_size * sizeof(cufftDoubleComplex));
    gpu_allocator.alloc(&m_d_y, m_size * sizeof(cufftDoubleComplex));
    gpu_allocator.alloc(&m_d_z, m_size * sizeof(cufftDoubleComplex));
}

SimpleGrid::~SimpleGrid() {
    gpu_allocator.free(m_d_grad);
    gpu_allocator.free(m_d_greens);
    gpu_allocator.free(m_d_grid);
    gpu_allocator.free(m_d_x);
    gpu_allocator.free(m_d_y);
    gpu_allocator.free(m_d_z);
}

void SimpleGrid::solve_gradient() {}

void SimpleGrid::solve() {}

void SimpleGrid::CIC(const Particles& particles) {}

void SimpleGrid::generate_fourier_amplitudes(Cosmo& cosmo) {
    LOG_INFO("generating fourier amplitudes");
    int blockSize = BLOCKSIZE;
    int numBlocks = (m_size + (blockSize - 1)) / blockSize;
    launch_generate_real_random(m_d_grid,m_params.seed(),0,m_size,numBlocks,blockSize);
}
