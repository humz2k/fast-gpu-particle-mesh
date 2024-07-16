#include "simple_grid.hpp"
#include "allocators.hpp"
#include "common.hpp"
#include "rng_initializer.hpp"

template <class fft_t>
SimpleGrid<fft_t>::SimpleGrid(const Params& params, int ng)
    : m_ng(ng), m_params(params), fft(m_ng), dist(ng) {
    m_size = m_ng * m_ng * m_ng;

    gpu_allocator.alloc(&m_d_grad, m_size * sizeof(float4));
    gpu_allocator.alloc(&m_d_greens, m_size * sizeof(float));
    gpu_allocator.alloc(&m_d_grid, m_size * sizeof(fft_t));
    gpu_allocator.alloc(&m_d_x, m_size * sizeof(fft_t));
    gpu_allocator.alloc(&m_d_y, m_size * sizeof(fft_t));
    gpu_allocator.alloc(&m_d_z, m_size * sizeof(fft_t));
}

template <class fft_t> SimpleGrid<fft_t>::~SimpleGrid() {
    gpu_allocator.free(m_d_grad);
    gpu_allocator.free(m_d_greens);
    gpu_allocator.free(m_d_grid);
    gpu_allocator.free(m_d_x);
    gpu_allocator.free(m_d_y);
    gpu_allocator.free(m_d_z);
}

template <class fft_t> void SimpleGrid<fft_t>::solve_gradient() {}

template <class fft_t> void SimpleGrid<fft_t>::solve() {}

template <class fft_t>
void SimpleGrid<fft_t>::CIC(const Particles& particles) {}

template <class fft_t>
void SimpleGrid<fft_t>::generate_fourier_amplitudes(Cosmo& cosmo) {
    LOG_INFO("generating fourier amplitudes");

    int blockSize = BLOCKSIZE;
    int numBlocks = (m_size + (blockSize - 1)) / blockSize;

    launch_generate_real_random(m_d_grid, m_params.seed(), dist, numBlocks,
                                blockSize);
    fft.forward(m_d_grid);
}

template class SimpleGrid<complexDoubleDevice>;
template class SimpleGrid<complexFloatDevice>;