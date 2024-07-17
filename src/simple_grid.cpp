#include "simple_grid.hpp"
#include "allocators.hpp"
#include "common.hpp"
#include "initializer.hpp"
#include "pk_bins.hpp"

template <class fft_t>
SimpleGrid<fft_t>::SimpleGrid(const Params& params, int ng)
    : m_ng(ng), m_params(params), fft(m_ng), m_dist(ng) {
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

    launch_generate_real_random(m_d_grid, m_params.seed(), m_dist, numBlocks,
                                blockSize);

    fft.forward(m_d_grid);

    launch_scale_amplitudes_by_power_spectrum(m_d_grid, cosmo.initial_pk(),
                                              m_params.rl(), m_dist, numBlocks,
                                              blockSize);
}

template <class fft_t> MPIDist SimpleGrid<fft_t>::dist() const {
    return m_dist;
};

template <class fft_t> double SimpleGrid<fft_t>::k_min() const {
    return (2.0 * M_PI)/m_params.rl();
};

template <class fft_t> double SimpleGrid<fft_t>::k_max() const {
    double d = (2.0 * M_PI)/m_params.rl();
    double ng = m_dist.ng();
    return sqrt(3.0 * (ng/2.0)* (ng/2.0) * d * d);
};

template <class fft_t> std::vector<double> SimpleGrid<fft_t>::bin(int nbins) const {
    int blockSize = BLOCKSIZE;
    int numBlocks = (m_size + (blockSize - 1)) / blockSize;

    double rl = m_params.rl();
    double k_delta = ((k_max()-k_min())/((double)nbins));

    double2* d_bins; gpu_allocator.alloc(&d_bins,sizeof(double2) * nbins);
    launch_bin_power(m_d_grid,d_bins, k_min(), k_delta, nbins, rl, m_dist, numBlocks, blockSize);

    double2* h_bins; cpu_allocator.alloc(&h_bins, sizeof(double2) * nbins);
    gpuMemcpy(h_bins,d_bins,sizeof(double2) * nbins,gpuMemcpyDeviceToHost);

    gpu_allocator.free(d_bins);

    std::vector<double> out;
    out.resize(nbins);

    for (int i = 0; i < nbins; i++){
        out[i] = h_bins[i].x / h_bins[i].y;
    }
    cpu_allocator.free(h_bins);
    return out;
};

template class SimpleGrid<complexDoubleDevice>;
template class SimpleGrid<complexFloatDevice>;