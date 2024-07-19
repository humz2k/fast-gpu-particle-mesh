#include "serial_grid.hpp"
#include "allocators.hpp"
#include "common.hpp"
#include "initializer.hpp"
#include "particle_actions.hpp"
#include "pk_bins.hpp"
#include "solver.hpp"

template <class fft_t>
SerialGrid<fft_t>::SerialGrid(const Params& params, int ng)
    : m_ng(ng), m_params(params), fft(m_ng), m_dist(ng) {
    m_size = m_ng * m_ng * m_ng;

    gpu_allocator.alloc(&m_d_grad, m_size * sizeof(float3));
    gpu_allocator.alloc(&m_d_grid, m_size * sizeof(fft_t));
}

template <class fft_t> SerialGrid<fft_t>::~SerialGrid() {
    gpu_allocator.free(m_d_grad);
    gpu_allocator.free(m_d_grid);
}

template <class fft_t>
void SerialGrid<fft_t>::CIC(const Particles<float3>& particles) {
    int n_particles = particles.nlocal();
    int blockSize = BLOCKSIZE;
    int numBlocks = (n_particles + (blockSize - 1)) / blockSize;

    float gpscale = ((float)m_params.ng()) / ((float)m_params.np());
    float mass = gpscale * gpscale * gpscale;

    launch_CIC_kernel(m_d_grid, particles.pos(), n_particles, mass, m_dist,
                      numBlocks, blockSize);
}

template <class fft_t>
void SerialGrid<fft_t>::generate_fourier_amplitudes(Cosmo& cosmo) {
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

template <class fft_t> MPIDist SerialGrid<fft_t>::dist() const {
    return m_dist;
}

template <class fft_t> double SerialGrid<fft_t>::k_min() const {
    return (2.0 * M_PI) / m_params.rl();
}

template <class fft_t> double SerialGrid<fft_t>::k_max() const {
    double d = (2.0 * M_PI) / m_params.rl();
    double ng = m_dist.ng();
    return sqrt(3.0 * (ng / 2.0) * (ng / 2.0) * d * d);
}

template <class fft_t> void SerialGrid<fft_t>::forward() {
    fft.forward(m_d_grid);
}

template <class fft_t> void SerialGrid<fft_t>::backward() {
    fft.backward(m_d_grid);
}

template <class fft_t> void SerialGrid<fft_t>::forward(fft_t* ptr) {
    fft.forward(ptr);
}

template <class fft_t> void SerialGrid<fft_t>::backward(fft_t* ptr) {
    fft.backward(ptr);
}

template <class fft_t> const float3* SerialGrid<fft_t>::grad() const {
    return m_d_grad;
}

template <class fft_t> float3* SerialGrid<fft_t>::grad() { return m_d_grad; }

template <class fft_t> size_t SerialGrid<fft_t>::size() const { return m_size; }

template <class fft_t> const Params& SerialGrid<fft_t>::params() const {
    return m_params;
}

template <class fft_t> fft_t* SerialGrid<fft_t>::grid() { return m_d_grid; }

template <class fft_t>
std::vector<double> SerialGrid<fft_t>::bin(int nbins) const {
    int blockSize = BLOCKSIZE;
    int numBlocks = (m_size + (blockSize - 1)) / blockSize;

    double rl = m_params.rl();
    double k_delta = ((k_max() - k_min()) / ((double)nbins));

    double2* d_bins;
    gpu_allocator.alloc(&d_bins, sizeof(double2) * nbins);
    launch_bin_power(m_d_grid, d_bins, k_min(), k_delta, nbins, rl, m_dist,
                     numBlocks, blockSize);

    double2* h_bins;
    cpu_allocator.alloc(&h_bins, sizeof(double2) * nbins);
    gpuMemcpy(h_bins, d_bins, sizeof(double2) * nbins, gpuMemcpyDeviceToHost);

    gpu_allocator.free(d_bins);

    std::vector<double> out;
    out.resize(nbins);

    for (int i = 0; i < nbins; i++) {
        out[i] = h_bins[i].x / h_bins[i].y;
    }
    cpu_allocator.free(h_bins);
    return out;
}

template class SerialGrid<complexDoubleDevice>;
template class SerialGrid<complexFloatDevice>;