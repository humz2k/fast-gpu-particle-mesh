#include "memory_minimizing_grid.hpp"
#include "allocators.hpp"
#include "common.hpp"
#include "initializer.hpp"
#include "particle_actions.hpp"
#include "pk_bins.hpp"
#include "solver.hpp"
#include <cassert>

template <class fft_t>
MemoryMinGrid<fft_t>::MemoryMinGrid(const Params& params, int ng)
    : m_ng(ng), m_params(params), m_d_grad(NULL), m_d_grid(NULL), fft(m_ng),
      m_dist(ng) {
    m_size = m_ng * m_ng * m_ng;

    // gpu_allocator.alloc(&m_d_grad, m_size * sizeof(float3));
    // gpu_allocator.alloc(&m_d_grid, m_size * sizeof(fft_t));
}

template <class fft_t> MemoryMinGrid<fft_t>::~MemoryMinGrid() {
    if (m_d_grad)
        gpu_allocator.free(m_d_grad);
    if (m_d_grid)
        gpu_allocator.free(m_d_grid);
}

template <class fft_t> void MemoryMinGrid<fft_t>::solve_gradient() {

    assert(m_d_grid);

    int blockSize = BLOCKSIZE;
    int numBlocks = (m_dist.local_grid_size() + (blockSize - 1)) / blockSize;

    if (m_d_grad) {
        gpu_allocator.free(m_d_grad);
        m_d_grad = NULL;
    }

    fft_t* d_x;
    gpu_allocator.alloc(&d_x, m_size * sizeof(fft_t));
    fft_t* d_y;
    gpu_allocator.alloc(&d_y, m_size * sizeof(fft_t));
    fft_t* d_z;
    gpu_allocator.alloc(&d_z, m_size * sizeof(fft_t));

    fft.forward(m_d_grid);
    launch_kspace_solve_gradient(m_d_grid, d_x, d_y, d_z, m_dist, numBlocks,
                                 blockSize);

    gpu_allocator.free(m_d_grid);
    m_d_grid = NULL;

    fft.backward(d_x);
    fft.backward(d_y);
    fft.backward(d_z);

    if (!m_d_grad) {
        gpu_allocator.alloc(&m_d_grad, m_size * sizeof(float3));
    }

    launch_combine_vectors(m_d_grad, d_x, d_y, d_z, m_dist, numBlocks,
                           blockSize);

    gpu_allocator.free(d_x);
    gpu_allocator.free(d_y);
    gpu_allocator.free(d_z);
}

template <class fft_t> void MemoryMinGrid<fft_t>::solve() {}

template <class fft_t>
void MemoryMinGrid<fft_t>::CIC(const Particles<float3>& particles) {
    int n_particles = particles.nlocal();
    int blockSize = BLOCKSIZE;
    int numBlocks = (n_particles + (blockSize - 1)) / blockSize;

    float gpscale = ((float)m_params.ng()) / ((float)m_params.np());
    float mass = gpscale * gpscale * gpscale;

    if (m_d_grad) {
        gpu_allocator.free(m_d_grad);
        m_d_grad = NULL;
    }

    if (!m_d_grid) {
        gpu_allocator.alloc(&m_d_grid, m_size * sizeof(fft_t));
    }

    launch_CIC_kernel(m_d_grid, particles.pos(), n_particles, mass, m_dist,
                      numBlocks, blockSize);
}

template <class fft_t>
void MemoryMinGrid<fft_t>::generate_fourier_amplitudes(Cosmo& cosmo) {
    LOG_INFO("generating fourier amplitudes");

    int blockSize = BLOCKSIZE;
    int numBlocks = (m_size + (blockSize - 1)) / blockSize;

    if (!m_d_grid) {
        gpu_allocator.alloc(&m_d_grid, m_size * sizeof(fft_t));
    }

    if (m_d_grad) {
        gpu_allocator.free(m_d_grad);
        m_d_grad = NULL;
    }

    launch_generate_real_random(m_d_grid, m_params.seed(), m_dist, numBlocks,
                                blockSize);

    fft.forward(m_d_grid);

    launch_scale_amplitudes_by_power_spectrum(m_d_grid, cosmo.initial_pk(),
                                              m_params.rl(), m_dist, numBlocks,
                                              blockSize);
}

template <class fft_t>
void MemoryMinGrid<fft_t>::generate_displacement_ic(
    Cosmo& cosmo, Timestepper& ts, Particles<float3>& particles) {
    LOG_INFO("generating displacement ic");

    int blockSize = BLOCKSIZE;
    int numBlocks = (m_size + (blockSize - 1)) / blockSize;

    generate_fourier_amplitudes(cosmo);
    float delta = cosmo.delta(ts.z());
    ts.reverse_half_step();
    float fscal = ts.fscal();
    float dot_delta = cosmo.dot_delta(ts.z());
    ts.advance_half_step();

    fft_t* d_x;
    gpu_allocator.alloc(&d_x, m_size * sizeof(fft_t));
    fft_t* d_y;
    gpu_allocator.alloc(&d_y, m_size * sizeof(fft_t));
    fft_t* d_z;
    gpu_allocator.alloc(&d_z, m_size * sizeof(fft_t));

    launch_transform_density_field(m_d_grid, d_x, d_y, d_z, delta,
                                   m_params.rl(), ts.a(), m_dist, numBlocks,
                                   blockSize);

    gpu_allocator.free(m_d_grid);
    m_d_grid = NULL;

    fft.backward(d_x);
    fft.backward(d_y);
    fft.backward(d_z);

    if (!m_d_grad) {
        gpu_allocator.alloc(&m_d_grad, m_size * sizeof(float3));
    }

    launch_combine_density_vectors(m_d_grad, d_x, d_y, d_z, m_dist, numBlocks,
                                   blockSize);

    gpu_allocator.free(d_x);
    gpu_allocator.free(d_y);
    gpu_allocator.free(d_z);

    launch_place_particles(particles.pos(), particles.vel(), m_d_grad, delta,
                           dot_delta, m_params.rl(), ts.a(), ts.deltaT(), fscal,
                           m_params.ng(), m_dist, numBlocks, blockSize);

    gpu_allocator.free(m_d_grad);
    m_d_grad = NULL;
}

template <class fft_t> MPIDist MemoryMinGrid<fft_t>::dist() const {
    return m_dist;
};

template <class fft_t> double MemoryMinGrid<fft_t>::k_min() const {
    return (2.0 * M_PI) / m_params.rl();
};

template <class fft_t> double MemoryMinGrid<fft_t>::k_max() const {
    double d = (2.0 * M_PI) / m_params.rl();
    double ng = m_dist.ng();
    return sqrt(3.0 * (ng / 2.0) * (ng / 2.0) * d * d);
};

template <class fft_t> void MemoryMinGrid<fft_t>::forward() {
    fft.forward(m_d_grid);
};

template <class fft_t> void MemoryMinGrid<fft_t>::backward() {
    fft.backward(m_d_grid);
};

template <class fft_t> const float3* MemoryMinGrid<fft_t>::grad() const {
    assert(m_d_grad);
    return m_d_grad;
};

template <class fft_t>
std::vector<double> MemoryMinGrid<fft_t>::bin(int nbins) const {

    assert(m_d_grid);

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
};

template class MemoryMinGrid<complexDoubleDevice>;
template class MemoryMinGrid<complexFloatDevice>;