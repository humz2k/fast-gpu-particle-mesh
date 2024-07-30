#ifndef _FGPM_SERIAL_GRID_HPP_
#define _FGPM_SERIAL_GRID_HPP_

#include "allocators.hpp"
#include "common.hpp"
#include "initializer.hpp"
#include "mpi_distribution.hpp"
#include "particle_actions.hpp"
#include "pk_bins.hpp"
#include "serial_fft.hpp"
#include "simulation.hpp"
#include "solver.hpp"

/**
 * @class SerialGrid
 * @brief Manages grid data and operations in the simulation.
 *
 * The SerialGrid class provides an implementation of the Grid class for
 * managing grid data, performing operations such as solving equations,
 * calculating gradients, and interacting with particles within the simulation.
 * @tparam fft_wrapper_t The object used for FFT operations (e.g., SerialFFT).
 * @tparam fft_t The type used for FFT operations (e.g., complexDoubleDevice).
 */
template <template <class> class fft_wrapper_t, class fft_t>
class SerialGrid : public Grid {
  private:
    int m_ng;               ///< The size of the grid.
    size_t m_size;          ///< The total size of the grid.
    const Params& m_params; ///< Reference to the simulation parameters.
    float3* m_d_grad;       ///< Pointer to the gradient data on the device.
    fft_t* m_d_grid;        ///< Pointer to the grid data on the device.
    SerialFFT<fft_t> fft;   ///< FFT object for performing Fourier transforms.
    MPIDist m_dist; ///< MPI distribution object for handling grid distribution.

  protected:
    /**
     * @brief Gets the raw GPU pointer to the grid.
     *
     * @return The raw GPU pointer to the grid.
     */
    fft_t* grid() { return m_d_grid; }

  public:
    /**
     * @brief Constructs a SerialGrid object with the given parameters and grid
     * size.
     *
     * @param params The simulation parameters.
     * @param ng The size of the grid.
     */
    SerialGrid(const Params& params, int ng)
        : m_ng(ng), m_params(params), fft(m_ng), m_dist(ng) {
        m_size = m_ng * m_ng * m_ng;

        gpu_allocator.alloc(&m_d_grad, m_size * sizeof(float3));
        gpu_allocator.alloc(&m_d_grid, m_size * sizeof(fft_t));
    }

    /**
     * @brief Destructor for SerialGrid.
     */
    ~SerialGrid() {
        gpu_allocator.free(m_d_grad);
        gpu_allocator.free(m_d_grid);
    }

    /**
     * @brief Solves for rho on the grid.
     */
    virtual void solve() = 0;

    /**
     * @brief Solves for grad rho on the grid.
     */
    virtual void solve_gradient() = 0;

    /**
     * @brief Assigns particle positions to the grid using Cloud-In-Cell (CIC)
     * method.
     *
     * @param particles The particles to be assigned to the grid.
     */
    void CIC(const Particles<float3>& particles) {
        int n_particles = particles.nlocal();
        int blockSize = BLOCKSIZE;
        int numBlocks = (n_particles + (blockSize - 1)) / blockSize;

        float gpscale = ((float)m_params.ng()) / ((float)m_params.np());
        float mass = gpscale * gpscale * gpscale;

        launch_CIC_kernel(m_d_grid, particles.pos(), n_particles, mass, m_dist,
                          numBlocks, blockSize);
    }

    /**
     * @brief Generates Fourier amplitudes for the grid using cosmological
     * parameters.
     *
     * @param cosmo The cosmological parameters.
     */
    void generate_fourier_amplitudes(Cosmo& cosmo) {
        LOG_INFO("generating fourier amplitudes");

        int blockSize = BLOCKSIZE;
        int numBlocks = (m_size + (blockSize - 1)) / blockSize;

        launch_generate_real_random(m_d_grid, m_params.seed(), m_dist,
                                    numBlocks, blockSize);

        fft.forward(m_d_grid);

        launch_scale_amplitudes_by_power_spectrum(m_d_grid, cosmo.initial_pk(),
                                                  m_params.rl(), m_dist,
                                                  numBlocks, blockSize);
    }

    /**
     * @brief Generates displacement initial conditions for the grid.
     *
     * @param cosmo The cosmological parameters.
     * @param ts The time stepper used in the simulation.
     * @param particles The particles whose displacement initial conditions are
     * to be generated.
     */
    virtual void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                          Particles<float3>& particles) = 0;

    /**
     * @brief Returns the MPI distribution of the grid.
     *
     * @return The MPI distribution.
     */
    MPIDist dist() const { return m_dist; }

    /**
     * @brief Returns the size of the grid.
     *
     * @return The size of the grid.
     */
    size_t size() const { return m_size; }

    /**
     * @brief Returns the params object of the grid.
     *
     * @return The params object.
     */
    const Params& params() const { return m_params; }

    /**
     * @brief Bins rho.
     *
     * @param nbins The number of bins to use.
     * @return The binned power spectrum.
     */
    std::vector<double> bin(int nbins) const {
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
        gpuMemcpy(h_bins, d_bins, sizeof(double2) * nbins,
                  gpuMemcpyDeviceToHost);

        gpu_allocator.free(d_bins);

        std::vector<double> out;
        out.resize(nbins);

        for (int i = 0; i < nbins; i++) {
            out[i] = h_bins[i].x / h_bins[i].y;
        }
        cpu_allocator.free(h_bins);
        return out;
    }

    /**
     * @brief Returns the minimum k value.
     *
     * @return The minimum k value.
     */
    double k_min() const { return (2.0 * M_PI) / m_params.rl(); }

    /**
     * @brief Returns the maximum k value.
     *
     * @return The maximum k value.
     */
    double k_max() const {
        double d = (2.0 * M_PI) / m_params.rl();
        double ng = m_dist.ng();
        return sqrt(3.0 * (ng / 2.0) * (ng / 2.0) * d * d);
    }

    /**
     * @brief Performs a forward FFT on the grid.
     */
    void forward() { fft.forward(m_d_grid); }

    /**
     * @brief Performs a backward FFT on the grid.
     */
    void backward() { fft.backward(m_d_grid); }

    /**
     * @brief Performs a forward FFT in place.
     */
    void forward(fft_t* ptr) { fft.forward(ptr); }

    /**
     * @brief Performs a backward FFT in place.
     */
    void backward(fft_t* ptr) { fft.backward(ptr); }

    /**
     * @brief Returns raw GPU pointer to gradient of the grid.
     *
     * @return The raw GPU pointer to the gradient of the grid.
     */
    const float3* grad() const { return m_d_grad; }

    /**
     * @brief Returns raw GPU pointer to gradient of the grid.
     *
     * @return The raw GPU pointer to the gradient of the grid.
     */
    float3* grad() { return m_d_grad; }
};

#endif