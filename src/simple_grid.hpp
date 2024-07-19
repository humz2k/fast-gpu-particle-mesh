#ifndef _FGPM_GRID_HPP_
#define _FGPM_GRID_HPP_

#include "mpi_distribution.hpp"
#include "serial_fft.hpp"
#include "simulation.hpp"

/**
 * @class SimpleGrid
 * @brief Manages grid data and operations in the simulation.
 *
 * The SimpleGrid class provides an implementation of the Grid class for
 * managing grid data, performing operations such as solving equations,
 * calculating gradients, and interacting with particles within the simulation.
 *
 * @tparam fft_t The type used for FFT operations (e.g., complexDoubleDevice).
 */
template <class fft_t> class SimpleGrid : public Grid {
  private:
    int m_ng;               ///< The size of the grid.
    size_t m_size;          ///< The total size of the grid.
    const Params& m_params; ///< Reference to the simulation parameters.
    float3* m_d_grad;       ///< Pointer to the gradient data on the device.
    fft_t* m_d_grid;        ///< Pointer to the grid data on the device.
    fft_t* m_d_x; ///< Pointer to the x-component of the transformed grid data.
    fft_t* m_d_y; ///< Pointer to the y-component of the transformed grid data.
    fft_t* m_d_z; ///< Pointer to the z-component of the transformed grid data.
    SerialFFT<fft_t> fft; ///< FFT object for performing Fourier transforms.
    MPIDist m_dist; ///< MPI distribution object for handling grid distribution.

  public:
    /**
     * @brief Constructs a SimpleGrid object with the given parameters and grid
     * size.
     *
     * @param params The simulation parameters.
     * @param ng The size of the grid.
     */
    SimpleGrid(const Params& params, int ng);

    /**
     * @brief Destructor for SimpleGrid.
     */
    ~SimpleGrid();

    /**
     * @brief Solves for rho on the grid.
     */
    void solve();

    /**
     * @brief Solves for grad rho on the grid.
     */
    void solve_gradient();

    /**
     * @brief Assigns particle positions to the grid using Cloud-In-Cell (CIC)
     * method.
     *
     * @param particles The particles to be assigned to the grid.
     */
    void CIC(const Particles<float3>& particles);

    /**
     * @brief Generates Fourier amplitudes for the grid using cosmological
     * parameters.
     *
     * @param cosmo The cosmological parameters.
     */
    void generate_fourier_amplitudes(Cosmo& cosmo);

    /**
     * @brief Generates displacement initial conditions for the grid.
     *
     * @param cosmo The cosmological parameters.
     * @param ts The time stepper used in the simulation.
     * @param particles The particles whose displacement initial conditions are
     * to be generated.
     */
    void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                  Particles<float3>& particles);

    /**
     * @brief Returns the MPI distribution of the grid.
     *
     * @return The MPI distribution.
     */
    MPIDist dist() const;

    /**
     * @brief Bins rho.
     *
     * @param nbins The number of bins to use.
     * @return The binned power spectrum.
     */
    std::vector<double> bin(int nbins) const;

    /**
     * @brief Returns the minimum k value.
     *
     * @return The minimum k value.
     */
    double k_min() const;

    /**
     * @brief Returns the maximum k value.
     *
     * @return The maximum k value.
     */
    double k_max() const;

    /**
     * @brief Performs a forward FFT on the grid.
     */
    void forward();

    /**
     * @brief Performs a backward FFT on the grid.
     */
    void backward();

    /**
     * @brief Returns the gradient of the grid.
     *
     * @return Pointer to the gradient of the grid.
     */
    const float3* grad() const;
};

#endif