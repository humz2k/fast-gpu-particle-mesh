#ifndef _FGPM_SERIAL_GRID_HPP_
#define _FGPM_SERIAL_GRID_HPP_

#include "mpi_distribution.hpp"
#include "serial_fft.hpp"
#include "simulation.hpp"

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
template <template <class> class fft_wrapper_t,class fft_t> class SerialGrid : public Grid {
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
    fft_t* grid();

  public:
    /**
     * @brief Constructs a SerialGrid object with the given parameters and grid
     * size.
     *
     * @param params The simulation parameters.
     * @param ng The size of the grid.
     */
    SerialGrid(const Params& params, int ng);

    /**
     * @brief Destructor for SerialGrid.
     */
    ~SerialGrid();

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
    virtual void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                          Particles<float3>& particles) = 0;

    /**
     * @brief Returns the MPI distribution of the grid.
     *
     * @return The MPI distribution.
     */
    MPIDist dist() const;

    /**
     * @brief Returns the size of the grid.
     *
     * @return The size of the grid.
     */
    size_t size() const;

    /**
     * @brief Returns the params object of the grid.
     *
     * @return The params object.
     */
    const Params& params() const;

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
     * @brief Performs a forward FFT in place.
     */
    void forward(fft_t* ptr);

    /**
     * @brief Performs a backward FFT in place.
     */
    void backward(fft_t* ptr);

    /**
     * @brief Returns raw GPU pointer to gradient of the grid.
     *
     * @return The raw GPU pointer to the gradient of the grid.
     */
    const float3* grad() const;

    /**
     * @brief Returns raw GPU pointer to gradient of the grid.
     *
     * @return The raw GPU pointer to the gradient of the grid.
     */
    float3* grad();
};

#endif