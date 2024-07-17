#ifndef _FGPM_SIMULATION_HPP_
#define _FGPM_SIMULATION_HPP_

#include "cosmo.hpp"
#include "mpi_distribution.hpp"
#include "params.hpp"
#include "timestepper.hpp"
#include <cuda_runtime.h>
#include <cufft.h>

class Particles;
class Grid;
template <class T> class FFT;

/**
 * @brief Template class for performing Fast Fourier Transforms (FFT).
 *
 * This template class provides an interface for performing forward and backward
 * FFTs. It can be specialized for different types (e.g., complexDoubleDevice,
 * complexFloatDevice).
 *
 * @tparam T The type of the elements in the FFT (e.g., complexDoubleDevice).
 */
template <class T> class FFT {
  public:
    /**
     * @brief Constructs an FFT object with a specified grid size.
     *
     * @param ng The size of the grid.
     */
    FFT(int ng){};

    /**
     * @brief Default constructor for FFT.
     */
    FFT(){};

    /**
     * @brief Virtual destructor for FFT.
     */
    virtual ~FFT(){};

    /**
     * @brief Performs a forward FFT.
     *
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     */
    virtual void forward(T* in, T* out) = 0;

    /**
     * @brief Performs a backward FFT.
     *
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     */
    virtual void backward(T* in, T* out) = 0;

    /**
     * @brief Performs a forward FFT in-place.
     *
     * @param in Pointer to the data to transform.
     */
    virtual void forward(T* in) = 0;

    /**
     * @brief Performs a backward FFT in-place.
     *
     * @param in Pointer to the data to transform.
     */
    virtual void backward(T* in) = 0;
};

/**
 * @class Grid
 * @brief Represents the simulation grid and manages related operations.
 *
 * The Grid class provides an interface for various grid-related operations,
 * such as solving equations, calculating gradients, and interacting with
 * particles.
 */
class Grid {
  public:
    /**
     * @brief Constructs a Grid object with the given parameters and grid size.
     *
     * @param params The simulation parameters.
     * @param ng The size of the grid.
     */
    Grid(const Params& params, int ng){};

    /**
     * @brief Default constructor for Grid.
     */
    Grid(){};

    /**
     * @brief Virtual destructor for Grid.
     */
    virtual ~Grid(){};

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
    virtual void CIC(const Particles& particles) = 0;

    /**
     * @brief Generates Fourier amplitudes for the grid using cosmological
     * parameters.
     *
     * @param cosmo The cosmological parameters.
     */
    virtual void generate_fourier_amplitudes(Cosmo& cosmo) = 0;

    /**
     * @brief Generates displacement initial conditions gradient.
     *
     * @param cosmo The cosmological parameters.
     * @param ts The time stepper used in the simulation.
     * @return Pointer to the generated displacement gradient.
     */
    virtual const float3* generate_displacement_ic_grad(Cosmo& cosmo,
                                                        Timestepper& ts) = 0;

    /**
     * @brief Returns the `MPIDist` of the grid.
     *
     * @return The `MPIDist`.
     */
    virtual MPIDist dist() const = 0;

    /**
     * @brief Bins rho.
     *
     * @return The binned power spectrum.
     */
    virtual std::vector<double> bin(int nbins) const = 0;

    /**
     * @brief Returns the minimum k value.
     *
     * @return The minimum k value.
     */
    virtual double k_min() const = 0;

    /**
     * @brief Returns the maximum k value.
     *
     * @return The maximum k value.
     */
    virtual double k_max() const = 0;
};

/**
 * @class Particles
 * @brief Represents the particles in the simulation and manages their
 * interactions.
 *
 * The Particles class provides an interface for particle-related operations,
 * such as updating positions and velocities based on grid data.
 */
class Particles {
  public:
    /**
     * @brief Constructs a Particles object with the given parameters.
     *
     * @param params The simulation parameters.
     */
    Particles(const Params& params){};

    /**
     * @brief Default constructor for Particles.
     */
    Particles(){};

    /**
     * @brief Virtual destructor for Particles.
     */
    virtual ~Particles(){};

    /**
     * @brief Updates the positions of the particles.
     */
    virtual void update_positions() = 0;

    /**
     * @brief Updates the velocities of the particles based on the grid data.
     *
     * @param grid The grid data used to update particle velocities.
     */
    virtual void update_velocities(const Grid& grid) = 0;
};

#endif