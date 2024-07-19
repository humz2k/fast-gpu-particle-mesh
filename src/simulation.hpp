#ifndef _FGPM_SIMULATION_HPP_
#define _FGPM_SIMULATION_HPP_

#include "cosmo.hpp"
#include "mpi_distribution.hpp"
#include "params.hpp"
#include "timestepper.hpp"
#include "event_logger.hpp"
#include <cuda_runtime.h>
#include <cufft.h>

template <class T> class Particles;
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
    virtual void CIC(const Particles<float3>& particles) = 0;

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
    virtual void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                          Particles<float3>& particles) = 0;

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

    /**
     * @brief Performs a forward FFT on the grid.
     */
    virtual void forward() = 0;

    /**
     * @brief Performs a backward FFT on the grid.
     */
    virtual void backward() = 0;

    /**
     * @brief Returns the gradient of the grid.
     *
     * @return Pointer to the gradient of the grid.
     */
    virtual const float3* grad() const = 0;

    /**
     * @brief Returns raw GPU pointer to gradient of the grid.
     *
     * @return The raw GPU pointer to the gradient of the grid.
     */
    virtual float3* grad() = 0;
};

/**
 * @class Particles
 * @brief Represents the particles in the simulation and manages their
 * interactions.
 *
 * The Particles class provides an interface for particle-related operations,
 * such as updating positions and velocities based on grid data.
 *
 *  * @tparam T The type of the elements of the particles (e.g. `float3`)
 */
template <class T> class Particles {
  public:
    /**
     * @brief Constructs a Particles object with the given parameters.
     *
     * @param params The simulation parameters.
     */
    Particles(const Params& params, Cosmo& cosmo, Timestepper& ts){};

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
    virtual void update_positions(Timestepper& ts, float frac) = 0;

    /**
     * @brief Updates the velocities of the particles based on the grid data.
     *
     * @param grid The grid data used to update particle velocities.
     */
    virtual void update_velocities(const Grid& grid, Timestepper& ts,
                                   float frac) = 0;

    /**
     * @brief Returns the positions of the particles.
     *
     * @return Pointer to the positions of the particles.
     */
    virtual T* pos() = 0;

    /**
     * @brief Returns the positions of the particles (const version).
     *
     * @return Pointer to the positions of the particles.
     */
    virtual const T* pos() const = 0;

    /**
     * @brief Returns the velocities of the particles.
     *
     * @return Pointer to the velocities of the particles.
     */
    virtual T* vel() = 0;

    /**
     * @brief Returns the velocities of the particles (const version).
     *
     * @return Pointer to the velocities of the particles.
     */
    virtual const T* vel() const = 0;

    /**
     * @brief Dumps the particle data to a file (as csv).
     *
     * @param filename The name of the file to dump the data to.
     */
    virtual void dump(std::string filename) const = 0;

    /**
     * @brief Returns the number of local particles.
     *
     * @return The number of local particles.
     */
    virtual int nlocal() const = 0;

    /**
     * @brief Returns the params of the particles.
     *
     * @return The params of the particles.
     */
    virtual const Params& params() const = 0;
};

template<class ParticleType, class GridType>
void run_simulation(std::string params_file){
    events.timers["dtot"].start();

    gpuCall(gpuFree(0));

    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    events.timers["dinit"].start();

    Params params(params_file);
    Timestepper ts(params);
    Cosmo cosmo(params);
    cosmo.initial_pk().to_csv("test.csv");

    ParticleType particles(params, cosmo, ts);

    events.timers["dinit"].end();

    GridType grid(params, params.ng());
    grid.CIC(particles);
    grid.forward();
    PowerSpectrum ic_power(grid, params.pk_n_bins());
    ic_power.to_csv("test2.csv");

    for (int step = 0; step < params.nsteps(); step++) {
        LOG_MINIMAL("STEP %d", step);
        events.timers["dstep"].start();

        events.timers["dpos"].start();
        particles.update_positions(ts, 0.5f);
        events.timers["dpos"].end();

        events.timers["dcic"].start();
        grid.CIC(particles);
        events.timers["dcic"].end();

        events.timers["dgrad"].start();
        grid.solve_gradient();
        events.timers["dgrad"].end();

        ts.advance_half_step();

        events.timers["dvel"].start();
        particles.update_velocities(grid, ts, 1.0f);
        events.timers["dvel"].end();

        ts.advance_half_step();

        events.timers["dpos"].start();
        particles.update_positions(ts, 0.5f);
        events.timers["dpos"].end();

        if (params.pk_dump(step)) {
            LOG_MINIMAL("dumping pk");
            events.timers["dpk"].start();
            grid.CIC(particles);
            grid.forward();
            PowerSpectrum(grid, params.pk_n_bins())
                .to_csv("steps/step" + std::to_string(step) + ".csv");
            events.timers["dpk"].end();
        }

        events.timers["dstep"].end();
    }

    grid.CIC(particles);
    grid.forward();
    PowerSpectrum power(grid, params.pk_n_bins());
    power.to_csv("final.csv");

    LOG_MINIMAL("done!");

    events.timers["dtot"].end();
}

#endif