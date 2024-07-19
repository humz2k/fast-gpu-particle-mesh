#ifndef _FGPM_SIMPLE_PARTICLES_HPP_
#define _FGPM_SIMPLE_PARTICLES_HPP_

#include "gpu.hpp"
#include "simulation.hpp"

/**
 * @class SimpleParticles
 * @brief Manages particle data and interactions in the simulation.
 *
 * The SimpleParticles class provides an implementation of the Particles class
 * for managing particle data, including positions and velocities, and their
 * interactions within the simulation.
 */
class SimpleParticles : public Particles<float3> {
  private:
    const Params& m_params; ///< Reference to the simulation parameters.
    float3* m_pos;          ///< Pointer to the particle positions.
    float3* m_vel;          ///< Pointer to the particle velocities.

  public:
    /**
     * @brief Constructs a SimpleParticles object with the given parameters.
     *
     * @param params The simulation parameters.
     * @param cosmo The cosmological parameters.
     * @param ts The time stepper used in the simulation.
     */
    SimpleParticles(const Params& params, Cosmo& cosmo, Timestepper& ts);

    /**
     * @brief Destructor for SimpleParticles.
     */
    ~SimpleParticles();

    /**
     * @brief Updates the positions of the particles.
     *
     * @param ts The time stepper used in the simulation.
     * @param frac The fraction of the time step to advance.
     */
    void update_positions(Timestepper& ts, float frac);

    /**
     * @brief Updates the velocities of the particles based on the grid data.
     *
     * @param grid The grid data used to update particle velocities.
     * @param ts The time stepper used in the simulation.
     * @param frac The fraction of the time step to advance.
     */
    void update_velocities(const Grid& grid, Timestepper& ts, float frac);

    /**
     * @brief Returns the positions of the particles.
     *
     * @return Pointer to the positions of the particles.
     */
    float3* pos();

    /**
     * @brief Returns the positions of the particles (const version).
     *
     * @return Pointer to the positions of the particles.
     */
    const float3* pos() const;

    /**
     * @brief Returns the velocities of the particles.
     *
     * @return Pointer to the velocities of the particles.
     */
    float3* vel();

    /**
     * @brief Returns the velocities of the particles (const version).
     *
     * @return Pointer to the velocities of the particles.
     */
    const float3* vel() const;

    /**
     * @brief Dumps the particle data to a file (as csv).
     *
     * @param filename The name of the file to dump the data to.
     */
    void dump(std::string filename) const;

    /**
     * @brief Returns the number of local particles.
     *
     * @return The number of local particles.
     */
    int nlocal() const;
};

#endif