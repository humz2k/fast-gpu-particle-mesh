#ifndef _FGPM_TIMESTEPPER_HPP_
#define _FGPM_TIMESTEPPER_HPP_

#include "params.hpp"

/**
 * @class Timestepper
 * @brief Manages the progression of simulation time steps.
 *
 * The Timestepper class is responsible for advancing and reversing simulation
 * time steps, and managing related parameters such as scale factor and
 * redshift.
 */
class Timestepper {
  private:
    const Params& m_params; /**< Reference to the simulation parameters. */
    double m_a;             /**< Scale factor. */
    double m_z;             /**< Redshift. */
    double m_deltaT;        /**< Time step size. */
  public:
    /**
     * @brief Constructs a Timestepper object with the given parameters.
     * @param params The simulation parameters.
     */
    Timestepper(const Params& params);

    /**
     * @brief Advances the simulation by half a time step.
     */
    void advance_half_step();

    /**
     * @brief Reverses the simulation by half a time step.
     */
    void reverse_half_step();

    /**
     * @brief Gets the current scale factor.
     * @return The current scale factor.
     */
    double a() const;

    /**
     * @brief Sets a new scale factor.
     * @param new_a The new scale factor to set.
     */
    void a(double new_a);

    /**
     * @brief Gets the current redshift.
     * @return The current redshift.
     */
    double z() const;

    /**
     * @brief Sets a new redshift.
     * @param new_z The new redshift to set.
     */
    void z(double new_z);

    /**
     * @brief Gets the current time step size.
     * @return The current time step size.
     */
    double deltaT() const;

    /**
     * @brief Calculates the rate of change of the scale factor.
     * @return The rate of change of the scale factor.
     */
    double adot() const;

    /**
     * @brief Calculates a scaling factor for the simulation.
     * @return The scaling factor.
     */
    double fscal() const;

    /**
     * @brief Calculates the density parameter for massive neutrinos.
     * @return The density parameter for massive neutrinos.
     */
    double omega_nu_massive() const;
};

#endif
