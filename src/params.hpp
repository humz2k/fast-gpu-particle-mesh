#ifndef _FGPM_PARAMS_HPP_
#define _FGPM_PARAMS_HPP_

#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class Params
 * @brief Manages simulation parameters and configuration settings.
 *
 * The Params class is responsible for storing and managing the various
 * parameters required for running a simulation. It includes methods to access
 * these parameters and to output the configuration.
 */
class Params {
  private:
    std::string m_params_filename; /**< Filename for parameter configuration. */

    int m_ng = 0;      /**< Number of grid cells. */
    int m_np = 0;      /**< Number of particles. */
    double m_rl = 0.0; /**< Simulation box size. */
    int m_seed = 0;    /**< Random seed for simulations. */

    double m_z_ini = 200.0; /**< Initial redshift. */
    double m_z_fin = 0.0;   /**< Final redshift. */
    int m_nsteps = 625;     /**< Number of steps in the simulation. */

    double m_omega_cdm = 0.22; /**< Density parameter for cold dark matter. */
    double m_deut = 0.02258;   /**< Density parameter for deuterium. */
    double m_omega_nu = 0.0;   /**< Density parameter for neutrinos. */
    double m_hubble = 0.71;    /**< Hubble parameter. */
    double m_ss8 = 0.8;        /**< Normalization of the power spectrum. */
    double m_ns = 0.963;       /**< Spectral idx of the primordial PK. */
    double m_w_de = -1.0;   /**< Equation of state parameter for dark energy. */
    double m_wa_de = 0.0;   /**< Time variation of eqn of state param for DE. */
    double m_T_cmb = 2.726; /**< Temp of the CMB. */
    double m_neff_massless = 3.04; /**< Eff. n of massless neutrino species. */
    double m_neff_massive = 0.0;   /**< Eff. n of massive neutrino species. */

    double m_omega_baryon; /**< Density parameter for baryons. */
    double m_omega_cb; /**< Combined density parameter for CDM and baryons. */
    double m_omega_matter;    /**< Density parameter for matter. */
    double m_omega_radiation; /**< Density parameter for radiation. */
    double m_f_nu_massless;   /**< Fraction of massless neutrinos. */
    double m_f_nu_massive;    /**< Fraction of massive neutrinos. */

    int m_pk_folds = 0; /**< Number of folds for power spectrum calculation. */

    std::string m_output_prefix = "run"; /**< Prefix for output files. */
    std::string m_ipk;                   /**< Input power spectrum file. */

    std::unordered_map<int, bool> m_pk_dumps;       /**< Steps to dump PK. */
    std::unordered_map<int, bool> m_particle_dumps; /**< Steps to dump parts. */

  public:
    /**
     * @brief Constructs a Params object and initializes it with parameters from
     * the given file.
     * @param filename The name of the parameter configuration file.
     */
    Params(std::string filename);

    /**
     * @brief Outputs the current parameter settings.
     */
    void dump();

    /**
     * @brief Gets the number of grid points.
     * @return The number of grid points.
     */
    int ng() const;

    /**
     * @brief Gets the number of particles.
     * @return The number of particles.
     */
    int np() const;

    /**
     * @brief Gets the simulation box size.
     * @return The simulation box size.
     */
    double rl() const;

    /**
     * @brief Gets the random seed for simulations.
     * @return The random seed for simulations.
     */
    int seed() const;

    /**
     * @brief Gets the initial redshift.
     * @return The initial redshift.
     */
    double z_ini() const;

    /**
     * @brief Gets the final redshift.
     * @return The final redshift.
     */
    double z_fin() const;

    /**
     * @brief Gets the number of steps in the simulation.
     * @return The number of steps in the simulation.
     */
    int nsteps() const;

    /**
     * @brief Gets the equation of state parameter for dark energy.
     * @return The equation of state parameter for dark energy.
     */
    double w_de() const;

    /**
     * @brief Gets the time variation of the equation of state parameter for
     * dark energy.
     * @return The time variation of the equation of state parameter for dark
     * energy.
     */
    double wa_de() const;

    /**
     * @brief Gets the combined density parameter for cold dark matter and
     * baryons.
     * @return The combined density parameter for cold dark matter and baryons.
     */
    double omega_cb() const;

    /**
     * @brief Gets the fraction of massless neutrinos.
     * @return The fraction of massless neutrinos.
     */
    double f_nu_massless() const;

    /**
     * @brief Gets the fraction of massive neutrinos.
     * @return The fraction of massive neutrinos.
     */
    double f_nu_massive() const;

    /**
     * @brief Gets the density parameter for radiation.
     * @return The density parameter for radiation.
     */
    double omega_radiation() const;

    /**
     * @brief Gets the density parameter for matter.
     * @return The density parameter for matter.
     */
    double omega_matter() const;

    /**
     * @brief Gets the density parameter for neutrinos.
     * @return The density parameter for neutrinos.
     */
    double omega_nu() const;

    /**
     * @brief Gets the input power spectrum file.
     * @return The input power spectrum file.
     */
    const std::string& ipk() const;
};
#endif