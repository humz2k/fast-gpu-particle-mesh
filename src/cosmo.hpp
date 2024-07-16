#ifndef _FGPM_COSMO_HPP_
#define _FGPM_COSMO_HPP_

#include "params.hpp"
#include "power_spectrum.hpp"
#include <string>
#include <vector>

/**
 * @class Cosmo
 * @brief Manages cosmological calculations and evolution.
 *
 * The Cosmo class handles various cosmological calculations, such as growth
 * factors, integration of differential equations, and interactions with the
 * power spectrum.
 */
class Cosmo {
  private:
    double m_last_growth_z = -1; /**< Last z for growth factor calc. */
    double m_gf;                 /**< Growth factor. */
    double m_g_dot;              /**< Time derivative of the growth factor. */

    const Params& m_params;     /**< Reference to simulation parameters. */
    PowerSpectrum m_initial_pk; /**< Initial power spectrum. */

    /**
     * @brief Performs the Cash-Karp Runge-Kutta integration step.
     *
     * This function integrates the given differential equations using the
     * Cash-Karp Runge-Kutta method.
     *
     * @param y Array of dependent variable values.
     * @param dydx Array of derivatives of y.
     * @param n Number of variables.
     * @param x Independent variable.
     * @param h Step size.
     * @param yout Output array of new dependent variable values.
     * @param yerr Output array of error estimates.
     * @param derivs Member function pointer to the derivative calculation
     * function.
     */
    void rkck(double* y, double* dydx, int n, double x, double h, double* yout,
              double* yerr, void (Cosmo::*derivs)(double, double*, double*));

    /**
     * @brief Performs the adaptive step size control for the Runge-Kutta
     * integration.
     *
     * This function adjusts the step size during the integration process to
     * maintain the desired accuracy.
     *
     * @param y Array of dependent variable values.
     * @param dydx Array of derivatives of y.
     * @param n Number of variables.
     * @param x Pointer to the independent variable.
     * @param htry Initial step size.
     * @param eps Desired accuracy.
     * @param yscal Array of scaling factors for error calculation.
     * @param hdid Output variable for the step size actually used.
     * @param hnext Output variable for the next step size.
     * @param feval Pointer to the function evaluation counter.
     * @param derivs Member function pointer to the derivative calculation
     * function.
     */
    void rkqs(double* y, double* dydx, int n, double* x, double htry,
              double eps, double* yscal, double* hdid, double* hnext,
              int* feval, void (Cosmo::*derivs)(double, double*, double*));

    /**
     * @brief Solves ordinary differential equations using adaptive step size
     * Runge-Kutta method.
     *
     * @param ystart Array of initial values of the dependent variables.
     * @param nvar Number of variables.
     * @param x1 Initial value of the independent variable.
     * @param x2 Final value of the independent variable.
     * @param eps Desired accuracy.
     * @param h1 Initial step size.
     * @param derivs Member function pointer to the derivative calculation
     * function.
     * @param print_stat Flag to print statistics during the solving process.
     */
    void odesolve(double* ystart, int nvar, double x1, double x2, double eps,
                  double h1, void (Cosmo::*derivs)(double, double*, double*),
                  bool print_stat);

    /**
     * @brief Calculates the derivatives for growth factor equations.
     *
     * @param a Scale factor.
     * @param y Array of dependent variable values.
     * @param dydx Output array of derivatives.
     */
    void growths(double a, double* y, double* dydx);

    /**
     * @brief Updates the growth factor for a given redshift.
     *
     * @param z Redshift value.
     */
    void update_growth_factor(double z);

    /**
     * @brief Calculates the density parameter for massive neutrinos.
     *
     * @param a Scale factor.
     * @return Density parameter for massive neutrinos.
     */
    double Omega_nu_massive(double a);

  public:
    /**
     * @brief Constructs a Cosmo object with the given parameters.
     *
     * @param params The simulation parameters.
     */
    Cosmo(const Params& params);

    /**
     * @brief Calculates the density contrast at a given redshift.
     *
     * @param z Redshift value.
     * @return The density contrast at the given redshift.
     */
    double delta(double z) const;

    /**
     * @brief Calculates the time derivative of the density contrast at a given
     * redshift.
     *
     * @param z Redshift value.
     * @return The time derivative of the density contrast at the given
     * redshift.
     */
    double dot_delta(double z) const;

    /**
     * @brief Gets the growth factor at a given redshift.
     *
     * @param z Redshift value.
     * @return The growth factor at the given redshift.
     */
    double gf(double z);

    /**
     * @brief Gets the time derivative of the growth factor at a given redshift.
     *
     * @param z Redshift value.
     * @return The time derivative of the growth factor at the given redshift.
     */
    double g_dot(double z);

    /**
     * @brief Gets the initial power spectrum.
     *
     * @return Reference to the initial power spectrum.
     */
    const PowerSpectrum& initial_pk() const;
};

#endif