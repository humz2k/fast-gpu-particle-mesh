#ifndef _FGPM_COSMO_HPP_
#define _FGPM_COSMO_HPP_

#include "params.hpp"
#include "power_spectrum.hpp"
#include <string>
#include <vector>

class Cosmo {
  private:
    double m_last_growth_z = -1;
    double m_gf;
    double m_g_dot;

    const Params& m_params;
    PowerSpectrum m_initial_pk;

    void rkck(double* y, double* dydx, int n, double x, double h, double* yout,
              double* yerr, void (Cosmo::*derivs)(double, double*, double*));
    void rkqs(double* y, double* dydx, int n, double* x, double htry,
              double eps, double* yscal, double* hdid, double* hnext,
              int* feval, void (Cosmo::*derivs)(double, double*, double*));
    void odesolve(double* ystart, int nvar, double x1, double x2, double eps,
                  double h1, void (Cosmo::*derivs)(double, double*, double*),
                  bool print_stat);

    void growths(double a, double* y, double* dydx);
    void update_growth_factor(double z);

    double Omega_nu_massive(double a);

  public:
    Cosmo(const Params& params);
    double delta(double z) const;
    double dot_delta(double z) const;
    double gf(double z);
    double g_dot(double z);
    const PowerSpectrum& initial_pk();
};

#endif