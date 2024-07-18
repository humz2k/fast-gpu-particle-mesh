#include "cosmo.hpp"
#include "common.hpp"
#include "logging.hpp"
#include <assert.h>
#include <math.h>

Cosmo::Cosmo(const Params& params)
    : m_params(params), m_initial_pk(m_params.ipk()) {}

static double da_dtau(double a, double OmM, double OmL) {
    double da_dtau_2 = 1 + OmM * ((1 / a) - 1) + OmL * ((a * a) - 1);
    return sqrt(da_dtau_2);
}

static double da_dtau__3(double a, double OmM, double OmL) {
    double da_dtau_1 = da_dtau(a, OmM, OmL);
    return 1 / (da_dtau_1 * da_dtau_1 * da_dtau_1);
}

static double int_1_da_dtau_3(double a, double OmM, double OmL, int bins) {
    double start = 0;
    double end = a;
    double delta = (end - start) / ((double)bins);
    double sum = 0;
    for (int k = 1; k < bins; k++) {
        sum += da_dtau__3(start + ((double)k) * delta, OmM, OmL);
    }
    sum += (da_dtau__3(start, OmM, OmL) + da_dtau__3(end, OmM, OmL)) / 2.0f;
    sum *= delta;
    return sum;
}

static double calc_delta(double a, double OmM, double OmL) {
    double integral = int_1_da_dtau_3(a, OmM, OmL, 100);
    double diff = da_dtau(a, OmM, OmL);
    double mul = (5 * OmM) / (2 * a);
    return mul * diff * integral;
}

static double calc_dot_delta(double a, double OmM, double OmL, double h) {
    return (calc_delta(a + h, OmM, OmL) - calc_delta(a - h, OmM, OmL)) /
           (2.0f * h);
}

double Cosmo::delta(double z) const {
    double OmM = m_params.omega_matter();
    double OmL = 1.0 - (OmM);// + m_params.omega_nu());
    return calc_delta(z2a(z), OmM, OmL);
}

double Cosmo::dot_delta(double z) const {
    double OmM = m_params.omega_matter();
    double OmL = 1.0 - (OmM);// + m_params.omega_nu());
    return calc_dot_delta(z2a(z), OmM, OmL, 0.0001);
}

void Cosmo::rkck(double* y, double* dydx, int n, double x, double h,
                 double* yout, double* yerr,
                 void (Cosmo::*derivs)(double, double*, double*)) {
    int i;
    static double a2 = 0.2, a3 = 0.3, a4 = 0.6, a5 = 1.0, a6 = 0.875, b21 = 0.2,
                  b31 = 3.0 / 40.0, b32 = 9.0 / 40.0, b41 = 0.3, b42 = -0.9,
                  b43 = 1.2, b51 = -11.0 / 54.0, b52 = 2.5, b53 = -70.0 / 27.0,
                  b54 = 35.0 / 27.0, b61 = 1631.0 / 55296.0,
                  b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0,
                  b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0,
                  c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0,
                  c6 = 512.0 / 1771.0, dc5 = -277.00 / 14336.0;
    double dc1 = c1 - 2825.0 / 27648.0, dc3 = c3 - 18575.0 / 48384.0,
           dc4 = c4 - 13525.0 / 55296.0, dc6 = c6 - 0.25;
    double *ak2, *ak3, *ak4, *ak5, *ak6, *ytemp;

    ak2 = (double*)malloc(n * sizeof(double));
    ak3 = (double*)malloc(n * sizeof(double));
    ak4 = (double*)malloc(n * sizeof(double));
    ak5 = (double*)malloc(n * sizeof(double));
    ak6 = (double*)malloc(n * sizeof(double));
    ytemp = (double*)malloc(n * sizeof(double));

    for (i = 0; i < n; ++i)
        ytemp[i] = y[i] + b21 * h * dydx[i];
    (this->*derivs)(x + a2 * h, ytemp, ak2);
    for (i = 0; i < n; ++i)
        ytemp[i] = y[i] + h * (b31 * dydx[i] + b32 * ak2[i]);
    (this->*derivs)(x + a3 * h, ytemp, ak3);
    for (i = 0; i < n; ++i)
        ytemp[i] = y[i] + h * (b41 * dydx[i] + b42 * ak2[i] + b43 * ak3[i]);
    (this->*derivs)(x + a4 * h, ytemp, ak4);
    for (i = 0; i < n; ++i)
        ytemp[i] = y[i] + h * (b51 * dydx[i] + b52 * ak2[i] + b53 * ak3[i] +
                               b54 * ak4[i]);
    (this->*derivs)(x + a5 * h, ytemp, ak5);
    for (i = 0; i < n; ++i)
        ytemp[i] = y[i] + h * (b61 * dydx[i] + b62 * ak2[i] + b63 * ak3[i] +
                               b64 * ak4[i] + b65 * ak5[i]);
    (this->*derivs)(x + a6 * h, ytemp, ak6);
    for (i = 0; i < n; ++i)
        yout[i] =
            y[i] + h * (c1 * dydx[i] + c3 * ak3[i] + c4 * ak4[i] + c6 * ak6[i]);
    for (i = 0; i < n; ++i)
        yerr[i] = h * (dc1 * dydx[i] + dc3 * ak3[i] + dc4 * ak4[i] +
                       dc5 * ak5[i] + dc6 * ak6[i]);

    free(ytemp);
    free(ak6);
    free(ak5);
    free(ak4);
    free(ak3);
    free(ak2);
}

void Cosmo::rkqs(double* y, double* dydx, int n, double* x, double htry,
                 double eps, double* yscal, double* hdid, double* hnext,
                 int* feval, void (Cosmo::*derivs)(double, double*, double*)) {
    const double safety = 0.9;
    const double pgrow = -0.2;
    const double pshrnk = -0.25;
    const double errcon = 1.89e-4;

    double errmax;

    double* yerr = (double*)malloc(n * sizeof(double));
    double* ytemp = (double*)malloc(n * sizeof(double));
    double h = htry;

    for (;;) {
        rkck(y, dydx, n, *x, h, ytemp, yerr, derivs);
        *feval += 5;
        errmax = 0.0;
        for (int i = 0; i < n; ++i) {
            errmax = MAX(errmax, fabs(yerr[i] / yscal[i]));
        }
        errmax /= eps;
        if (errmax <= 1.0)
            break;
        double htemp = safety * h * pow((double)errmax, pshrnk);
        h = (h >= 0.0 ? MAX(htemp, 0.1 * h) : MIN(htemp, 0.1 * h));
        double xnew = (*x) + h;
        if (xnew == *x) {
            LOG_ERROR("Stepsize underflow in ODEsolve rkqs");
            exit(1);
        }
    }
    if (errmax > errcon)
        *hnext = safety * h * pow((double)errmax, pgrow);
    else
        *hnext = 5.0 * h;
    *x += (*hdid = h);
    for (int i = 0; i < n; ++i) {
        y[i] = ytemp[i];
    }
    free(ytemp);
    free(yerr);
}

void Cosmo::odesolve(double* ystart, int nvar, double x1, double x2, double eps,
                     double h1, void (Cosmo::*derivs)(double, double*, double*),
                     bool print_stat) {
    const double maxstp = 10000;
    const double tiny = 1.0e-30;

    double hnext, hdid;
    const double hmin = 0.0;

    int feval = 0;

    double* yscal = (double*)malloc(nvar * sizeof(double));
    double* y = (double*)malloc(nvar * sizeof(double));
    double* dydx = (double*)malloc(nvar * sizeof(double));

    double x = x1;
    double h = SIGN(h1, x2 - x1);
    int nok = 0;
    int nbad = 0;
    for (int i = 0; i < nvar; ++i) {
        y[i] = ystart[i];
    }

    for (int nstp = 0; nstp < maxstp; ++nstp) {
        (this->*derivs)(x, y, dydx);
        ++feval;
        for (int i = 0; i < nvar; ++i) {
            yscal[i] = fabs(y[i]) + fabs(dydx[i] * h) + tiny;
        }
        if ((x + h - x2) * (x + h - x1) > 0.0)
            h = x2 - x;
        rkqs(y, dydx, nvar, &x, h, eps, yscal, &hdid, &hnext, &feval, derivs);
        if (hdid == h)
            ++nok;
        else
            ++nbad;
        if ((x - x2) * (x2 - x1) >= 0.0) {
            for (int i = 0; i < nvar; ++i) {
                ystart[i] = y[i];
            }
            free(dydx);
            free(y);
            free(yscal);
            if (print_stat) {
                LOG_INFO("ODEsolve:\n");
                LOG_INFO(" Evolved from x = %f to x = %f\n", x1, x2);
                LOG_INFO(" successful steps: %d\n", nok);
                LOG_INFO(" bad steps: %d\n", nbad);
                LOG_INFO(" function evaluations: %d\n", feval);
            }
            return;
        }
        if (fabs(hnext) <= hmin) {
            LOG_ERROR("Step size too small in ODEsolve");
            exit(1);
        }
        h = hnext;
    }
    LOG_ERROR("Too many steps in ODEsolve");
    exit(1);
}

double Cosmo::Omega_nu_massive(double a) {
    double mat = m_params.omega_nu() / pow(a, 3.0f);
    double rad =
        m_params.f_nu_massive() * m_params.omega_radiation() / pow(a, 4.0f);
    return (mat >= rad) * mat + (rad > mat) * rad;
}

void Cosmo::growths(double a, double* y, double* dydx) {
    double H;
    H = sqrt(m_params.omega_cb() / pow(a, 3.0) +
             (1.0 + m_params.f_nu_massless()) * m_params.omega_radiation() /
                 pow(a, 4.0) +
             Omega_nu_massive(a) +
             (1.0 - m_params.omega_matter() -
              (1.0 + m_params.f_nu_massless()) * m_params.omega_radiation() *
                  pow(a, (-3.0 * (1.0 + m_params.w_de() + m_params.wa_de()))) *
                  exp(-3.0 * m_params.wa_de() * (1.0 - a))));
    dydx[0] = y[1] / (a * H);
    dydx[1] =
        -2.0 * y[1] / a + 1.5 * m_params.omega_cb() * y[0] / (H * pow(a, 4.0f));
}

void Cosmo::update_growth_factor(double z) {
    double x1, x2, dplus, ddot;
    const double zinfinity = 100000.0;

    x1 = 1.0 / (1.0 + zinfinity);
    x2 = 1.0 / (1.0 + z);
    double ystart[2];
    ystart[0] = x1;
    ystart[1] = 0.0;

    odesolve(ystart, 2, x1, x2, 1.0e-6, 1.0e-6, &Cosmo::growths, false);
    LOG_DEBUG("Dplus = %f;  Ddot = %f \n", ystart[0], ystart[1]);

    dplus = ystart[0];
    ddot = ystart[1];
    x1 = 1.0 / (1.0 + zinfinity);
    x2 = 1.0;
    ystart[0] = x1;
    ystart[1] = 0.0;

    odesolve(ystart, 2, x1, x2, 1.0e-6, 1.0e-6, &Cosmo::growths, false);
    LOG_DEBUG("Dplus = %f;  Ddot = %f \n", ystart[0], ystart[1]);

    m_gf = dplus / ystart[0];
    m_g_dot = ddot / ystart[0];
    m_last_growth_z = z;
}

double Cosmo::gf(double z) {
    if (m_last_growth_z != z) {
        update_growth_factor(z);
    }
    return m_gf;
}

double Cosmo::g_dot(double z) {
    if (m_last_growth_z != z) {
        update_growth_factor(z);
    }
    return m_g_dot;
}

const PowerSpectrum& Cosmo::initial_pk() const { return m_initial_pk; }