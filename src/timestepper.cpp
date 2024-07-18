#include "timestepper.hpp"
#include "common.hpp"
#include "logging.hpp"
#include <math.h>

Timestepper::Timestepper(const Params& _params)
    : m_params(_params), m_z(m_params.z_ini()), m_a(z2a(m_z)) {
    m_deltaT =
        (((1.0 / (m_params.z_fin() + 1.0)) - (1.0 / (m_params.z_ini() + 1.0))) /
         (double)m_params.nsteps());
    LOG_INFO("a = %g, z = %g, deltaT = %g", a(), z(), m_deltaT);
}

double Timestepper::omega_nu_massive() const {
    double mat = m_params.omega_nu() / pow(m_a, 3.0);
    double rad =
        m_params.f_nu_massive() * m_params.omega_radiation() / pow(m_a, 4.0);
    return (mat >= rad) * mat + (rad > mat) * rad;
}

double Timestepper::adot() const {
    double pp1 = pow(m_a, -3.0 * (m_params.w_de() + m_params.wa_de())) *
                 exp(-3.0 * m_params.wa_de() * (1.0 - m_a));

    double tmp =
        m_params.omega_cb() +
        (1.0 + m_params.f_nu_massless()) * m_params.omega_radiation() / m_a +
        omega_nu_massive() * pow(m_a, 3.0) +
        (1.0 - m_params.omega_matter() -
         (1.0 + m_params.f_nu_massless()) * m_params.omega_radiation()) *
            pp1;

    return sqrt(tmp);
}

double Timestepper::fscal() const {
    float dtdy = a() / (a() * adot());
    double phiscal =
        1.5 *
        m_params
            .omega_cb(); // Poisson equation is grad^2 phi = 3/2 omega_m (rho-1)
    float out = phiscal * dtdy * (1.0 / m_a);
    LOG_INFO("a = %g, adot = %g, dtdy = %g, phiscal = %g,fscal = %g",a(),adot(),dtdy,phiscal,out);
    return out;
}

double Timestepper::deltaT() const { return m_deltaT; }

double Timestepper::a() const { return m_a; }

void Timestepper::a(double new_a) {
    m_a = new_a;
    m_z = a2z(m_a);
}

double Timestepper::z() const { return m_z; }

void Timestepper::z(double new_z) {
    m_z = new_z;
    m_a = z2a(new_z);
}

void Timestepper::advance_half_step() {
    a(a() + deltaT() * 0.5);
    LOG_INFO("a = %g, z = %g", a(), z());
}

void Timestepper::reverse_half_step() {
    a(a() - deltaT() * 0.5);
    LOG_INFO("a = %g, z = %g", a(), z());
}