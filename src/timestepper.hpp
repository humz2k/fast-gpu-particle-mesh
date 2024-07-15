#ifndef _FGPM_TIMESTEPPER_HPP_
#define _FGPM_TIMESTEPPER_HPP_

#include "params.hpp"

class Timestepper{
    private:
        const Params& m_params;
        double m_a;
        double m_z;
        double m_deltaT;
    public:
        Timestepper(const Params& params);

        void advance_half_step();
        void reverse_half_step();

        double a() const;
        void a(double new_a);

        double z() const;
        void z(double new_z);

        double deltaT() const;
        double adot() const;
        double fscal() const;
        double omega_nu_massive() const;
};

#endif
