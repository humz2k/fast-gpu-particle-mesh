#ifndef _FGPM_SIMPLE_PARTICLES_HPP_
#define _FGPM_SIMPLE_PARTICLES_HPP_

#include "serial_particles.hpp"
#include "simple_grid.hpp"

class SimpleParticles : public SerialParticles {
  public:
    SimpleParticles(const Params& params, Cosmo& cosmo, Timestepper& ts)
        : SerialParticles(params, cosmo, ts) {
        /**
         * SerialParticles(...) will allocate the correct amount of space in
         * this->pos() and this->vel(); but will not initialize particles in the
         * arrays. So, we need to call generate_displacement_ic on some grid in
         * order to generate the initial conditions.
         */
        SimpleGrid<SerialFFT, complexDoubleDevice> ic_grid(this->params(),
                                                           this->params().np());
        ic_grid.generate_displacement_ic(cosmo, ts, *this);
    }
};

#endif