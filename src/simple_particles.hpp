#ifndef _FGPM_SIMPLE_PARTICLES_HPP_
#define _FGPM_SIMPLE_PARTICLES_HPP_

#include "simulation.hpp"
#include "gpu.hpp"

class SimpleParticles : public Particles<float3> {
    private:
        const Params& m_params;
        float3* m_pos;
        float3* m_vel;

    public:
        SimpleParticles(const Params& params, Cosmo& cosmo, Timestepper& ts);
        ~SimpleParticles();

        void update_positions();
        void update_velocities(const Grid& grid);
};

#endif