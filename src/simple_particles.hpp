#ifndef _FGPM_SIMPLE_PARTICLES_HPP_
#define _FGPM_SIMPLE_PARTICLES_HPP_

#include "gpu.hpp"
#include "simulation.hpp"

class SimpleParticles : public Particles<float3> {
  private:
    const Params& m_params;
    float3* m_pos;
    float3* m_vel;

  public:
    SimpleParticles(const Params& params, Cosmo& cosmo, Timestepper& ts);
    ~SimpleParticles();

    void update_positions(Timestepper& ts, float frac);
    void update_velocities(const Grid& grid);

    float3* pos();
    const float3* pos() const;

    float3* vel();
    const float3* vel() const;

    void dump(std::string filename) const;

    int nlocal() const;
};

#endif