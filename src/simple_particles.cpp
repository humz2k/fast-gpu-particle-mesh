#include "simple_particles.hpp"
#include "allocators.hpp"
#include "gpu.hpp"
#include "simple_grid.hpp"

SimpleParticles::SimpleParticles(const Params& params, Cosmo& cosmo,
                                 Timestepper& ts)
    : m_params(params) {
    gpu_allocator.alloc(&m_pos, sizeof(float3) * m_params.np() * m_params.np() *
                                    m_params.np());
    gpu_allocator.alloc(&m_vel, sizeof(float3) * m_params.np() * m_params.np() *
                                    m_params.np());

    SimpleGrid<complexDoubleDevice> ic_grid(m_params, m_params.np());

    ic_grid.generate_displacement_ic(cosmo, ts, *this);
}

SimpleParticles::~SimpleParticles() {
    gpu_allocator.free(m_pos);
    gpu_allocator.free(m_vel);
}

void SimpleParticles::update_positions() {}

void SimpleParticles::update_velocities(const Grid& grid) {}

float3* SimpleParticles::pos(){return m_pos;}
const float3* SimpleParticles::pos() const{return m_pos;}

float3* SimpleParticles::vel(){return m_vel;}
const float3* SimpleParticles::vel() const{return m_vel;}