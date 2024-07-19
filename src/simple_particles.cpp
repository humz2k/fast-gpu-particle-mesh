#include "simple_particles.hpp"
#include "allocators.hpp"
#include "gpu.hpp"
#include "particle_actions.hpp"
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

void SimpleParticles::update_positions(Timestepper& ts, float frac) {
    int blockSize = BLOCKSIZE;
    int numBlocks = (nlocal() + (blockSize - 1)) / blockSize;
    float prefactor = ((ts.deltaT()) / (ts.a() * ts.a() * ts.adot())) * frac;
    launch_update_positions_kernel(m_pos, m_vel, prefactor, m_params.ng(),
                                   nlocal(), numBlocks, blockSize);
}

void SimpleParticles::update_velocities(const Grid& grid, Timestepper& ts,
                                        float frac) {
    int blockSize = BLOCKSIZE;
    int numBlocks = (nlocal() + (blockSize - 1)) / blockSize;
    float deltaT = ts.deltaT() * frac;
    LOG_INFO("fscal = %g", ts.fscal());
    launch_update_velocities_kernel(m_vel, m_pos, grid.grad(), deltaT,
                                    ts.fscal(), nlocal(), grid.dist(),
                                    numBlocks, blockSize);
}

float3* SimpleParticles::pos() { return m_pos; }

const float3* SimpleParticles::pos() const { return m_pos; }

float3* SimpleParticles::vel() { return m_vel; }

const float3* SimpleParticles::vel() const { return m_vel; }

void SimpleParticles::dump(std::string filename) const {
    float3* h_pos;
    cpu_allocator.alloc(&h_pos, sizeof(float3) * m_params.np() * m_params.np() *
                                    m_params.np());
    float3* h_vel;
    cpu_allocator.alloc(&h_vel, sizeof(float3) * m_params.np() * m_params.np() *
                                    m_params.np());
    gpuCall(gpuMemcpy(h_pos, m_pos,
                      sizeof(float3) * m_params.np() * m_params.np() *
                          m_params.np(),
                      gpuMemcpyDeviceToHost));
    gpuCall(gpuMemcpy(h_vel, m_vel,
                      sizeof(float3) * m_params.np() * m_params.np() *
                          m_params.np(),
                      gpuMemcpyDeviceToHost));
    gpuCall(gpuDeviceSynchronize());

    std::ofstream output(filename);

    output << "x,y,z,vx,vy,vz\n";

    for (int i = 0; i < m_params.np() * m_params.np() * m_params.np(); i++) {
        float3 p = h_pos[i];
        float3 v = h_vel[i];
        output << p.x << "," << p.y << "," << p.z << "," << v.x << "," << v.y
               << "," << v.z << "\n";
    }

    output.close();

    cpu_allocator.free(h_pos);
    cpu_allocator.free(h_vel);
}

int SimpleParticles::nlocal() const {
    return m_params.np() * m_params.np() * m_params.np();
}