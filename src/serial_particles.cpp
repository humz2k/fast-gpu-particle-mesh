#include "serial_particles.hpp"
#include "allocators.hpp"
#include "gpu.hpp"
#include "particle_actions.hpp"

SerialParticles::SerialParticles(const Params& params, Cosmo& cosmo,
                                 Timestepper& ts)
    : m_params(params) {
    gpu_allocator.alloc(&m_pos, sizeof(float3) * m_params.np() * m_params.np() *
                                    m_params.np());
    gpu_allocator.alloc(&m_vel, sizeof(float3) * m_params.np() * m_params.np() *
                                    m_params.np());
}

SerialParticles::~SerialParticles() {
    gpu_allocator.free(m_pos);
    gpu_allocator.free(m_vel);
}

void SerialParticles::update_positions(Timestepper& ts, float frac) {
    int blockSize = BLOCKSIZE;
    int numBlocks = (nlocal() + (blockSize - 1)) / blockSize;
    float prefactor = ((ts.deltaT()) / (ts.a() * ts.a() * ts.adot())) * frac;
    launch_update_positions_kernel(m_pos, m_vel, prefactor, m_params.ng(),
                                   nlocal(), numBlocks, blockSize);
}

void SerialParticles::update_velocities(const Grid& grid, Timestepper& ts,
                                        float frac) {
    int blockSize = BLOCKSIZE;
    int numBlocks = (nlocal() + (blockSize - 1)) / blockSize;
    float deltaT = ts.deltaT() * frac;
    LOG_INFO("fscal = %g", ts.fscal());
    launch_update_velocities_kernel(m_vel, m_pos, grid.grad(), deltaT,
                                    ts.fscal(), nlocal(), grid.dist(),
                                    numBlocks, blockSize);
}

float3* SerialParticles::pos() { return m_pos; }

const float3* SerialParticles::pos() const { return m_pos; }

float3* SerialParticles::vel() { return m_vel; }

const float3* SerialParticles::vel() const { return m_vel; }

void SerialParticles::dump(std::string filename) const {
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

int SerialParticles::nlocal() const {
    return m_params.np() * m_params.np() * m_params.np();
}

const Params& SerialParticles::params() const { return m_params; }

void SerialParticles::fold(int nfolds) {
    int blockSize = BLOCKSIZE;
    int numBlocks = (nlocal() + (blockSize - 1)) / blockSize;
    assert(m_folded_copy == NULL);
    gpu_allocator.alloc(&m_folded_copy, sizeof(float3) * nlocal());
    gpuMemcpy(m_folded_copy, m_pos, sizeof(float3) * nlocal(),
              gpuMemcpyDeviceToDevice);
    launch_fold_particles_kernel(m_pos, nfolds, nlocal(), params().ng(),
                                 numBlocks, blockSize);
}

void SerialParticles::unfold() {
    assert(m_folded_copy != NULL);
    gpuMemcpy(m_pos, m_folded_copy, sizeof(float3) * nlocal(),
              gpuMemcpyDeviceToDevice);
    gpu_allocator.free(m_folded_copy);
    m_folded_copy = NULL;
}