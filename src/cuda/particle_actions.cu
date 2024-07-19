#include "allocators.hpp"
#include "event_logger.hpp"
#include "gpu.hpp"
#include "logging.hpp"
#include "mpi_distribution.hpp"
#include "particle_actions.hpp"

template <class T>
__global__ void CIC_kernel(T* __restrict d_grid, const float3* __restrict d_pos, int n_particles,
                           float mass, MPIDist dist) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n_particles)
        return;

    float3 my_particle = d_pos[idx];

    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float3 diff = my_particle - floor(my_particle);

    int ng = dist.ng();

    for (int x = 0; x < 2; x++) {
        int nx = (i + x) % ng;
        for (int y = 0; y < 2; y++) {
            int ny = (j + y) % ng;
            for (int z = 0; z < 2; z++) {
                int nz = (k + z) % ng;
                int indx = (nx * ng + ny) * ng + nz;

                float dx = (x == 0) ? (1.0f - diff.x) : diff.x;
                float dy = (y == 0) ? (1.0f - diff.y) : diff.y;
                float dz = (z == 0) ? (1.0f - diff.z) : diff.z;

                float mul = dx * dy * dz * mass;

                atomicAdd(&d_grid[indx].x, mul);
            }
        }
    }
}

template <class T>
void launch_CIC_kernel(T* d_grid, const float3* d_pos, int n_particles,
                       float mass, MPIDist dist, int numBlocks, int blockSize) {
    events.timers["kernel_cic_memset"].start();
    gpuCall(gpuMemset(d_grid, 0, sizeof(T) * dist.local_grid_size()));
    events.timers["kernel_cic_memset"].end();

    events.timers["kernel_cic"].start();
    gpuLaunch(CIC_kernel, numBlocks, blockSize, d_grid, d_pos, n_particles,
              mass, dist);
    events.timers["kernel_cic"].end();
}

template void launch_CIC_kernel<complexDoubleDevice>(complexDoubleDevice*,
                                                     const float3*, int, float,
                                                     MPIDist, int, int);
template void launch_CIC_kernel<complexFloatDevice>(complexFloatDevice*,
                                                    const float3*, int, float,
                                                    MPIDist, int, int);

__global__ void update_positions_kernel(float3* __restrict d_pos, const float3* __restrict d_vel,
                                        float prefactor, float ng, int nlocal) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= (nlocal))
        return;

    d_pos[idx] = fmod((d_pos[idx] + d_vel[idx] * prefactor) + ng, ng);
}

void launch_update_positions_kernel(float3* d_pos, const float3* d_vel,
                                    float prefactor, float ng, int nlocal,
                                    int numBlocks, int blockSize) {
    events.timers["kernel_update_positions"].start();
    gpuLaunch(update_positions_kernel, numBlocks, blockSize, d_pos, d_vel,
              prefactor, ng, nlocal);
    events.timers["kernel_update_positions"].end();
}

__global__ void update_velocities_kernel(float3* __restrict d_vel, const float3* __restrict d_pos,
                                         const float3* __restrict d_grad, double deltaT,
                                         double fscal, int nlocal,
                                         MPIDist dist) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= nlocal)
        return;

    int ng = dist.ng();

    float3 my_particle = d_pos[idx];

    float3 my_deltaV = make_float3(0.0, 0.0, 0.0);

    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float3 diff = my_particle - floor(my_particle);

    for (int x = 0; x < 2; x++) {
        int nx = (i + x) % ng;
        for (int y = 0; y < 2; y++) {
            int ny = (j + y) % ng;
            for (int z = 0; z < 2; z++) {
                int nz = (k + z) % ng;
                int indx = (nx * ng + ny) * ng + nz;

                float dx = (x == 0) ? (1.0f - diff.x) : diff.x;
                float dy = (y == 0) ? (1.0f - diff.y) : diff.y;
                float dz = (z == 0) ? (1.0f - diff.z) : diff.z;

                my_deltaV += d_grad[indx] * (dx * dy * dz * deltaT * fscal);
            }
        }
    }
    d_vel[idx] += my_deltaV;
}

void launch_update_velocities_kernel(float3* d_vel, const float3* d_pos,
                                     const float3* d_grad, double deltaT,
                                     double fscal, int nlocal, MPIDist dist,
                                     int numBlocks, int blockSize) {
    events.timers["kernel_update_velocities"].start();
    gpuLaunch(update_velocities_kernel, numBlocks, blockSize, d_vel, d_pos,
              d_grad, deltaT, fscal, nlocal, dist);
    events.timers["kernel_update_velocities"].end();
}