#include "allocators.hpp"
#include "gpu.hpp"
#include "logging.hpp"
#include "mpi_distribution.hpp"
#include "particle_actions.hpp"

template <class T>
__global__ void CIC_kernel(T* d_grid, const float3* d_pos, int n_particles,
                           float mass, MPIDist dist) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n_particles)
        return;

    float3 my_particle = d_pos[idx];

    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float diffx = (my_particle.x - (float)i);
    float diffy = (my_particle.y - (float)j);
    float diffz = (my_particle.z - (float)k);

    int ng = dist.ng();

    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                int nx = (i + x) % ng;
                int ny = (j + y) % ng;
                int nz = (k + z) % ng;
                int indx = (nx)*ng * ng + (ny)*ng + nz;

                float dx = diffx;
                if (x == 0) {
                    dx = 1 - dx;
                }
                float dy = diffy;
                if (y == 0) {
                    dy = 1 - dy;
                }
                float dz = diffz;
                if (z == 0) {
                    dz = 1 - dz;
                }

                float mul = dx * dy * dz * mass; //* (1.0f/(ng*ng*ng));

                atomicAdd(&d_grid[indx].x, mul);
            }
        }
    }
}

template <class T>
void launch_CIC_kernel(T* d_grid, const float3* d_pos, int n_particles,
                       float mass, MPIDist dist, int numBlocks, int blockSize) {
    gpuCall(gpuMemset(d_grid, 0, sizeof(T) * dist.local_grid_size()));
    gpuLaunch(CIC_kernel, numBlocks, blockSize, d_grid, d_pos, n_particles,
              mass, dist);
}

template void launch_CIC_kernel<complexDoubleDevice>(complexDoubleDevice*,
                                                     const float3*, int, float,
                                                     MPIDist, int, int);
template void launch_CIC_kernel<complexFloatDevice>(complexFloatDevice*,
                                                    const float3*, int, float,
                                                    MPIDist, int, int);

__global__ void update_positions_kernel(float3* d_pos, const float3* d_vel, float prefactor, float ng, int nlocal){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= (nlocal))return;

    d_pos[idx] = fmod((d_pos[idx] + d_vel[idx] * prefactor) + ng,ng);
}

void launch_update_positions_kernel(float3* d_pos, const float3* d_vel, float prefactor, float ng, int nlocal, int numBlocks, int blockSize){
    gpuLaunch(update_positions_kernel,numBlocks,blockSize,d_pos,d_vel,prefactor,ng,nlocal);
}

__global__ void update_velocities_kernel(float3* d_vel, const float3* d_pos, const float3* d_grad, double deltaT, double fscal, int nlocal, MPIDist dist){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= nlocal)return;

    int ng = dist.ng();

    float3 my_particle = d_pos[idx];

    float3 my_deltaV = make_float3(0.0,0.0,0.0);
    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float diffx = (my_particle.x - (float)i);
    float diffy = (my_particle.y - (float)j);
    float diffz = (my_particle.z - (float)k);

    for (int x = 0; x < 2; x++){
        for (int y = 0; y < 2; y++){
            for (int z = 0; z < 2; z++){
                int nx = (i + x)%ng;
                int ny = (j + y)%ng;
                int nz = (k + z)%ng;
                int indx = (nx)*ng*ng + (ny)*ng + nz;

                float dx = diffx;
                if (x == 0){
                    dx = 1 - dx;
                }
                float dy = diffy;
                if (y == 0){
                    dy = 1 - dy;
                }
                float dz = diffz;
                if (z == 0){
                    dz = 1 - dz;
                }

                float3 grad = d_grad[indx];
                //printf("grad: %g %g %g\n",grad.x,grad.y,grad.z);
                //float gp_scale = np/(float)ng;
                float mul = dx*dy*dz * deltaT * fscal;// * gp_scale * gp_scale * gp_scale;//* (1.0f/((double)(ng*ng*ng)));// (1.0f/((double)(ng*ng*ng)));// * deltaT * fscal * (1.0f/((double)(ng*ng*ng)));
                my_deltaV.x += mul*grad.x;
                my_deltaV.y += mul*grad.y;
                my_deltaV.z += mul*grad.z;

                //atomicAdd(&grid[indx].x,(double)mul);
            }
        }
    }
    //printf("%g %g %g\n",my_deltaV.x,my_deltaV.y,my_deltaV.z);
    d_vel[idx] = d_vel[idx] + my_deltaV;

}

void launch_update_velocities_kernel(float3* d_vel, const float3* d_pos, const float3* d_grad, double deltaT, double fscal, int nlocal, MPIDist dist, int numBlocks, int blockSize){
    gpuLaunch(update_velocities_kernel,numBlocks,blockSize,d_vel,d_pos,d_grad,deltaT,fscal,nlocal,dist);
}