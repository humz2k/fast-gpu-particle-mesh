#include "allocators.hpp"
#include "gpu.hpp"
#include "logging.hpp"
#include "mpi_distribution.hpp"
#include "particle_actions.hpp"

template<class T>
__global__ void CIC_kernel(T* d_grid, const float3* d_pos, int n_particles, float mass, MPIDist dist){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n_particles)return;

    float3 my_particle = d_pos[idx];

    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float diffx = (my_particle.x - (float)i);
    float diffy = (my_particle.y - (float)j);
    float diffz = (my_particle.z - (float)k);

    int ng = dist.ng();

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

                float mul = dx*dy*dz*mass; //* (1.0f/(ng*ng*ng));

                atomicAdd(&d_grid[indx].x,mul);
            }
        }
    }
}

template<class T>
void launch_CIC_kernel(T* d_grid, const float3* d_pos, int n_particles, float mass, MPIDist dist, int numBlocks, int blockSize){
    gpuCall(gpuMemset(d_grid,0,sizeof(T)*dist.local_grid_size()));
    gpuLaunch(CIC_kernel,numBlocks,blockSize,d_grid,d_pos,n_particles,mass,dist);
}

template void launch_CIC_kernel<complexDoubleDevice>(complexDoubleDevice*,const float3*,int,float,MPIDist,int,int);
template void launch_CIC_kernel<complexFloatDevice>(complexFloatDevice*,const float3*,int,float,MPIDist,int,int);