#include "event_logger.hpp"
#include "gpu.hpp"
#include "mpi_distribution.hpp"
#include "solver.hpp"

__forceinline__ __device__ float calc_greens(int idx, MPIDist dist) {
    if (dist.is_global_origin(idx))
        return 0.0;
    float ng = dist.ng();
    float3 c = cos(to_float3(dist.global_coords(idx)) * ((2.0f * M_PI) / ng));
    return (0.5f / (ng * ng * ng)) / (c.x + c.y + c.z - 3.0f);
}

template <class T>
__global__ void kspace_solve_gradient(const T* grid, T* d_x, T* d_y, T* d_z,
                                      MPIDist dist) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dist.local_grid_size())
        return;

    float greens = -calc_greens(idx, dist);

    float ng = dist.ng();
    float d = (2.0f * M_PI) / ng;

    float3 c = sin(dist.kmodes(idx, d)) * greens;

    T i_rho = flip_phase(grid[idx]);

    d_x[idx] = c.x * i_rho;
    d_y[idx] = c.y * i_rho;
    d_z[idx] = c.z * i_rho;
}

template <class T>
void launch_kspace_solve_gradient(const T* grid, T* d_x, T* d_y, T* d_z,
                                  MPIDist dist, int numBlocks, int blockSize) {
    events.timers["kernel_kspace_solve_gradient"].start();
    gpuLaunch(kspace_solve_gradient, numBlocks, blockSize, grid, d_x, d_y, d_z,
              dist);
    events.timers["kernel_kspace_solve_gradient"].end();
}

template void launch_kspace_solve_gradient<complexDoubleDevice>(
    const complexDoubleDevice*, complexDoubleDevice*, complexDoubleDevice*,
    complexDoubleDevice*, MPIDist, int, int);

template void launch_kspace_solve_gradient<complexFloatDevice>(
    const complexFloatDevice*, complexFloatDevice*, complexFloatDevice*,
    complexFloatDevice*, MPIDist, int, int);

template <class T>
__global__ void combine_vectors(float3* d_grad, const T* d_x, const T* d_y,
                                const T* d_z, MPIDist dist) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dist.local_grid_size())
        return;
    d_grad[idx] = make_float3(d_x[idx].x, d_y[idx].x, d_z[idx].x);
}

template <class T>
void launch_combine_vectors(float3* d_grad, const T* d_x, const T* d_y,
                            const T* d_z, MPIDist dist, int numBlocks,
                            int blockSize) {
    events.timers["kernel_combine_vectors"].start();
    gpuLaunch(combine_vectors, numBlocks, blockSize, d_grad, d_x, d_y, d_z,
              dist);
    events.timers["kernel_combine_vectors"].end();
}

template void launch_combine_vectors<complexDoubleDevice>(
    float3*, const complexDoubleDevice*, const complexDoubleDevice*,
    const complexDoubleDevice*, MPIDist, int, int);

template void launch_combine_vectors<complexFloatDevice>(
    float3*, const complexFloatDevice*, const complexFloatDevice*,
    const complexFloatDevice*, MPIDist, int, int);
