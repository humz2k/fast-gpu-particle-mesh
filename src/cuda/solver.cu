#include "event_logger.hpp"
#include "gpu.hpp"
#include "mpi_distribution.hpp"
#include "solver.hpp"

__forceinline__ __device__ float calc_greens(int idx, MPIDist dist) {
    int3 global_coords = dist.global_coords(idx);
    if ((global_coords.x == 0) && (global_coords.y == 0) &&
        (global_coords.z == 0))
        return 0.0;

    float ng = dist.ng();

    float d = (2.0f * M_PI) / ng;
    float3 idx3d =
        make_float3(global_coords.x, global_coords.y, global_coords.z);

    float3 c = cos(idx3d * d);

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

    T rho = grid[idx];

    T out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;
    T out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;
    d_y[idx] = out_y;

    T out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;
    d_z[idx] = out_z;
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
