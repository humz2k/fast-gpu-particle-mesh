#include "allocators.hpp"
#include "event_logger.hpp"
#include "gpu.hpp"
#include "logging.hpp"
#include "mpi_distribution.hpp"
#include "pk_bins.hpp"
#include "power_spectrum.hpp"
#include <cassert>
#include <stdio.h>
#include <vector>

template <class T> __forceinline__ __device__ T sqr(T v) { return v * v; }

__forceinline__ __device__ float3 rcp(float3 v) {
    return make_float3(__frcp_rn(v.x), __frcp_rn(v.y), __frcp_rn(v.z));
}

__forceinline__ __device__ double get_pk_cic_filter(int idx, MPIDist dist) {
    double d = ((2 * M_PI) / (((double)(dist.ng()))));
    float3 kmodes = dist.kmodes(idx, d);

    float3 filt = rcp(sqr(sqr(sin(0.5f * kmodes) / (0.5f * kmodes))));

    float filt1 = (kmodes.x == 0) ? 1.0f : filt.x;
    float filt2 = (kmodes.y == 0) ? 1.0f : filt.y;
    float filt3 = (kmodes.z == 0) ? 1.0f : filt.z;

    return filt1 * filt2 * filt3;
}

template <class T>
__global__ void
bin_power(const T* __restrict d_grid, double2* __restrict d_bins, double k_min,
          double k_delta, int n_k_bins, double rl, MPIDist dist) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dist.local_grid_size())
        return;

    int3 global_coords = dist.global_coords(idx);

    int ng = dist.ng();
    int ng_2 = ng / 2;

    if (((global_coords.x == 0) && (global_coords.y == 0) &&
         (global_coords.z == 0)) ||
        ((global_coords.x == ng_2) && (global_coords.y == ng_2) &&
         (global_coords.z == ng_2)))
        return;

    float3 kmodes = dist.kmodes(idx, (2.0 * M_PI) / rl);

    float k_mag = len(kmodes);
    int k_bin_idx = (int)((k_mag - k_min) / k_delta);

    assert((k_bin_idx >= 0) && (k_bin_idx < n_k_bins));

    double scale = ((rl * rl * rl) / ((double)ng * ng * ng)) *
                   (1.0 / ((double)ng * ng * ng));
    double val = len2(d_grid[idx]) * scale * get_pk_cic_filter(idx, dist);

    atomicAdd(&(d_bins[k_bin_idx].x), val);
    atomicAdd(&(d_bins[k_bin_idx].y), 1.0);
}

template <class T>
void launch_bin_power(const T* d_grid, double2* d_bins, double k_min,
                      double k_delta, int n_k_bins, double rl, MPIDist dist,
                      int numBlocks, int blockSize) {
    events.timers["kernel_bin_power_memset"].start();
    gpuMemset(d_bins, 0, sizeof(double2) * n_k_bins);
    events.timers["kernel_bin_power_memset"].end();

    events.timers["kernel_bin_power"].start();
    gpuLaunch(bin_power, numBlocks, blockSize, d_grid, d_bins, k_min, k_delta,
              n_k_bins, rl, dist);
    events.timers["kernel_bin_power"].end();
}

template void launch_bin_power<complexDoubleDevice>(const complexDoubleDevice*,
                                                    double2*, double, double,
                                                    int, double, MPIDist, int,
                                                    int);

template void launch_bin_power<complexFloatDevice>(const complexFloatDevice*,
                                                   double2*, double, double,
                                                   int, double, MPIDist, int,
                                                   int);