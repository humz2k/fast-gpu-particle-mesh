#include "allocators.hpp"
#include "gpu.hpp"
#include "logging.hpp"
#include "mpi_distribution.hpp"
#include "pk_bins.hpp"
#include "power_spectrum.hpp"
#include <cassert>
#include <vector>

__forceinline__ __device__ double get_pk_cic_filter(float3 kmodes, double d) {
    float filt1 = sinf(0.5f * kmodes.x) / (0.5 * kmodes.x);
    filt1 = filt1 * filt1;
    filt1 = __frcp_rn(filt1 * filt1);
    if (kmodes.x == 0) {
        filt1 = 1.0;
    }

    float filt2 = sinf(0.5f * kmodes.y) / (0.5 * kmodes.y);
    filt2 = filt2 * filt2;
    filt2 = __frcp_rn(filt2 * filt2);
    if (kmodes.y == 0) {
        filt2 = 1.0;
    }

    float filt3 = sinf(0.5f * kmodes.z) / (0.5 * kmodes.z);
    filt3 = filt3 * filt3;
    filt3 = __frcp_rn(filt3 * filt3);
    if (kmodes.z == 0) {
        filt3 = 1.0;
    }
    double filter = filt1 * filt2 * filt3;

    return filter;
}

template <class T>
__global__ void bin_power(const T* d_grid, double2* d_bins, double k_min,
                          double k_delta, int n_k_bins, double rl,
                          MPIDist dist) {
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

    double d = (2.0 * M_PI) / rl;

    float3 kmodes = dist.kmodes(idx, d);

    float k_mag = len(kmodes);
    int k_bin_idx = (int)((k_mag - k_min) / k_delta);

    assert((k_bin_idx >= 0) && (k_bin_idx < n_k_bins));

    double scale = (1.0 / ((double)(ng * ng * ng))) * (rl / ((double)ng));
    double val = len2(d_grid[idx]) * scale * get_pk_cic_filter(kmodes, d);

    atomicAdd(&(d_bins[k_bin_idx].x), val);
    atomicAdd(&(d_bins[k_bin_idx].y), 1.0);
}

template <class T>
void launch_bin_power(const T* d_grid, double2* d_bins, double k_min,
                      double k_delta, int n_k_bins, double rl, MPIDist dist,
                      int numBlocks, int blockSize) {
    gpuMemset(d_bins, 0, sizeof(double2) * n_k_bins);
    gpuLaunch(bin_power, numBlocks, blockSize, d_grid, d_bins, k_min, k_delta,
              n_k_bins, rl, dist);
}

template void launch_bin_power<complexDoubleDevice>(const complexDoubleDevice*,
                                                    double2*, double, double,
                                                    int, double, MPIDist, int,
                                                    int);

template void launch_bin_power<complexFloatDevice>(const complexFloatDevice*,
                                                   double2*, double, double,
                                                   int, double, MPIDist, int,
                                                   int);