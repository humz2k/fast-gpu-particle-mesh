#include "allocators.hpp"
#include "gpu.hpp"
#include "initializer.hpp"
#include "logging.hpp"
#include "power_spectrum.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

template <class T>
__global__ void generate_real_random(T* __restrict grid, int seed,
                                     const MPIDist dist) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= dist.local_grid_size())
        return;

    curandState state;
    curand_init(seed, dist.global_idx(idx), 0, &state);

    double amp = curand_normal_double(&state);
    T out;
    out.x = amp;
    out.y = 0;
    grid[idx] = out;
}

template <class T>
void launch_generate_real_random(T* d_grid, int seed, const MPIDist dist,
                                 int numBlocks, int blockSize) {
    gpuLaunch(generate_real_random, numBlocks, blockSize, d_grid, seed, dist);
}

template void
launch_generate_real_random<complexDoubleDevice>(complexDoubleDevice*, int,
                                                 const MPIDist, int, int);
template void
launch_generate_real_random<complexFloatDevice>(complexFloatDevice*, int,
                                                const MPIDist, int, int);

__forceinline__ __device__ double interp_power_spectrum(int idx, double* in,
                                                        double k_delta,
                                                        double k_min, double rl,
                                                        const MPIDist dist) {

    int3 idx3d = dist.global_coords(idx);

    // pole
    if ((idx3d.x == 0) && (idx3d.y == 0) && (idx3d.z == 0))
        return 0;

    // k modes and magnitude
    float3 kmodes = dist.kmodes(idx, (2.0f * M_PI) / rl);
    double k_mag =
        sqrt(kmodes.x * kmodes.x + kmodes.y * kmodes.y + kmodes.z * kmodes.z);

    // bins for interpolation
    int left_bin = (int)(k_mag / k_delta);
    int right_bin = left_bin + 1;

    // linear interpolation in log space
    double logy1 = log(in[left_bin]);
    double logx1 = log(k_delta * (double)left_bin);
    double logy2 = log(in[right_bin]);
    double logx2 = log(k_delta * (double)right_bin);
    double logx = log(k_mag);

    double logy = logy1 + ((logy2 - logy1) / (logx2 - logx1)) * (logx - logx1);

    // convert the interpolated log value back to linear space and scale it
    double y = exp(logy) *
               (((double)(dist.ng() * dist.ng() * dist.ng())) / (rl * rl * rl));

    // special case where the left bin is 0 (?)
    if (left_bin == 0) {
        y = in[right_bin] *
            (((double)(dist.ng() * dist.ng() * dist.ng())) / (rl * rl * rl));
    }

    return y;
}

template <class T>
__global__ void scale_amplitudes_by_power_spectrum(T* __restrict grid,
                                                   double* in, double k_delta,
                                                   double k_min, double rl,
                                                   const MPIDist dist) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= dist.local_grid_size())
        return;

    grid[idx] = grid[idx] *
                sqrt(interp_power_spectrum(idx, in, k_delta, k_min, rl, dist));
}

template <class T>
void launch_scale_amplitudes_by_power_spectrum(T* grid,
                                               const PowerSpectrum& initial_pk,
                                               double rl, const MPIDist dist,
                                               int numBlocks, int blockSize) {
    const double* h_values = initial_pk.h_values().data();

    double max_k =
        sqrt(pow(((((double)dist.ng()) / 2.0) * 2.0 * M_PI) / rl, 3.0));
    LOG_INFO("max k = %g", max_k);
    if (max_k > initial_pk.k_max()) {
        LOG_ERROR("input ipk only goes to %g", initial_pk.k_max());
        exit(1);
    }

    double* d_values;
    gpu_allocator.alloc(&d_values,
                        sizeof(double) * initial_pk.h_values().size());
    gpuCall(gpuMemcpy(d_values, h_values,
                      sizeof(double) * initial_pk.h_values().size(),
                      gpuMemcpyHostToDevice));

    gpuLaunch(scale_amplitudes_by_power_spectrum, numBlocks, blockSize, grid,
              d_values, initial_pk.k_delta(), initial_pk.k_min(), rl, dist);

    gpu_allocator.free(d_values);
}

template void launch_scale_amplitudes_by_power_spectrum<complexDoubleDevice>(
    complexDoubleDevice*, const PowerSpectrum& inital_pk, double, const MPIDist,
    int, int);

template void launch_scale_amplitudes_by_power_spectrum<complexFloatDevice>(
    complexFloatDevice*, const PowerSpectrum& inital_pk, double, const MPIDist,
    int, int);