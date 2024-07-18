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
    double k_mag = len(kmodes);

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

template <class T>
__global__ void transform_density_field(const T* d_grid, T* d_x, T* d_y, T* d_z,
                                        double delta, double rl, double a,
                                        MPIDist dist) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dist.local_grid_size())
        return;

    float3 kmodes = dist.kmodes(idx, (2.0 * M_PI) / rl);

    double k2 = len2(kmodes);
    double k2mul = (k2 == 0.0) ? 0.0 : (1.0 / k2);
    double mul = (1.0 / delta) * k2mul;

    T current = d_grid[idx] * mul;
    T sx = current * kmodes.x;
    T sy = current * kmodes.y;
    T sz = current * kmodes.z;

    T out_x;
    out_x.x = sx.y;
    out_x.y = -sx.x;
    T out_y;
    out_y.x = sy.y;
    out_y.y = -sy.x;
    T out_z;
    out_z.x = sz.y;
    out_z.y = -sz.x;

    d_x[idx] = out_x;
    d_y[idx] = out_y;
    d_z[idx] = out_z;
}

template <class T>
void launch_transform_density_field(T* d_grid, T* d_x, T* d_y, T* d_z,
                                    double delta, double rl, double a,
                                    MPIDist dist, int numBlocks,
                                    int blockSize) {
    gpuLaunch(transform_density_field, numBlocks, blockSize, d_grid, d_x, d_y,
              d_z, delta, rl, a, dist);
}

template void launch_transform_density_field<complexDoubleDevice>(
    complexDoubleDevice*, complexDoubleDevice*, complexDoubleDevice*,
    complexDoubleDevice*, double, double, double, MPIDist, int, int);
template void launch_transform_density_field<complexFloatDevice>(
    complexFloatDevice*, complexFloatDevice*, complexFloatDevice*,
    complexFloatDevice*, double, double, double, MPIDist, int, int);

template <class T>
__global__ void combine_density_vectors(float3* d_grad, T* d_x, T* d_y, T* d_z,
                                        MPIDist dist) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dist.local_grid_size())
        return;
    double ng = dist.ng();
    double scale = 1.0 / (ng * ng * ng);
    d_grad[idx] =
        make_float3(d_x[idx].x * scale, d_y[idx].x * scale, d_z[idx].x * scale);
}

template <class T>
void launch_combine_density_vectors(float3* d_grad, T* d_x, T* d_y, T* d_z,
                                    MPIDist dist, int numBlocks,
                                    int blockSize) {
    gpuLaunch(combine_density_vectors, numBlocks, blockSize, d_grad, d_x, d_y,
              d_z, dist);
}

template void launch_combine_density_vectors<complexDoubleDevice>(
    float3*, complexDoubleDevice*, complexDoubleDevice*, complexDoubleDevice*,
    MPIDist, int, int);
template void launch_combine_density_vectors<complexFloatDevice>(
    float3*, complexFloatDevice*, complexFloatDevice*, complexFloatDevice*,
    MPIDist, int, int);

__global__ void place_particles(float3* d_pos, float3* d_vel, const float3* d_grad, double delta, double dot_delta, double rl, double a, double deltaT, double fscal, int ng, MPIDist dist){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dist.local_grid_size())return;

    int3 local_coords = dist.local_coords(idx);

    float3 S = d_grad[idx];

    float3 pos = fmod((make_float3(local_coords.x,local_coords.y,local_coords.z) + (S * delta)) * (((float)ng)/(float)dist.ng()),(float)ng);

    float vel_a = a - (deltaT * 0.5f);
    float vel_mul = (vel_a * vel_a * dot_delta * fscal);
    float3 vel = S * vel_mul;

    d_pos[idx] = pos;
    d_vel[idx] = vel;
}

void launch_place_particles(float3* d_pos, float3* d_vel, const float3* d_grad, double delta, double dot_delta, double rl, double a, double deltaT, double fscal, int ng, MPIDist dist, int numBlocks, int blockSize){
    gpuLaunch(place_particles,numBlocks,blockSize,d_pos,d_vel,d_grad,delta,dot_delta,rl,a,deltaT,fscal,ng,dist);
}