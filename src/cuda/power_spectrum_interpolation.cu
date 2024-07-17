#include "gpu.hpp"
#include "mpi_distribution.hpp"
#include "power_spectrum_interpolation.hpp"

/**
 * @brief CUDA kernel for interpolating the power spectrum.
 *
 * This kernel interpolates the power spectrum values for a given grid. It
 * calculates the k-modes and interpolates the power spectrum value for each
 * grid point.
 *
 * @tparam T The type of the elements in the output array.
 * @param out Pointer to the output array where interpolated values will be
 * stored.
 * @param in Pointer to the input array containing power spectrum values.
 * @param k_delta The spacing between k bins in the power spectrum.
 * @param k_min The minimum k value in the power spectrum.
 * @param rl The size of the simulation box.
 * @param dist The MPIDist object containing distribution and grid information.
 */
template <class T>
__global__ void interp_pk(T* out, double* in, double k_delta, double k_min,
                          double rl, const MPIDist dist) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dist.local_grid_size())
        return;

    int3 idx3d = dist.global_coords(idx);

    // pole
    if ((idx3d.x == 0) && (idx3d.y == 0) && (idx3d.z == 0)) {
        out[idx] = 0;
        return;
    }

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

    out[idx] = y;
}

template <class T>
void launch_interp_pk(T* out, double* in, double k_delta, double k_min,
                      double rl, const MPIDist dist, int numBlocks,
                      int blockSize) {
    gpuLaunch(interp_pk, numBlocks, blockSize, out, in, k_delta, k_min, rl,
              dist);
}

template void launch_interp_pk<double>(double*, double*, double, double, double,
                                       const MPIDist, int, int);

template void launch_interp_pk<float>(float*, double*, double, double, double,
                                      const MPIDist, int, int);