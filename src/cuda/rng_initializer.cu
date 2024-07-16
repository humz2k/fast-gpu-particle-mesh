#include "gpu.hpp"
#include "logging.hpp"
#include "rng_initializer.hpp"
#include <curand.h>
#include <curand_kernel.h>

template <class T>
__global__ void generate_real_random(T* __restrict grid, int seed,
                                     MPIDist dist) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= dist.m_local_grid_size.x * dist.m_local_grid_size.y *
                   dist.m_local_grid_size.z)
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
void launch_generate_real_random(T* d_grid, int seed, MPIDist dist,
                                 int numBlocks, int blockSize) {
    gpuLaunch(generate_real_random, numBlocks, blockSize, d_grid, seed, dist);
}

template void
launch_generate_real_random<complexDoubleDevice>(complexDoubleDevice*, int,
                                                 MPIDist, int, int);
template void
launch_generate_real_random<complexFloatDevice>(complexFloatDevice*, int,
                                                MPIDist, int, int);