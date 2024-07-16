#include "gpu.hpp"
#include "logging.hpp"
#include "rng_initializer.hpp"
#include <curand.h>
#include <curand_kernel.h>

template <class T>
__global__ void generate_real_random(T* __restrict grid, int seed, int offset,
                                     int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;

    curandState state;
    curand_init(seed, idx + offset, 0, &state);

    double amp = curand_normal_double(&state);
    T out;
    out.x = amp;
    out.y = 0;
    grid[idx] = out;
}

template <class T>
void launch_generate_real_random(T* d_grid, int seed, int offset, int size,
                                 int numBlocks, int blockSize) {
    gpuLaunch(generate_real_random, numBlocks, blockSize, d_grid, seed, offset,
              size);
}

template void
launch_generate_real_random<complexDoubleDevice>(complexDoubleDevice*, int, int,
                                                 int, int, int);
template void
launch_generate_real_random<complexFloatDevice>(complexFloatDevice*, int, int,
                                                int, int, int);