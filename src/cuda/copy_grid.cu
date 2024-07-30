#include "copy_grid.hpp"
#include "event_logger.hpp"
#include "gpu.hpp"
#include <string>
#include <typeinfo>

template <class T1, class T2>
__global__ void copy_grid(const T1* __restrict in, T2* __restrict out,
                          size_t n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n)
        return;

    T1 input = in[idx];
    T2 output;
    output.x = input.x;
    output.y = input.y;
    out[idx] = output;
}

template <class T1, class T2>
void launch_copy_grid(const T1* in, T2* out, size_t n, int numBlocks,
                      int blockSize) {
    std::string timer_name = "copy_grid_" + std::string(typeid(T1).name()) +
                             "_" + std::string(typeid(T2).name());
    events.timers[timer_name].start();
    gpuLaunch(copy_grid, numBlocks, blockSize, in, out, n);
    events.timers[timer_name].end();
}

template void launch_copy_grid<complexHalfDevice, complexHalfDevice>(
    const complexHalfDevice*, complexHalfDevice*, size_t, int, int);

template void launch_copy_grid<complexFloatDevice, complexHalfDevice>(
    const complexFloatDevice*, complexHalfDevice*, size_t, int, int);

template void launch_copy_grid<complexHalfDevice, complexFloatDevice>(
    const complexHalfDevice*, complexFloatDevice*, size_t, int, int);

template void launch_copy_grid<complexDoubleDevice, complexHalfDevice>(
    const complexDoubleDevice*, complexHalfDevice*, size_t, int, int);

template void launch_copy_grid<complexHalfDevice, complexDoubleDevice>(
    const complexHalfDevice*, complexDoubleDevice*, size_t, int, int);

template void launch_copy_grid<complexFloatDevice, complexFloatDevice>(
    const complexFloatDevice*, complexFloatDevice*, size_t, int, int);

template void launch_copy_grid<complexDoubleDevice, complexFloatDevice>(
    const complexDoubleDevice*, complexFloatDevice*, size_t, int, int);

template void launch_copy_grid<complexFloatDevice, complexDoubleDevice>(
    const complexFloatDevice*, complexDoubleDevice*, size_t, int, int);

template void launch_copy_grid<complexDoubleDevice, complexDoubleDevice>(
    const complexDoubleDevice*, complexDoubleDevice*, size_t, int, int);