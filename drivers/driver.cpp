#include "allocators.hpp"
#include "cosmo.hpp"
#include "simple_grid.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "simulation.hpp"
#include "timestepper.hpp"

int main() {
    gpuCall(gpuFree(0));

    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    Params params("test.params");
    Timestepper ts(params);
    Cosmo cosmo(params);

    SimpleGrid<complexDoubleDevice> simple_grid(params, params.ng());
    Grid& grid = simple_grid;

    grid.generate_fourier_amplitudes(cosmo);

    void* test;
    gpu_allocator.alloc(&test, sizeof(float));
    gpu_allocator.free(test);

    LOG_MINIMAL("done!");
    return 0;
}