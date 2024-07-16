#include "cosmo.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "timestepper.hpp"
#include "allocators.hpp"
#include "simulation.hpp"
#include "grid.hpp"

int main() {
    gpuCall(gpuFree(0));

    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    Params params("test.params");
    Timestepper ts(params);
    Cosmo cosmo(params);

    SimpleGrid grid(params);

    void* test; gpu_allocator.alloc(&test,sizeof(float));
    gpu_allocator.free(test);

    LOG_MINIMAL("done!");
    return 0;
}