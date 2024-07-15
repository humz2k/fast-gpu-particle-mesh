#include "cosmo.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "timestepper.hpp"
#include "allocators.hpp"

int main() {
    //gpuCall(gpuFree(0));

    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    //Params params("test.params");
    //Timestepper ts(params);
    //Cosmo cosmo(params);

    //void* test; alloc_gpu(&test,sizeof(float));
    //free_gpu(test);

    LOG_MINIMAL("done!");
    return 0;
}