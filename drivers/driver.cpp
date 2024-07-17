#include "allocators.hpp"
#include "cosmo.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "simple_grid.hpp"
#include "simulation.hpp"
#include "timestepper.hpp"

int main() {
    gpuCall(gpuFree(0));

    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    Params params("test.params");
    Timestepper ts(params);
    Cosmo cosmo(params);
    cosmo.initial_pk().to_csv("test.csv");

    SimpleGrid<complexDoubleDevice> simple_grid(params, params.ng());
    Grid& grid = simple_grid;

    grid.generate_displacement_ic_grad(cosmo,ts);

    //PowerSpectrum pk(grid,10);
    //pk.to_csv("test2.csv");

    LOG_MINIMAL("done!");
    return 0;
}