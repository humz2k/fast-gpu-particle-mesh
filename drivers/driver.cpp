#include "allocators.hpp"
#include "cosmo.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "simple_grid.hpp"
#include "simple_particles.hpp"
#include "simulation.hpp"
#include "timestepper.hpp"
#include <string.h>

int main() {
    gpuCall(gpuFree(0));

    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    Params params("test.params");
    Timestepper ts(params);
    Cosmo cosmo(params);
    cosmo.initial_pk().to_csv("test.csv");

    SimpleParticles particles(params, cosmo, ts);
    //particles.dump("particles.csv");
    SimpleGrid<complexDoubleDevice> grid(params, params.ng());
    grid.CIC(particles);
    grid.forward();
    PowerSpectrum ic_power(grid, 221);
    ic_power.to_csv("test2.csv");

    for (int i = 0; i < params.nsteps(); i++){

        particles.update_positions(ts,0.5f);
        grid.CIC(particles);
        grid.solve_gradient();

        ts.advance_half_step();

        particles.update_velocities(grid,ts,1.0f);

        ts.advance_half_step();

        particles.update_positions(ts,0.5f);

        if ((i % 50) == 0){
            PowerSpectrum(grid, 221).to_csv("steps/step" + std::to_string(i) + ".csv");
        }

    }

    grid.CIC(particles);
    grid.forward();
    PowerSpectrum power(grid, 221);
    power.to_csv("final.csv");

    //particles.dump("particles1.csv");

    LOG_MINIMAL("done!");
    return 0;
}