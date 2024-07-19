#include "allocators.hpp"
#include "cosmo.hpp"
#include "event_logger.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "simple_grid.hpp"
#include "simple_particles.hpp"
#include "simulation.hpp"
#include "timestepper.hpp"
#include <string.h>

int main() {
    events.timers["dtot"].start();

    gpuCall(gpuFree(0));

    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    events.timers["dinit"].start();

    Params params("test.params");
    Timestepper ts(params);
    Cosmo cosmo(params);
    cosmo.initial_pk().to_csv("test.csv");

    SimpleParticles particles(params, cosmo, ts);

    events.timers["dinit"].end();

    SimpleGrid<complexDoubleDevice> grid(params, params.ng());
    grid.CIC(particles);
    grid.forward();
    PowerSpectrum ic_power(grid, params.pk_n_bins());
    ic_power.to_csv("test2.csv");

    for (int step = 0; step < params.nsteps(); step++) {
        LOG_MINIMAL("STEP %d", step);
        events.timers["dstep"].start();

        events.timers["dpos"].start();
        particles.update_positions(ts, 0.5f);
        events.timers["dpos"].end();

        events.timers["dcic"].start();
        grid.CIC(particles);
        events.timers["dcic"].end();

        events.timers["dgrad"].start();
        grid.solve_gradient();
        events.timers["dgrad"].end();

        ts.advance_half_step();

        events.timers["dvel"].start();
        particles.update_velocities(grid, ts, 1.0f);
        events.timers["dvel"].end();

        ts.advance_half_step();

        events.timers["dpos"].start();
        particles.update_positions(ts, 0.5f);
        events.timers["dpos"].end();

        if (params.pk_dump(step)) {
            LOG_MINIMAL("dumping pk");
            events.timers["dpk"].start();
            grid.CIC(particles);
            grid.forward();
            PowerSpectrum(grid, params.pk_n_bins())
                .to_csv("steps/step" + std::to_string(step) + ".csv");
            events.timers["dpk"].end();
        }

        events.timers["dstep"].end();
    }

    grid.CIC(particles);
    grid.forward();
    PowerSpectrum power(grid, params.pk_n_bins());
    power.to_csv("final.csv");

    LOG_MINIMAL("done!");

    events.timers["dtot"].end();

    return 0;
}