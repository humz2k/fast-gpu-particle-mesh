#include "allocators.hpp"
#include "cosmo.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "simple_grid.hpp"
#include "simple_particles.hpp"
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

    SimpleParticles particles(params, cosmo, ts);
    particles.dump("particles.csv");
    SimpleGrid<complexDoubleDevice> grid(params, params.ng());
    grid.CIC(particles);
    grid.forward();
    // grid.generate_fourier_amplitudes(cosmo);

    PowerSpectrum ic_power(grid, 10);
    ic_power.to_csv("test2.csv");

    particles.update_positions(ts,0.5f);
    grid.CIC(particles);

    //grid.forward();

    //PowerSpectrum power2(grid, 10);
    //power2.to_csv("test3.csv");

    LOG_MINIMAL("done!");
    return 0;
}