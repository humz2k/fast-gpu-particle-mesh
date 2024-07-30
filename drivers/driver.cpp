#include "fp16_fft.hpp"
#include "simple_grid.hpp"
#include "simple_particles.hpp"
#include "simulation.hpp"

int main() {

    run_simulation<SimpleParticles, SimpleGrid<SerialFFT, complexFloatDevice>>(
        "test.params");

    events.dump();
    return 0;
}