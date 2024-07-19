#include "simple_grid.hpp"
#include "simple_particles.hpp"
#include "simulation.hpp"

int main() {

    run_simulation<SimpleParticles, SimpleGrid<complexFloatDevice>>(
        "test.params");

    return 0;
}