#include <cuda.h>

#include "params.hpp"
#include "cosmo.hpp"
#include "timestepper.hpp"

class Grid{
    public:
        Grid(int ng);
        ~Grid();
        void solve();
        void solve_gradient();
};

class Particles{
    public:
        Particles(int np);
        ~Particles();

        void do_cic(Grid& grid);
        void update_velocities(Grid& grid, Timestepper& ts);
        void update_positions(Timestepper& ts);
};