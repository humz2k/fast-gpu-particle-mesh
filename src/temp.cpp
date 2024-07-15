#include <cuda.h>

#include "cosmo.hpp"
#include "params.hpp"
#include "timestepper.hpp"
#include "grid.hpp"

class Particles {
  public:
    Particles(const Params& params);
    ~Particles();

    void cic(Grid& grid);
    void update_velocities(Grid& grid, Timestepper& ts);
    void update_positions(Timestepper& ts);
};