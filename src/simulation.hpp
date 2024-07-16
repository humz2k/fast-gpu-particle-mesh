#ifndef _FGPM_SIMULATION_HPP_
#define _FGPM_SIMULATION_HPP_

#include <cuda_runtime.h>
#include <cufft.h>
#include "params.hpp"

class Particles;
class Grid;

class Grid {
  public:
    Grid(const Params& params);
    Grid();
    virtual ~Grid();
    virtual void solve() = 0;
    virtual void solve_gradient() = 0;
    virtual void CIC(const Particles& particles) = 0;
};

class Particles{
    public:
        Particles(const Params& params);
        Particles();
        virtual ~Particles();
};

#endif