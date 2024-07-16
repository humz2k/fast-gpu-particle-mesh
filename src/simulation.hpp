#ifndef _FGPM_SIMULATION_HPP_
#define _FGPM_SIMULATION_HPP_

#include "cosmo.hpp"
#include "params.hpp"
#include <cuda_runtime.h>
#include <cufft.h>

class Particles;
class Grid;

template <class T> class FFT;

template <class T> class FFT {
  public:
    FFT(int ng){};
    FFT(){};
    virtual ~FFT(){};
    virtual void forward(T* in, T* out) = 0;
    virtual void backward(T* in, T* out) = 0;
    virtual void forward(T* in) = 0;
    virtual void backward(T* in) = 0;
};

class Grid {
  public:
    Grid(const Params& params, int ng){};
    Grid(){};
    virtual ~Grid(){};
    virtual void solve() = 0;
    virtual void solve_gradient() = 0;
    virtual void CIC(const Particles& particles) = 0;
    virtual void generate_fourier_amplitudes(Cosmo& cosmo) = 0;
};

class Particles {
  public:
    Particles(const Params& params){};
    Particles(){};
    virtual ~Particles(){};
    virtual void update_positions() = 0;
    virtual void update_velocities(const Grid& grid) = 0;
};

#endif