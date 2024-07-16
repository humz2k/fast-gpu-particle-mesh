#ifndef _FGPM_GRID_HPP_
#define _FGPM_GRID_HPP_

#include "simulation.hpp"

class SimpleGrid : public Grid {
  private:
    int m_ng;
    size_t m_size;
    const Params& m_params;
    float4* m_d_grad;
    float* m_d_greens;
    cufftDoubleComplex* m_d_grid;
    cufftDoubleComplex* m_d_x;
    cufftDoubleComplex* m_d_y;
    cufftDoubleComplex* m_d_z;

  public:
    SimpleGrid(const Params& params, int ng);
    ~SimpleGrid();
    void solve();
    void solve_gradient();
    void CIC(const Particles& particles);
    void generate_fourier_amplitudes(Cosmo& cosmo);
};

#endif