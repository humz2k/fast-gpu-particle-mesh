#ifndef _FGPM_GRID_HPP_
#define _FGPM_GRID_HPP_

#include "simulation.hpp"
#include "serial_fft.hpp"

template<class fft_t>
class SimpleGrid : public Grid {
  private:
    int m_ng;
    size_t m_size;
    const Params& m_params;
    float4* m_d_grad;
    float* m_d_greens;
    fft_t* m_d_grid;
    fft_t* m_d_x;
    fft_t* m_d_y;
    fft_t* m_d_z;
    SerialFFT<fft_t> fft;

  public:
    SimpleGrid(const Params& params, int ng);
    ~SimpleGrid();
    void solve();
    void solve_gradient();
    void CIC(const Particles& particles);
    void generate_fourier_amplitudes(Cosmo& cosmo);
};

#endif