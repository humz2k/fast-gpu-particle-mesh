#ifndef _FGPM_MEM_MIN_GRID_HPP_
#define _FGPM_MEM_MIN_GRID_HPP_

#include "mpi_distribution.hpp"
#include "serial_fft.hpp"
#include "simulation.hpp"

template <class fft_t> class MemoryMinGrid : public Grid {
  private:
    int m_ng;
    size_t m_size;
    const Params& m_params;
    float3* m_d_grad;
    fft_t* m_d_grid;
    SerialFFT<fft_t> fft;
    MPIDist m_dist;

  public:
    MemoryMinGrid(const Params& params, int ng);
    ~MemoryMinGrid();
    void solve();
    void solve_gradient();
    void CIC(const Particles<float3>& particles);
    void generate_fourier_amplitudes(Cosmo& cosmo);
    void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                  Particles<float3>& particles);
    MPIDist dist() const;
    std::vector<double> bin(int nbins) const;
    double k_min() const;
    double k_max() const;
    void forward();
    void backward();
    const float3* grad() const;
};

#endif