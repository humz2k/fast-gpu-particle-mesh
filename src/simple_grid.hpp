#ifndef _FGPM_SIMPLE_GRID_HPP_
#define _FGPM_SIMPLE_GRID_HPP_

#include "serial_grid.hpp"

template<class fft_t>
class SimpleGrid : public SerialGrid<fft_t>{
    public:
        using SerialGrid<fft_t>::SerialGrid;//(Params& params, int ng);
        void solve();
        void solve_gradient();
        void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                          Particles<float3>& particles);
};

#endif