#ifndef _FGPM_GRID_HPP_
#define _FGPM_GRID_HPP_

#include <cuda_runtime.h>
#include <cufft.h>
#include "params.hpp"

class Grid {
    private:
        const Params& m_params;
        float* m_d_greens = NULL;
        float4* m_d_grad = NULL;
        cufftDoubleComplex* m_d_grid = NULL;
        cufftDoubleComplex* m_d_x = NULL;
        cufftDoubleComplex* m_d_y = NULL;
        cufftDoubleComplex* m_d_z = NULL;
  public:
    Grid(const Params& params);
    ~Grid();
    void solve();
    void solve_gradient();
};

/*template<class cmplx, class real>
class SimpleGrid : public Grid{
    private:
        float4* m_d_grad;
        float* m_d_greens;
        cufftDoubleComplex* m_d_grid;
        cufftDoubleComplex* m_d_x;
        cufftDoubleComplex* m_d_y;
        cufftDoubleComplex* m_d_z;

    public:
        SimpleGrid(const Params& params);
        ~SimpleGrid();
        void solve();
        void solve_gradient();
};*/

#endif