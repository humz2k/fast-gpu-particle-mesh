#ifndef _FGPM_COSMO_HPP_
#define _FGPM_COSMO_HPP_

#include "params.hpp"

class Cosmo{
    private:
        const Params& m_params;
        double* pk = NULL;

    public:
        Cosmo(const Params& params);
};

#endif