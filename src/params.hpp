#ifndef _FGPM_PARAMS_HPP_
#define _FGPM_PARAMS_HPP_

#include <string>
#include <vector>
#include <unordered_map>

#define MAX_STEPS 625

class Params{
    private:
        std::string params_filename;

        int ng = 0;
        int np = 0;
        double rl = 0.0;
        int seed = 0;

        double z_ini = 200.0;
        double z_fin = 0.0;
        int nsteps = 625;

        double omega_cdm = 0.22;
        double deut = 0.02258;
        double omega_nu = 0.0;
        double hubble = 0.71;
        double ss8 = 0.8;
        double ns = 0.963;
        double w_de = -1.0;
        double wa_de = 0.0;
        double T_cmb = 2.726;
        double neff_massless = 3.04;
        double neff_massive = 0.0;

        double omega_baryon;
        double omega_cb;
        double omega_matter;
        double omega_radiation;
        double f_nu_massless;
        double f_nu_massive;

        int pk_folds = 0;

        std::string output_prefix = "run";

        std::unordered_map<int,bool> pk_dumps;
        std::unordered_map<int,bool> particle_dumps;

    public:
        Params(std::string filename);
        void dump();
};
#endif