#ifndef _FGPM_PARAMS_HPP_
#define _FGPM_PARAMS_HPP_

#include <string>
#include <vector>
#include <unordered_map>

#define MAX_STEPS 625

class Params{
    private:
        std::string m_params_filename;

        int m_ng = 0;
        int m_np = 0;
        double m_rl = 0.0;
        int m_seed = 0;

        double m_z_ini = 200.0;
        double m_z_fin = 0.0;
        int m_nsteps = 625;

        double m_omega_cdm = 0.22;
        double m_deut = 0.02258;
        double m_omega_nu = 0.0;
        double m_hubble = 0.71;
        double m_ss8 = 0.8;
        double m_ns = 0.963;
        double m_w_de = -1.0;
        double m_wa_de = 0.0;
        double m_T_cmb = 2.726;
        double m_neff_massless = 3.04;
        double m_neff_massive = 0.0;

        double m_omega_baryon;
        double m_omega_cb;
        double m_omega_matter;
        double m_omega_radiation;
        double m_f_nu_massless;
        double m_f_nu_massive;

        int m_pk_folds = 0;

        std::string m_output_prefix = "run";
        std::string m_ipk;

        std::unordered_map<int,bool> m_pk_dumps;
        std::unordered_map<int,bool> m_particle_dumps;

    public:
        Params(std::string filename);
        void dump();

        int ng() const;
        int np() const;
        double rl() const;
        int seed() const;
        double z_ini() const;
        double z_fin() const;
        int nsteps() const;

        double w_de() const;
        double wa_de() const;

        double omega_cb() const;
        double f_nu_massless() const;
        double f_nu_massive() const;
        double omega_radiation() const;
        double omega_matter() const;
        double omega_nu() const;

        const std::string& ipk() const;
};
#endif