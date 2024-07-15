#include <cuda.h>

#include "params.hpp"

class Cosmo{
    private:
        float Omega_m;
        float Omega_cdm;
        float Omega_bar;
        float Omega_cb;
        float Omega_nu;
        float f_nu_massless;
        float f_nu_massive;
        float Omega_r;
        float h;
        float w_de;
        float wa_de;

    public:
        Cosmo(Params& params);
        ~Cosmo();
};

class Timestepper{
    private:
        Params& params;
        double aa;
        double z;
        double deltaT;
        double adot;
        double fscal;
    public:
        Timestepper(Params& _params);

        void set_initial_a(double a);
        void set_initial_z(double z);
        void advance_half_step();
        void reverse_half_step();

        double get_aa();
        double get_z();
        double get_deltaT();
        double get_adot();
        double get_fscal();
};

class Grid{
    public:
        Grid(int ng);
        ~Grid();
        void solve();
        void solve_gradient();
};

class Particles{
    public:
        Particles(int np);
        ~Particles();

        void do_cic(Grid& grid);
        void update_velocities(Grid& grid, Timestepper& ts);
        void update_positions(Timestepper& ts);
};