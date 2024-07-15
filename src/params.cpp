#include "params.hpp"
#include "logging.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <bits/stdc++.h>
#include <assert.h>

static std::vector<std::string> split(const std::string &s){
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream token_stream(s);
    while(token_stream >> token){
        tokens.push_back(token);
    }
    return tokens;
}

Params::Params(std::string filename) : params_filename(filename){
    LOG_MINIMAL("reading %s",params_filename.c_str());

    std::ifstream file(params_filename);
    std::string line;

    assert(file.is_open());

    while (file.good()){
        std::getline(file,line);
        std::vector<std::string> tokens = split(line);
        if (tokens.size() == 0)continue;
        assert(tokens.size() >= 2);
        LOG_DEBUG("params line: %s %s",tokens[0].c_str(),tokens[1].c_str());

        if (tokens[0] == "NG"){
            ng = std::stoi(tokens[1]);
        } else if (tokens[0] == "NP"){
            np = std::stoi(tokens[1]);
        } else if (tokens[0] == "RL"){
            rl = std::stof(tokens[1]);
        } else if (tokens[0] == "SEED"){
            seed = std::stoi(tokens[1]);
        } else if (tokens[0] == "Z_IN"){
            z_ini = std::stof(tokens[1]);
        } else if (tokens[0] == "Z_FIN"){
            z_fin = std::stof(tokens[1]);
        } else if (tokens[0] == "N_STEPS"){
            nsteps = std::stoi(tokens[1]);
        } else if (tokens[0] == "OMEGA_CDM"){
            omega_cdm = std::stof(tokens[1]);
        } else if (tokens[0] == "DEUT"){
            deut = std::stof(tokens[1]);
        } else if (tokens[0] == "OMEGA_NU"){
            omega_nu = std::stof(tokens[1]);
        } else if (tokens[0] == "HUBBLE"){
            hubble = std::stof(tokens[1]);
        } else if (tokens[0] == "SS8"){
            ss8 = std::stof(tokens[1]);
        } else if (tokens[0] == "NS"){
            ns = std::stof(tokens[1]);
        } else if (tokens[0] == "W_DE"){
            w_de = std::stof(tokens[1]);
        } else if (tokens[0] == "WA_DE"){
            wa_de = std::stof(tokens[1]);
        } else if (tokens[0] == "T_CMB"){
            T_cmb = std::stof(tokens[1]);
        } else if (tokens[0] == "N_EFF_MASSLESS"){
            neff_massless = std::stof(tokens[1]);
        } else if (tokens[0] == "N_EFF_MASSIVE"){
            neff_massive = std::stof(tokens[1]);
        } else if (tokens[0] == "OUTPUT_BASE_NAME"){
            output_prefix = tokens[0];
        } else if (tokens[0] == "PK_DUMP"){
            for (size_t i = 1; i < tokens.size(); i++){
                pk_dumps[std::stoi(tokens[i])] = true;
            }
        } else if (tokens[0] == "PARTICLE_DUMP"){
            for (size_t i = 1; i < tokens.size(); i++){
                particle_dumps[std::stoi(tokens[i])] = true;
            }
        } else if (tokens[0] == "PK_FOLDS"){
            pk_folds = std::stoi(tokens[1]);
        }
    }

    assert((np > 0) && (ng > 0) && (rl > 0.0));

    omega_baryon = deut / hubble / hubble;
    omega_cb = omega_cdm + omega_baryon;
    omega_matter = omega_cb + omega_nu;

    omega_radiation = 2.471e-5*pow(T_cmb/2.725f,4.0f)/pow(hubble,2.0f);
    f_nu_massless = neff_massless*7.0/8.0*pow(4.0/11.0,4.0/3.0);
    f_nu_massive = neff_massive*7.0/8.0*pow(4.0/11.0,4.0/3.0);

    this->dump();

}

void Params::dump(){
    std::cout << std::endl << "###################################" << "\n";
    std::cout << "Params(" << params_filename << "):" << "\n";
    std::cout << "###################################" << "\n";
    std::cout << "NG " << ng << "\n";
    std::cout << "NP " << np << "\n";
    std::cout << "RL " << rl << "\n";
    std::cout << "SEED " << seed << "\n";
    std::cout << "Z_IN " << z_ini << "\n";
    std::cout << "Z_FIN " << z_fin << "\n";
    std::cout << "N_STEPS " << nsteps << "\n";
    std::cout << "OMEGA_CDM " << omega_cdm << "\n";
    std::cout << "DEUT " << deut << "\n";
    std::cout << "OMEGA_NU " << omega_nu << "\n";
    std::cout << "HUBBLE " << hubble << "\n";
    std::cout << "SS8 " << ss8 << "\n";
    std::cout << "NS " << ns << "\n";
    std::cout << "W_DE " << w_de << "\n";
    std::cout << "WA_DE " << wa_de << "\n";
    std::cout << "T_CMB " << T_cmb << "\n";
    std::cout << "N_EFF_MASSLESS " << neff_massless << "\n";
    std::cout << "N_EFF_MASSIVE " << neff_massive << "\n";
    std::cout << "OUTPUT_BASE_NAME " << output_prefix << "\n";
    std::cout << "PK_DUMP ";
    for (int i = 0; i < nsteps; i++){
        if (pk_dumps[i]) std::cout << i << " ";
    }
    std::cout << "\n";

    std::cout << "PARTICLE_DUMP ";
    for (int i = 0; i < nsteps; i++){
        if (particle_dumps[i]) std::cout << i << " ";
    }
    std::cout << "\n";

    std::cout << "PK_FOLDS " << pk_folds << "\n";

    std::cout << "###################################" << "\n";
    std::cout << "omega_baryon " << omega_baryon << "\n";
    std::cout << "omega_cb " << omega_cb << "\n";
    std::cout << "omega_matter " << omega_matter << "\n";
    std::cout << "omega_radiation " << omega_radiation << "\n";
    std::cout << "f_nu_massless " << f_nu_massless << "\n";
    std::cout << "f_nu_massive " << f_nu_massive << "\n";
    std::cout << "###################################" << std::endl;
}