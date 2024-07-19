#include "params.hpp"
#include "logging.hpp"
#include <assert.h>
#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

/**
 * @brief Splits a string into a vector of words based on whitespace.
 *
 * This function takes a string and splits it into individual words using
 * whitespace as the delimiter. Each word is stored in a vector of strings.
 *
 * @param s The input string to be split.
 * @return A vector of strings containing the words from the input string.
 */
static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream token_stream(s);
    while (token_stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

Params::Params(std::string filename) : m_params_filename(filename) {
    LOG_MINIMAL("reading %s", m_params_filename.c_str());

    std::ifstream file(m_params_filename);
    std::string line;

    assert(file.is_open());

    while (file.good()) {
        std::getline(file, line);
        std::vector<std::string> tokens = split(line);
        if (tokens.size() == 0)
            continue;
        assert(tokens.size() >= 2);
        LOG_DEBUG("params line: %s %s", tokens[0].c_str(), tokens[1].c_str());

        if (tokens[0] == "NG") {
            m_ng = std::stoi(tokens[1]);
        } else if (tokens[0] == "NP") {
            m_np = std::stoi(tokens[1]);
        } else if (tokens[0] == "RL") {
            m_rl = std::stof(tokens[1]);
        } else if (tokens[0] == "SEED") {
            m_seed = std::stoi(tokens[1]);
        } else if (tokens[0] == "Z_IN") {
            m_z_ini = std::stof(tokens[1]);
        } else if (tokens[0] == "Z_FIN") {
            m_z_fin = std::stof(tokens[1]);
        } else if (tokens[0] == "N_STEPS") {
            m_nsteps = std::stoi(tokens[1]);
        } else if (tokens[0] == "OMEGA_CDM") {
            m_omega_cdm = std::stof(tokens[1]);
        } else if (tokens[0] == "DEUT") {
            m_deut = std::stof(tokens[1]);
        } else if (tokens[0] == "OMEGA_NU") {
            m_omega_nu = std::stof(tokens[1]);
        } else if (tokens[0] == "HUBBLE") {
            m_hubble = std::stof(tokens[1]);
        } else if (tokens[0] == "SS8") {
            m_ss8 = std::stof(tokens[1]);
        } else if (tokens[0] == "NS") {
            m_ns = std::stof(tokens[1]);
        } else if (tokens[0] == "W_DE") {
            m_w_de = std::stof(tokens[1]);
        } else if (tokens[0] == "WA_DE") {
            m_wa_de = std::stof(tokens[1]);
        } else if (tokens[0] == "T_CMB") {
            m_T_cmb = std::stof(tokens[1]);
        } else if (tokens[0] == "N_EFF_MASSLESS") {
            m_neff_massless = std::stof(tokens[1]);
        } else if (tokens[0] == "N_EFF_MASSIVE") {
            m_neff_massive = std::stof(tokens[1]);
        } else if (tokens[0] == "OUTPUT_BASE_NAME") {
            m_output_prefix = tokens[0];
        } else if (tokens[0] == "PK_DUMP") {
            for (size_t i = 1; i < tokens.size(); i++) {
                m_pk_dumps[std::stoi(tokens[i])] = true;
            }
        } else if (tokens[0] == "PARTICLE_DUMP") {
            for (size_t i = 1; i < tokens.size(); i++) {
                m_particle_dumps[std::stoi(tokens[i])] = true;
            }
        } else if (tokens[0] == "PK_FOLDS") {
            m_pk_folds = std::stoi(tokens[1]);
        } else if (tokens[0] == "IPK") {
            m_ipk = tokens[1];
        }
    }

    assert((m_np > 0) && (m_ng > 0) && (m_rl > 0.0));

    m_omega_baryon = m_deut / m_hubble / m_hubble;
    m_omega_cb = m_omega_cdm + m_omega_baryon;
    m_omega_matter = m_omega_cb + m_omega_nu;

    m_omega_radiation =
        2.471e-5 * pow(m_T_cmb / 2.725f, 4.0f) / pow(m_hubble, 2.0f);
    m_f_nu_massless = m_neff_massless * 7.0 / 8.0 * pow(4.0 / 11.0, 4.0 / 3.0);
    m_f_nu_massive = m_neff_massive * 7.0 / 8.0 * pow(4.0 / 11.0, 4.0 / 3.0);

    this->dump();
}

void Params::dump() {
    std::cout << std::endl
              << "###################################"
              << "\n";
    std::cout << "Params(" << m_params_filename << "):"
              << "\n";
    std::cout << "###################################"
              << "\n";
    std::cout << "NG " << m_ng << "\n";
    std::cout << "NP " << m_np << "\n";
    std::cout << "RL " << m_rl << "\n";
    std::cout << "SEED " << m_seed << "\n";
    std::cout << "Z_IN " << m_z_ini << "\n";
    std::cout << "Z_FIN " << m_z_fin << "\n";
    std::cout << "N_STEPS " << m_nsteps << "\n";
    std::cout << "OMEGA_CDM " << m_omega_cdm << "\n";
    std::cout << "DEUT " << m_deut << "\n";
    std::cout << "OMEGA_NU " << m_omega_nu << "\n";
    std::cout << "HUBBLE " << m_hubble << "\n";
    std::cout << "SS8 " << m_ss8 << "\n";
    std::cout << "NS " << m_ns << "\n";
    std::cout << "W_DE " << m_w_de << "\n";
    std::cout << "WA_DE " << m_wa_de << "\n";
    std::cout << "T_CMB " << m_T_cmb << "\n";
    std::cout << "N_EFF_MASSLESS " << m_neff_massless << "\n";
    std::cout << "N_EFF_MASSIVE " << m_neff_massive << "\n";
    std::cout << "OUTPUT_BASE_NAME " << m_output_prefix << "\n";
    std::cout << "PK_DUMP ";
    for (int i = 0; i < m_nsteps; i++) {
        if (m_pk_dumps[i])
            std::cout << i << " ";
    }
    std::cout << "\n";

    std::cout << "PARTICLE_DUMP ";
    for (int i = 0; i < m_nsteps; i++) {
        if (m_particle_dumps[i])
            std::cout << i << " ";
    }
    std::cout << "\n";

    std::cout << "PK_FOLDS " << m_pk_folds << "\n";
    std::cout << "IPK " << m_ipk << "\n";

    std::cout << "###################################"
              << "\n";
    std::cout << "omega_baryon " << m_omega_baryon << "\n";
    std::cout << "omega_cb " << m_omega_cb << "\n";
    std::cout << "omega_matter " << m_omega_matter << "\n";
    std::cout << "omega_radiation " << m_omega_radiation << "\n";
    std::cout << "f_nu_massless " << m_f_nu_massless << "\n";
    std::cout << "f_nu_massive " << m_f_nu_massive << "\n";
    std::cout << "###################################" << std::endl;
}

double Params::z_ini() const { return m_z_ini; }

double Params::z_fin() const { return m_z_fin; }

int Params::nsteps() const { return m_nsteps; }

int Params::ng() const { return m_ng; }

int Params::np() const { return m_np; }

double Params::rl() const { return m_rl; }

int Params::seed() const { return m_seed; }

double Params::w_de() const { return m_w_de; }

double Params::wa_de() const { return m_wa_de; }

double Params::omega_cb() const { return m_omega_cb; }

double Params::f_nu_massless() const { return m_f_nu_massless; }

double Params::f_nu_massive() const { return m_f_nu_massive; }

double Params::omega_radiation() const { return m_omega_radiation; }

double Params::omega_matter() const { return m_omega_matter; }

double Params::omega_nu() const { return m_omega_nu; }

const std::string& Params::ipk() const { return m_ipk; }

int Params::pk_n_bins() const {return m_pk_n_bins; }

bool Params::pk_dump(int step) { return m_pk_dumps[step]; }