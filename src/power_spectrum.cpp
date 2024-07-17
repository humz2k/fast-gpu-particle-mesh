#include "power_spectrum.hpp"
#include "logging.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>

PowerSpectrum::PowerSpectrum(const std::string& filename) {
    LOG_DEBUG("reading ipk %s", filename.c_str());

    double header[20];
    FILE* ptr = fopen(filename.c_str(), "rb");
    assert(ptr);

    assert(fread(header, sizeof(header), 1, ptr) == 1);

    m_k_min = header[0];
    m_k_max = header[1];
    m_k_bins = header[2];
    m_k_delta = header[3];

    m_h_values.resize(m_k_bins + 20);
    assert(fread(m_h_values.data(), sizeof(double), m_k_bins, ptr) ==
           (size_t)m_k_bins);

    fclose(ptr);
}

double PowerSpectrum::k_min() const { return m_k_min; }

double PowerSpectrum::k_max() const { return m_k_max; }

double PowerSpectrum::k_delta() const { return m_k_delta; }

int PowerSpectrum::k_bins() const { return m_k_bins; }

const std::vector<double>& PowerSpectrum::h_values() const {
    return m_h_values;
}

void PowerSpectrum::to_csv(std::string filename) const {
    std::ofstream output(filename);

    std::cout << "k_min = " << m_k_min << ", k_max = " << m_k_max << std::endl;

    output << "k,v\n";

    for (int i = 0; i < m_k_bins; i++){
        double k_bin = m_k_min + m_k_delta * (((float)i));
        double value = m_h_values[i];
        output << k_bin << "," << value << "\n";
    }

    output.close();
}