#include "power_spectrum.hpp"
#include "logging.hpp"
#include "simulation.hpp"
#include <assert.h>
#include <fstream>
#include <iostream>

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

PowerSpectrum::PowerSpectrum(const Grid& grid, int nbins)
    : m_k_min(grid.k_min()), m_k_max(grid.k_max()), m_k_bins(nbins),
      m_k_delta((m_k_max - m_k_min) / ((double)m_k_bins)),
      m_h_values(grid.bin(nbins)) {
    m_k_min += m_k_delta * 0.5;
    m_k_max += m_k_delta * 0.5;
}

PowerSpectrum::PowerSpectrum(Particles<float3>& particles, Grid& grid,
                             int nbins, int nfolds)
    : m_k_min(grid.k_min(nfolds)), m_k_max(grid.k_max(nfolds)), m_k_bins(nbins),
      m_k_delta((m_k_max - m_k_min) / ((double)m_k_bins)) {
    m_k_min += m_k_delta * 0.5;
    m_k_max += m_k_delta * 0.5;

    if (nfolds > 0) {
        particles.fold(nfolds);
    }

    grid.CIC(particles);
    grid.forward();
    m_h_values = grid.bin(nbins);

    if (nfolds > 0) {
        particles.unfold();
    }
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

    output << "k,v\n";

    for (int i = 0; i < m_k_bins; i++) {
        double k_bin = m_k_min + m_k_delta * (((float)i));
        double value = m_h_values[i];
        output << k_bin << "," << value << "\n";
    }

    output.close();
}