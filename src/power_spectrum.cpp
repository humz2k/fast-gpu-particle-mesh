#include "power_spectrum.hpp"
#include <assert.h>
#include "logging.hpp"

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