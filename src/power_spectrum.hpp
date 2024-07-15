#ifndef _FGPM_POWER_SPECTRUM_HPP_
#define _FGPM_POWER_SPECTRUM_HPP_

#include <string>
#include <vector>

class PowerSpectrum {
  private:
    double m_k_min;
    double m_k_max;
    int m_k_bins;
    double m_k_delta;

    std::vector<double> m_h_values;

  public:
    PowerSpectrum(const std::string& file);
};

#endif
