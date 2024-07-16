#ifndef _FGPM_POWER_SPECTRUM_HPP_
#define _FGPM_POWER_SPECTRUM_HPP_

#include <string>
#include <vector>

/**
 * @class PowerSpectrum
 * @brief Represents the power spectrum used in cosmological simulations.
 *
 * The PowerSpectrum class handles the loading and storing of power spectrum
 * data, including the wave number range, binning, and associated values.
 */
class PowerSpectrum {
  private:
    double m_k_min;   /**< Minimum wave number. */
    double m_k_max;   /**< Maximum wave number. */
    int m_k_bins;     /**< Number of bins in the wave number range. */
    double m_k_delta; /**< Delta value for the wave number. */

    std::vector<double> m_h_values; /**< CPU values of the power spectrum. */

  public:
    /**
     * @brief Constructs a PowerSpectrum object and loads data from a file.
     *
     * @param file The filename from which to load the power spectrum data.
     */
    PowerSpectrum(const std::string& file);

    /**
     * @brief Gets the minimum wave number.
     *
     * @return The minimum wave number.
     */
    double k_min() const;

    /**
     * @brief Gets the maximum wave number.
     *
     * @return The maximum wave number.
     */
    double k_max() const;

    /**
     * @brief Gets the delta value for the wave number.
     *
     * @return The delta value for the wave number.
     */
    double k_delta() const;

    /**
     * @brief Gets the number of bins in the wave number range.
     *
     * @return The number of bins in the wave number range.
     */
    int k_bins() const;

    /**
     * @brief Gets the values of the power spectrum.
     *
     * @return A const reference to the vector of power spectrum values.
     */
    const std::vector<double>& h_values() const;
};

#endif
