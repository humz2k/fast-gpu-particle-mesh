#ifndef _FGPM_SERIAL_FFT_HPP_
#define _FGPM_SERIAL_FFT_HPP_

#include "gpu.hpp"
#include "simulation.hpp"

/**
 * @class SerialFFT
 * @brief Implements Fast Fourier Transform (FFT) operations for serial
 * execution.
 *
 * The SerialFFT class provides methods for performing forward and backward FFTs
 * in both in-place and out-of-place modes. It is derived from the FFT base
 * class.
 *
 * @tparam T The type of the elements in the FFT (e.g., float, double).
 */
template <class T> class SerialFFT : public FFT<T> {
  private:
    int m_ng;            /**< Size of the grid. */
    gpufftHandle m_plan; /**< Handle for the FFT plan. */

    /**
     * @brief Performs the FFT operation in a specified direction.
     *
     * This private method handles the actual FFT computation.
     *
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     * @param direction The direction of the FFT (forward or backward).
     */
    void fft(T* in, T* out, int direction);

  public:
    /**
     * @brief Constructs a SerialFFT object with the specified grid size.
     *
     * @param ng The size of the grid.
     */
    SerialFFT(int ng);

    /**
     * @brief Destructor for SerialFFT.
     *
     * Cleans up resources allocated for the FFT plan.
     */
    ~SerialFFT();

    /**
     * @brief Performs a forward FFT.
     *
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     */
    void forward(T* in, T* out);

    /**
     * @brief Performs a backward FFT.
     *
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     */
    void backward(T* in, T* out);

    /**
     * @brief Performs a forward FFT in-place.
     *
     * @param in Pointer to the data to transform.
     */
    void forward(T* in);

    /**
     * @brief Performs a backward FFT in-place.
     *
     * @param in Pointer to the data to transform.
     */
    void backward(T* in);

    /**
     * @brief Gets the size of the FFT grid.
     *
     * @return The size of the FFT grid.
     */
    int ng() const;
};

#endif