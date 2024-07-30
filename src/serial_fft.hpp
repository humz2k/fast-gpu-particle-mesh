#ifndef _FGPM_SERIAL_FFT_HPP_
#define _FGPM_SERIAL_FFT_HPP_

#include "gpu.hpp"
#include "simulation.hpp"
#include "cufft_wrapper.hpp"

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
    void fft(T* in, T* out, int direction) {
        LOG_INFO("doing %s fft (ng = %d)",
                 (direction == GPUFFT_FORWARD) ? "forward" : "backward", m_ng);
        gpufftResult result = exec_fft(m_plan, in, out, direction);
        if (result != CUFFT_SUCCESS) {
            LOG_ERROR("CUFFT error: %s", gpufftResult_to_string(result));
            exit(1);
        }
        gpuErrchk(gpuDeviceSynchronize());
        LOG_INFO("done %s fft (ng = %d)",
                 (direction == GPUFFT_FORWARD) ? "forward" : "backward", m_ng);
    }

  public:
    /**
     * @brief Constructs a SerialFFT object with the specified grid size.
     *
     * @param ng The size of the grid.
     */
    SerialFFT(int ng) : m_ng(ng) {
        LOG_INFO("making plan");
        gpufftResult result =
            gpufftPlan3d(&m_plan, m_ng, m_ng, m_ng, find_type<T>());
        if (result != CUFFT_SUCCESS) {
            LOG_ERROR("CUFFT error: %s", gpufftResult_to_string(result));
            exit(1);
        }
    }

    /**
     * @brief Destructor for SerialFFT.
     *
     * Cleans up resources allocated for the FFT plan.
     */
    ~SerialFFT() {
        LOG_INFO("destroying plan");
        gpufftResult result = gpufftDestroy(m_plan);
        if (result != CUFFT_SUCCESS) {
            LOG_ERROR("CUFFT error: %s", gpufftResult_to_string(result));
            exit(1);
        }
    }

    /**
     * @brief Performs a forward FFT.
     *
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     */
    void forward(T* in, T* out) {
        events.timers["fft_forward_op"].start();
        this->fft(in, out, GPUFFT_FORWARD);
        events.timers["fft_forward_op"].end();
    }

    /**
     * @brief Performs a backward FFT.
     *
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     */
    void backward(T* in, T* out) {
        events.timers["fft_backward_op"].start();
        this->fft(in, out, GPUFFT_INVERSE);
        events.timers["fft_backward_op"].end();
    }

    /**
     * @brief Performs a forward FFT in-place.
     *
     * @param in Pointer to the data to transform.
     */
    void forward(T* in) {
        events.timers["fft_forward_ip"].start();
        this->fft(in, in, GPUFFT_FORWARD);
        events.timers["fft_forward_ip"].end();
    }

    /**
     * @brief Performs a backward FFT in-place.
     *
     * @param in Pointer to the data to transform.
     */
    void backward(T* in) {
        events.timers["fft_backward_ip"].start();
        this->fft(in, in, GPUFFT_INVERSE);
        events.timers["fft_backward_ip"].end();
    }

    /**
     * @brief Gets the size of the FFT grid.
     *
     * @return The size of the FFT grid.
     */
    int ng() const { return m_ng; }
};

#endif