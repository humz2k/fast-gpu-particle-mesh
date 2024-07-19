#include "serial_fft.hpp"
#include "event_logger.hpp"
#include "gpu.hpp"

static inline const char* cufftResult_to_string(gpufftResult error) {
    switch (error) {
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    default:
        return "CUFFT_UNKNOWN";
    }
}

template <class T> gpufftType find_type();

template <> gpufftType find_type<complexDoubleDevice>() { return CUFFT_Z2Z; }

template <> gpufftType find_type<complexFloatDevice>() { return CUFFT_C2C; }

template <class T> SerialFFT<T>::SerialFFT(int ng) : m_ng(ng) {
    LOG_INFO("making plan");
    gpufftResult result =
        gpufftPlan3d(&m_plan, m_ng, m_ng, m_ng, find_type<T>());
    if (result != CUFFT_SUCCESS) {
        LOG_ERROR("CUFFT error: %s", cufftResult_to_string(result));
        exit(1);
    }
}

template <class T> SerialFFT<T>::~SerialFFT() {
    LOG_INFO("destroying plan");
    gpufftResult result = gpufftDestroy(m_plan);
    if (result != CUFFT_SUCCESS) {
        LOG_ERROR("CUFFT error: %s", cufftResult_to_string(result));
        exit(1);
    }
}

template <class T>
gpufftResult exec_fft(gpufftHandle plan, T* in, T* out, int dir);

template <>
gpufftResult exec_fft(gpufftHandle plan, complexDoubleDevice* in,
                      complexDoubleDevice* out, int dir) {
    return gpufftExecZ2Z(plan, in, out, dir);
}

template <>
gpufftResult exec_fft(gpufftHandle plan, complexFloatDevice* in,
                      complexFloatDevice* out, int dir) {
    return gpufftExecC2C(plan, in, out, dir);
}

template <class T> void SerialFFT<T>::fft(T* in, T* out, int direction) {
    LOG_INFO("doing %s fft (ng = %d)",
             (direction == GPUFFT_FORWARD) ? "forward" : "backward", m_ng);
    gpufftResult result = exec_fft(m_plan, in, out, direction);
    if (result != CUFFT_SUCCESS) {
        LOG_ERROR("CUFFT error: %s", cufftResult_to_string(result));
        exit(1);
    }
    gpuErrchk(gpuDeviceSynchronize());
    LOG_INFO("done %s fft (ng = %d)",
             (direction == GPUFFT_FORWARD) ? "forward" : "backward", m_ng);
}

template <class T> void SerialFFT<T>::forward(T* in, T* out) {
    events.timers["fft_forward_op"].start();
    this->fft(in, out, GPUFFT_FORWARD);
    events.timers["fft_forward_op"].end();
}

template <class T> void SerialFFT<T>::backward(T* in, T* out) {
    events.timers["fft_backward_op"].start();
    this->fft(in, out, GPUFFT_INVERSE);
    events.timers["fft_backward_op"].end();
}

template <class T> void SerialFFT<T>::forward(T* in) {
    events.timers["fft_forward_ip"].start();
    this->fft(in, in, GPUFFT_FORWARD);
    events.timers["fft_forward_ip"].end();
}

template <class T> void SerialFFT<T>::backward(T* in) {
    events.timers["fft_backward_ip"].start();
    this->fft(in, in, GPUFFT_INVERSE);
    events.timers["fft_backward_ip"].end();
}

template <class T> int SerialFFT<T>::ng() const { return m_ng; }

template class SerialFFT<complexDoubleDevice>;
template class SerialFFT<complexFloatDevice>;
