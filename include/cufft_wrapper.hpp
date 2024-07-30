#ifndef _FGPM_CUFFT_WRAPPER_HPP_
#define _FGPM_CUFFT_WRAPPER_HPP_

#include "gpu.hpp"

static inline const char* gpufftResult_to_string(gpufftResult error) {
    switch (error) {
    case GPUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case GPUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case GPUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case GPUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case GPUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case GPUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case GPUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    default:
        return "CUFFT_UNKNOWN";
    }
}

template <class T> static inline gpufftType find_type();

template <> inline gpufftType find_type<complexDoubleDevice>() {
    return GPUFFT_Z2Z;
}

template <> inline gpufftType find_type<complexFloatDevice>() {
    return GPUFFT_C2C;
}

static inline gpufftResult exec_fft(gpufftHandle plan, complexDoubleDevice* in,
                                    complexDoubleDevice* out, int dir) {
    return gpufftExecZ2Z(plan, in, out, dir);
}

static inline gpufftResult exec_fft(gpufftHandle plan, complexFloatDevice* in,
                                    complexFloatDevice* out, int dir) {
    return gpufftExecC2C(plan, in, out, dir);
}

#endif // _FGPM_CUFFT_WRAPPER_HPP_