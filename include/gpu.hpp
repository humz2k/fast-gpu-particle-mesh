#ifndef _FGPM_GPU_HPP_
#define _FGPM_GPU_HPP_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufftXt.h>

#include "common.hpp"
#include "logging.hpp"

#include "operators.hpp"

typedef cufftDoubleComplex complexDoubleDevice;
typedef cufftComplex complexFloatDevice;
typedef half2 complexHalfDevice;

#define make_complexDoubleDevice make_cuDoubleComplex
#define make_complexFloatDevice make_cuFloatComplex

#define gpufftCreate cufftCreate

typedef cufftHandle gpufftHandle;

typedef cufftResult gpufftResult;

#define gpufftPlan1d cufftPlan1d

#define gpufftPlan3d cufftPlan3d

#define gpufftXtMakePlanMany cufftXtMakePlanMany

#define gpufftXtExec cufftXtExec

#define gpufftDestroy cufftDestroy

typedef cufftType gpufftType;

#define GPUFFT_Z2Z CUFFT_Z2Z
#define GPUFFT_C2C CUFFT_C2C
#define GPUFFT_SUCCESS CUFFT_SUCCESS
#define GPUFFT_FORWARD CUFFT_FORWARD
#define GPUFFT_INVERSE CUFFT_INVERSE

#define GPUFFT_INVALID_PLAN CUFFT_INVALID_PLAN
#define GPUFFT_ALLOC_FAILED CUFFT_ALLOC_FAILED
#define GPUFFT_INVALID_VALUE CUFFT_INVALID_VALUE
#define GPUFFT_INTERNAL_ERROR CUFFT_INTERNAL_ERROR
#define GPUFFT_SETUP_FAILED CUFFT_SETUP_FAILED
#define GPUFFT_INVALID_SIZE CUFFT_INVALID_SIZE
#define GPUFFT_SUCCESS CUFFT_SUCCESS

#define gpufftExecZ2Z cufftExecZ2Z
#define gpufftExecC2C cufftExecC2C

typedef cudaEvent_t gpuEvent_t;

typedef cudaStream_t gpuStream_t;

#define gpuEventDestroy cudaEventDestroy

#define gpuStreamDestroy cudaStreamDestroy

#define gpuEventRecord cudaEventRecord

#define gpuEventSynchronize cudaEventSynchronize

#define gpuMalloc cudaMalloc

#define gpuMemset cudaMemset

#define gpuMemcpy cudaMemcpy

#define gpuDeviceSynchronize cudaDeviceSynchronize

#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice

#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost

#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define gpuFree cudaFree

#define gpuStreamCreate cudaStreamCreate

#define gpuStreamSynchronize cudaStreamSynchronize

#define gpuSuccess cudaSuccess
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
typedef cudaError_t gpuError_t;
#define gpuPeekAtLastError cudaPeekAtLastError

#define gpuErrchk(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(gpuError_t code, const char* file, int line,
                      bool abort = true) {
    if (code != gpuSuccess) {
        LOG_ERROR("GPUassert: %s %s %d\n", gpuGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define gpuCall(func)                                                          \
    {                                                                          \
        gpuErrchk(func);                                                       \
        gpuErrchk(gpuDeviceSynchronize());                                     \
    }

#define gpuLaunch(kernel, numBlocks, blockSize, ...)                           \
    {                                                                          \
        LOG_INFO("launching %s<%d,%d>%s", TOSTRING(kernel), numBlocks,         \
                 blockSize, TOSTRING((__VA_ARGS__)));                          \
        CPUTimer_t start = CPUTimer();                                         \
        kernel<<<numBlocks, blockSize>>>(__VA_ARGS__);                         \
        gpuErrchk(gpuPeekAtLastError());                                       \
        gpuErrchk(gpuDeviceSynchronize());                                     \
        CPUTimer_t end = CPUTimer();                                           \
        LOG_INFO("%s took %llu us", TOSTRING(kernel), end - start);            \
    }

#define gpuMemcpyAsync cudaMemcpyAsync

#define gpuEventCreate cudaEventCreate

#endif
