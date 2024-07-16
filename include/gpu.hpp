#ifndef _FGPM_GPU_HPP_
#define _FGPM_GPU_HPP_

#include <cuda_runtime.h>
#include <cufft.h>

#include "logging.hpp"

typedef cufftDoubleComplex complexDoubleDevice;
typedef cufftComplex complexFloatDevice;

#define gpufftHandle cufftHandle

#define gpufftPlan1d cufftPlan1d

#define gpufftPlan3d cufftPlan3d

#define gpufftDestroy cufftDestroy

#define gpufftType cufftType

#define GPUFFT_Z2Z CUFFT_Z2Z
#define GPUFFT_C2C CUFFT_C2C
#define GPUFFT_SUCCESS CUFFT_SUCCESS
#define GPUFFT_FORWARD CUFFT_FORWARD
#define GPUFFT_INVERSE CUFFT_INVERSE

#define gpufftExecZ2Z cufftExecZ2Z
#define gpufftExecC2C cufftExecC2C

#define gpuEvent_t cudaEvent_t

#define gpuStream_t cudaStream_t

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
#define gpuError_t cudaError_t
#define gpuPeekAtLastError cudaPeekAtLastError

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(gpuError_t code, const char *file, int line, bool abort=true)
{
   if (code != gpuSuccess)
   {
      LOG_ERROR("GPUassert: %s %s %d\n", gpuGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuCall(func) {gpuErrchk(func);gpuErrchk(gpuDeviceSynchronize());}

#define gpuLaunch(kernel,numBlocks,blockSize,...) {LOG_INFO("launching %s<%d,%d>%s",TOSTRING(kernel),numBlocks,blockSize,TOSTRING((__VA_ARGS__)));kernel<<<numBlocks,blockSize>>>(__VA_ARGS__);gpuErrchk(gpuPeekAtLastError());}

#define gpuMemcpyAsync cudaMemcpyAsync

#define gpuEventCreate cudaEventCreate

#endif
