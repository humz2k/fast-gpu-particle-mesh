#ifndef _FGPM_GPU_HPP_
#define _FGPM_GPU_HPP_

#include <cuda_runtime.h>
#include <cufft.h>

#include "common.hpp"
#include "logging.hpp"

typedef cufftDoubleComplex complexDoubleDevice;
typedef cufftComplex complexFloatDevice;

__forceinline__ __host__ __device__ int3 operator+(int3 l, int3 r) {
    return make_int3(l.x + r.x, l.y + r.y, l.z + r.z);
}

__forceinline__ __host__ __device__ int3 operator-(int3 l, int3 r) {
    return make_int3(l.x - r.x, l.y - r.y, l.z - r.z);
}

__forceinline__ __host__ __device__ int3 operator*(int3 l, int3 r) {
    return make_int3(l.x * r.x, l.y * r.y, l.z * r.z);
}

__forceinline__ __host__ __device__ int3 operator/(int3 l, int3 r) {
    return make_int3(l.x / r.x, l.y / r.y, l.z / r.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 l, float r) {
    return make_float3(l.x * r, l.y * r, l.z * r);
}

__forceinline__ __host__ __device__ float3 operator+(float3 l, float3 r) {
    return make_float3(l.x + r.x, l.y + r.y, l.z + r.z);
}

__forceinline__ __host__ __device__ float3 fmod(float3 l, float r){
    return make_float3(fmod(l.x,r),fmod(l.y,r),fmod(l.z,r));
}

__forceinline__ __host__ __device__ float len2(float3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__forceinline__ __host__ __device__ float len(float3 v) {
    return sqrtf(len2(v));
}

__forceinline__ __host__ __device__ double len2(complexDoubleDevice v) {
    return v.x * v.x + v.y * v.y;
}

__forceinline__ __host__ __device__ float len2(complexFloatDevice v) {
    return v.x * v.x + v.y * v.y;
}

#define make_complexDoubleDevice make_cuDoubleComplex
#define make_complexFloatDevice make_cuFloatComplex

__forceinline__ __host__ __device__ complexDoubleDevice
operator*(complexDoubleDevice l, double r) {
    return make_complexDoubleDevice(l.x * r, l.y * r);
}

__forceinline__ __host__ __device__ complexFloatDevice
operator*(complexFloatDevice l, double r) {
    return make_complexFloatDevice(l.x * r, l.y * r);
}

typedef cufftHandle gpufftHandle;

typedef cufftResult gpufftResult;

#define gpufftPlan1d cufftPlan1d

#define gpufftPlan3d cufftPlan3d

#define gpufftDestroy cufftDestroy

typedef cufftType gpufftType;

#define GPUFFT_Z2Z CUFFT_Z2Z
#define GPUFFT_C2C CUFFT_C2C
#define GPUFFT_SUCCESS CUFFT_SUCCESS
#define GPUFFT_FORWARD CUFFT_FORWARD
#define GPUFFT_INVERSE CUFFT_INVERSE

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
