#include "allocators.hpp"
#include "gpu.hpp"
#include "logging.hpp"

static std::unordered_map<void*, size_t> gpu_allocations;

int alloc_gpu(void** ptr, size_t sz){
    gpuCall(gpuMalloc(ptr,sz));
    LOG_INFO("allocated %lu bytes on GPU for %p",sz,*ptr);
    gpu_allocations[*ptr] = sz;
    return 0;
}

int free_gpu(void* ptr){
    LOG_INFO("freeing gpu pointer %p (previous size %lu)",ptr,gpu_allocations[ptr]);
    gpu_allocations[ptr] = 0;
    gpuCall(gpuFree(ptr));
    return 0;
}