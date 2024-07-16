#ifndef _FGPM_ALLOCATORS_HPP_
#define _FGPM_ALLOCATORS_HPP_

#include "gpu.hpp"
#include "logging.hpp"
#include <cstdio>
#include <iostream>
#include <unordered_map>

// extern std::unordered_map<void*, size_t> gpu_allocations;
extern std::unordered_map<void*, size_t> cpu_allocations;

static inline void
show_alive_allocations(const std::unordered_map<void*, size_t>& allocations,
                       const char* name) {
#if LOGLEVEL > 3
    LOG_DEBUG(" ");
    LOG_DEBUG("##############################");
    if (allocations.size() == 0) {
        LOG_DEBUG("No Alive %s Allocations", name);
    } else {
        LOG_DEBUG("Alive %s Allocations:", name);
        for (const auto& [key, value] : allocations) {
            LOG_DEBUG("   - %p : %lu bytes", key, value);
        }
    }
    LOG_DEBUG("##############################");
    LOG_DEBUG(" ");
#endif
}

class GPUAllocator {
  private:
    std::unordered_map<void*, size_t> gpu_allocations;
    size_t total_size;
    size_t current_size;

  public:
    GPUAllocator() : total_size(0), current_size(0) {
        LOG_DEBUG("Initializing GPU Allocator");
    }

    ~GPUAllocator() {
        if (gpu_allocations.size() != 0) {
            LOG_ERROR("%lu unfreed GPU pointers!", gpu_allocations.size());
            show_alive_allocations(gpu_allocations, "GPU");
        } else {
            LOG_DEBUG("%lu unfreed GPU pointers.", gpu_allocations.size());
        }

        LOG_INFO("GPU Allocations (total = %lu bytes, current = %lu bytes)",
                 total_size, current_size);
    }

    template <class T> inline int alloc(T** ptr, size_t sz) {
        gpuCall(gpuMalloc(ptr, sz));
        LOG_DEBUG("allocated %lu bytes on GPU for %p", sz, (void*)*ptr);
        gpu_allocations[*ptr] = sz;

        total_size += sz;
        current_size += sz;

        show_alive_allocations(gpu_allocations, "GPU");

        LOG_DEBUG("GPU Allocations (total = %lu bytes, current = %lu bytes)",
                  total_size, current_size);

        return 0;
    }

    template <class T> inline int free(T* ptr) {
        size_t sz = gpu_allocations[ptr];

        LOG_DEBUG("freeing GPU pointer %p (previous size %lu bytes)",
                  (void*)ptr, sz);
        gpu_allocations.erase(ptr);
        gpuCall(gpuFree(ptr));

        current_size -= sz;

        show_alive_allocations(gpu_allocations, "GPU");

        LOG_DEBUG("GPU Allocations (total = %lu bytes, current = %lu bytes)",
                  total_size, current_size);
        return 0;
    }
};

extern GPUAllocator gpu_allocator;

class CPUAllocator {
  private:
    std::unordered_map<void*, size_t> cpu_allocations;
    size_t total_size;
    size_t current_size;

  public:
    CPUAllocator() : total_size(0), current_size(0) {
        LOG_DEBUG("Initializing CPU Allocator");
    }

    ~CPUAllocator() {
        if (cpu_allocations.size() != 0) {
            LOG_ERROR("%lu unfreed CPU pointers!", cpu_allocations.size());
            show_alive_allocations(cpu_allocations, "CPU");
        } else {
            LOG_DEBUG("%lu unfreed CPU pointers.", cpu_allocations.size());
        }

        LOG_INFO("CPU Allocations (total = %lu bytes, current = %lu bytes)",
                 total_size, current_size);
    }

    template <class T> inline int alloc(T** ptr, size_t sz) {
        assert(*ptr = malloc(sz));
        LOG_DEBUG("allocated %lu bytes on CPU for %p", sz, (void*)*ptr);

        total_size += sz;
        current_size += sz;

        cpu_allocations[*ptr] = sz;

        show_alive_allocations(cpu_allocations, "CPU");

        LOG_DEBUG("CPU Allocations (total = %lu bytes, current = %lu bytes)",
                  total_size, current_size);

        return 0;
    }

    template <class T> inline int free(T* ptr) {
        size_t sz = cpu_allocations[ptr];
        LOG_DEBUG("freeing CPU pointer %p (previous size %lu bytes)",
                  (void*)ptr, sz);
        current_size -= sz;

        cpu_allocations.erase(ptr);
        free(ptr);

        show_alive_allocations(cpu_allocations, "CPU");

        LOG_DEBUG("CPU Allocations (total = %lu bytes, current = %lu bytes)",
                  total_size, current_size);
        return 0;
    }
};

extern CPUAllocator cpu_allocator;

#endif