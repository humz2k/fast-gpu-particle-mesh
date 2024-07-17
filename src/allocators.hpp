#ifndef _FGPM_ALLOCATORS_HPP_
#define _FGPM_ALLOCATORS_HPP_

#include "gpu.hpp"
#include "logging.hpp"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <unordered_map>

/**
 * @brief Displays the current alive allocations.
 *
 * This function logs the current alive allocations for a given set of
 * allocations and a name to identify the type of allocations.
 *
 * @param allocations A map of pointers to their allocated sizes.
 * @param name The name of the allocation type (e.g., "GPU", "CPU").
 */
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

/**
 * @class GPUAllocator
 * @brief Manages GPU memory allocations.
 *
 * The GPUAllocator class handles the allocation and deallocation of memory on
 * the GPU. It keeps track of all active allocations and provides logging
 * information about memory usage.
 */
class GPUAllocator {
  private:
    std::unordered_map<void*, size_t> gpu_allocations; /**< Map ptr->sz. */
    size_t total_size;   /**< Total size of all allocations. */
    size_t current_size; /**< Current size of active allocations. */

  public:
    /**
     * @brief Constructs a GPUAllocator object.
     *
     * Initializes the allocator and logs the initialization.
     */
    GPUAllocator() : total_size(0), current_size(0) {
        LOG_DEBUG("Initializing GPU Allocator");
    }

    /**
     * @brief Destroys the GPUAllocator object.
     *
     * Checks for any unfreed GPU allocations and logs the total and current
     * allocation sizes.
     */
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

    /**
     * @brief Allocates memory on the GPU.
     *
     * Allocates the specified size of memory on the GPU for the given pointer.
     *
     * @tparam T The type of the pointer.
     * @param ptr The pointer to allocate memory for.
     * @param sz The size of the memory to allocate.
     * @return 0 on success.
     */
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

    /**
     * @brief Frees memory on the GPU.
     *
     * Frees the memory allocated for the given pointer on the GPU.
     *
     * @tparam T The type of the pointer.
     * @param ptr The pointer to free memory for.
     * @return 0 on success.
     */
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

extern GPUAllocator gpu_allocator; /**< Global GPUAllocator instance. */

/**
 * @class CPUAllocator
 * @brief Manages CPU memory allocations.
 *
 * The CPUAllocator class handles the allocation and deallocation of memory on
 * the CPU. It keeps track of all active allocations and provides logging
 * information about memory usage.
 */
class CPUAllocator {
  private:
    std::unordered_map<void*, size_t> cpu_allocations; /**< Map ptr->sz. */
    size_t total_size;   /**< Total size of all allocations. */
    size_t current_size; /**< Current size of active allocations. */

  public:
    /**
     * @brief Constructs a CPUAllocator object.
     *
     * Initializes the allocator and logs the initialization.
     */
    CPUAllocator() : total_size(0), current_size(0) {
        LOG_DEBUG("Initializing CPU Allocator");
    }

    /**
     * @brief Destroys the CPUAllocator object.
     *
     * Checks for any unfreed CPU allocations and logs the total and current
     * allocation sizes.
     */
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

    /**
     * @brief Allocates memory on the CPU.
     *
     * Allocates the specified size of memory on the CPU for the given pointer.
     *
     * @tparam T The type of the pointer.
     * @param ptr The pointer to allocate memory for.
     * @param sz The size of the memory to allocate.
     * @return 0 on success.
     */
    template <class T> inline int alloc(T** ptr, size_t sz) {
        assert(*ptr = (T*)malloc(sz));
        LOG_DEBUG("allocated %lu bytes on CPU for %p", sz, (void*)*ptr);

        total_size += sz;
        current_size += sz;

        cpu_allocations[*ptr] = sz;

        show_alive_allocations(cpu_allocations, "CPU");

        LOG_DEBUG("CPU Allocations (total = %lu bytes, current = %lu bytes)",
                  total_size, current_size);

        return 0;
    }

    /**
     * @brief Frees memory on the CPU.
     *
     * Frees the memory allocated for the given pointer on the CPU.
     *
     * @tparam T The type of the pointer.
     * @param ptr The pointer to free memory for.
     * @return 0 on success.
     */
    template <class T> inline int free(T* ptr) {
        size_t sz = cpu_allocations[ptr];
        LOG_DEBUG("freeing CPU pointer %p (previous size %lu bytes)",
                  (void*)ptr, sz);
        current_size -= sz;

        cpu_allocations.erase(ptr);
        ::free(ptr);

        show_alive_allocations(cpu_allocations, "CPU");

        LOG_DEBUG("CPU Allocations (total = %lu bytes, current = %lu bytes)",
                  total_size, current_size);
        return 0;
    }
};

extern CPUAllocator cpu_allocator; /**< Global CPUAllocator instance. */

#endif