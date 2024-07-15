#ifndef _FGPM_ALLOCATORS_HPP_
#define _FGPM_ALLOCATORS_HPP_

#include <unordered_map>
#include <cstdio>
#include "gpu.hpp"

int alloc_gpu(void** ptr, size_t sz);
int free_gpu(void* ptr);

#endif