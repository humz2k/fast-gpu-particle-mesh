#ifndef _FGPM_RNG_INITIALIZER_HPP_
#define _FGPM_RNG_INITIALIZER_HPP_

#include "gpu.hpp"

template<class T>
void launch_generate_real_random(T* d_grid, int seed, int offset, int size, int numBlocks, int blockSize);

#endif