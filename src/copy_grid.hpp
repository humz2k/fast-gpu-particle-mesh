#ifndef _FGPM_COPY_GRID_HPP_
#define _FGPM_COPY_GRID_HPP_

#include "gpu.hpp"

template <class T1, class T2>
void launch_copy_grid(const T1* in, T2* out, size_t n, int numBlocks,
                      int blockSize);

#endif // _FGPM_COPY_GRID_HPP_