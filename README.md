# fast-gpu-particle-mesh

![build](https://github.com/humz2k/fast-gpu-particle-mesh/actions/workflows/build.yml/badge.svg)

## Building
1. Clone repository:
```
git clone https://github.com/humz2k/fast-gpu-particle-mesh.git
cd fast-gpu-particle-mesh
```
2. If needed, set environment variables `$CUDA_PATH` and `$MPI_CXX` (defaults to `$CUDA_PATH=/usr/local/cuda` and `$MPI_CXX=mpicxx`).
3. Run `make`.
4. The output binary is then `build/driver`.

