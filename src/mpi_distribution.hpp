#ifndef _FGPM_MPI_DISTRIBUTIONS_HPP_
#define _FGPM_MPI_DISTRIBUTIONS_HPP_

#include "gpu.hpp"

/**
 * @class MPIDist
 * @brief Manages MPI distributions and grid calculations.
 *
 * The MPIDist class handles the distribution of a computational grid across
 * multiple MPI ranks. It provides methods to calculate local and global
 * coordinates and indices based on the MPI rank and grid configuration.
 */
class MPIDist {
  private:
    int m_rank;         /**< MPI rank of the current process. */
    int m_ng;           /**< Global grid size. */
    int3 m_rank_dims;   /**< Dimensions of the MPI rank grid. */
    int3 m_rank_coords; /**< Coords of the current rank in the MPI rank grid. */
    int3 m_local_grid_dims; /**< Size of the local grid for the current rank. */

  public:
    /**
     * @brief Constructs an MPIDist object with the given parameters.
     *
     * @param ng Global grid size.
     * @param rank MPI rank of the current process.
     * @param rank_dims Dimensions of the MPI rank grid.
     * @param rank_coords Coordinates of the current rank in the MPI rank grid.
     * @param local_grid_size Size of the local grid for the current rank.
     */
    MPIDist(int ng, int rank, int3 rank_dims, int3 rank_coords,
            int3 local_grid_dims)
        : m_rank(rank), m_ng(ng), m_rank_dims(rank_dims),
          m_rank_coords(rank_coords), m_local_grid_dims(local_grid_dims) {}

    /**
     * @brief Constructs an MPIDist object with the given global grid size.
     *
     * This constructor initializes the rank to 0 and assumes a single rank
     * grid.
     *
     * @param ng Global grid size.
     */
    MPIDist(int ng)
        : m_rank(0), m_ng(ng), m_rank_dims{1, 1, 1}, m_rank_coords{0, 0, 0},
          m_local_grid_dims{ng, ng, ng} {}

    /**
     * @brief Calculates local coordinates from a local index.
     *
     * This method converts a linear local index to 3D local coordinates.
     *
     * @param local_idx The local index.
     * @return The 3D local coordinates.
     */
    __forceinline__ __host__ __device__ int3 local_coords(int local_idx) const {
        return make_int3(
            (local_idx / m_local_grid_dims.z) / m_local_grid_dims.y,
            (local_idx / m_local_grid_dims.z) % m_local_grid_dims.y,
            local_idx % m_local_grid_dims.z);
    }

    /**
     * @brief Calculates global coordinates from a local index.
     *
     * This method converts a linear local index to 3D global coordinates.
     *
     * @param local_idx The local index.
     * @return The 3D global coordinates.
     */
    __forceinline__ __host__ __device__ int3
    global_coords(int local_idx) const {
        return local_coords(local_idx) + (m_rank_coords * m_local_grid_dims);
    }

    /**
     * @brief Checks if the local index is the global origin.
     *
     * This method converts a linear local index to 3D global coordinates and
     * then checks if it is the global origin.
     *
     * @param local_idx The local index.
     * @return If the index is the global origin.
     */
    __forceinline__ __host__ __device__ bool
    is_global_origin(int local_idx) const {
        int3 global = global_coords(local_idx);
        return (global.x == 0) && (global.y == 0) && (global.z == 0);
    }

    /**
     * @brief Calculates the global index from a local index.
     *
     * This method converts a linear local index to a linear global index.
     *
     * @param local_idx The local index.
     * @return The linear global index.
     */
    __forceinline__ __host__ __device__ int global_idx(int local_idx) const {
        int3 global = global_coords(local_idx);
        return global.x * m_ng * m_ng + global.y * m_ng + global.z;
    }

    /**
     * @brief Gets the MPI rank of the current process.
     *
     * @return The MPI rank of the current process.
     */
    __forceinline__ __host__ __device__ int rank() const { return m_rank; }

    /**
     * @brief Gets the global grid size.
     *
     * @return The global grid size.
     */
    __forceinline__ __host__ __device__ int ng() const { return m_ng; }

    /**
     * @brief Gets the dimensions of the MPI rank grid.
     *
     * @return The dimensions of the MPI rank grid.
     */
    __forceinline__ __host__ __device__ int3 rank_dims() const {
        return m_rank_dims;
    }

    /**
     * @brief Gets the coordinates of the current rank in the MPI rank grid.
     *
     * @return The coordinates of the current rank in the MPI rank grid.
     */
    __forceinline__ __host__ __device__ int3 rank_coords() const {
        return m_rank_coords;
    }

    /**
     * @brief Gets the dimensions of the local grid.
     *
     * @return The dimensions of the local grid.
     */
    __forceinline__ __host__ __device__ int3 local_grid_dims() const {
        return m_local_grid_dims;
    }

    /**
     * @brief Gets the size of the local grid.
     *
     * @return The size of the local grid.
     */
    __forceinline__ __host__ __device__ int local_grid_size() const {
        return m_local_grid_dims.x * m_local_grid_dims.y * m_local_grid_dims.z;
    }

    /**
     * @brief Gets the size of the global grid.
     *
     * @return The size of the global grid.
     */
    __forceinline__ __host__ __device__ int global_grid_size() const {
        return m_ng * m_ng * m_ng;
    }

    /**
     * @brief Calculates the wave numbers (k-modes) for a given local index and
     * spacing.
     *
     * This method converts a linear local index to 3D global coordinates and
     * adjusts them based on the specified spacing to determine the wave numbers
     * (k-modes). It handles the periodic boundary conditions by wrapping around
     * the indices.
     *
     * @param local_idx The local index.
     * @param d The spacing between grid points.
     * @return A float3 containing the k-modes.
     */
    __forceinline__ __host__ __device__ float3 kmodes(int local_idx,
                                                      float d) const {
        int3 idx3d = global_coords(local_idx);
        // periodic
        float l = (idx3d.x > ((m_ng / 2) - 1)) ? -(m_ng - idx3d.x) : idx3d.x;
        float m = (idx3d.y > ((m_ng / 2) - 1)) ? -(m_ng - idx3d.y) : idx3d.y;
        float n = (idx3d.z > ((m_ng / 2) - 1)) ? -(m_ng - idx3d.z) : idx3d.z;
        return make_float3(l, m, n) * d; // scale by grid spacing
    }
};

#endif