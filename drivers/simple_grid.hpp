#ifndef _FGPM_SIMPLE_GRID_HPP_
#define _FGPM_SIMPLE_GRID_HPP_

#include "allocators.hpp"
#include "common.hpp"
#include "initializer.hpp"
#include "serial_grid.hpp"
#include "solver.hpp"

template <template <class> class fft_wrapper_t, class fft_t>
class SimpleGrid : public SerialGrid<fft_wrapper_t, fft_t> {
  public:
    using SerialGrid<fft_wrapper_t, fft_t>::SerialGrid;

    void solve(){
        // not implemented
    };

    void solve_gradient() {

        /**
         * This is called after CICing to the grid. So, we need to solve for the
         * gradient (i.e., write to this->grad()) using the input from
         * this->grid().
         */

        // calculate blockSize/numBlocks for the CUDA kernels
        int blockSize = BLOCKSIZE;
        int numBlocks = (this->size() + (blockSize - 1)) / blockSize;

        // do an inplace FFT on this->grid()
        this->forward();

        // Allocate space to store the x/y/z components of the gradient.
        // This is not the most memory efficient way to do this...
        fft_t* d_x;
        gpu_allocator.alloc(&d_x, sizeof(fft_t) * this->size());
        fft_t* d_y;
        gpu_allocator.alloc(&d_y, sizeof(fft_t) * this->size());
        fft_t* d_z;
        gpu_allocator.alloc(&d_z, sizeof(fft_t) * this->size());

        // launch a CUDA kernel to solve for the gradient in kspace
        launch_kspace_solve_gradient(this->grid(), d_x, d_y, d_z, this->dist(),
                                     numBlocks, blockSize);

        // do inplace backward FFTs on the x/y/z components
        this->backward(d_x);
        this->backward(d_y);
        this->backward(d_z);

        // finally combine this into this->grad()
        launch_combine_vectors(this->grad(), d_x, d_y, d_z, this->dist(),
                               numBlocks, blockSize);

        // and free the extra memory we allocated
        gpu_allocator.free(d_x);
        gpu_allocator.free(d_y);
        gpu_allocator.free(d_z);
    };

    void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                  Particles<float3>& particles) {
        /**
         * This is called right after we allocate memory for the particles. We
         * need to initialize particle positions and velocities and store them
         * in particles.pos() and particles.vel() respectively.
         */

        // calculate blockSize/numBlocks for the CUDA kernels
        int blockSize = BLOCKSIZE;
        int numBlocks = (this->size() + (blockSize - 1)) / blockSize;

        // etc..

        this->generate_fourier_amplitudes(cosmo);
        float delta = cosmo.delta(ts.z());
        ts.reverse_half_step();
        float fscal = ts.fscal();
        float dot_delta = cosmo.dot_delta(ts.z());
        ts.advance_half_step();

        fft_t* d_x;
        gpu_allocator.alloc(&d_x, sizeof(fft_t) * this->size());
        fft_t* d_y;
        gpu_allocator.alloc(&d_y, sizeof(fft_t) * this->size());
        fft_t* d_z;
        gpu_allocator.alloc(&d_z, sizeof(fft_t) * this->size());

        launch_transform_density_field(this->grid(), d_x, d_y, d_z, delta,
                                       this->params().rl(), ts.a(),
                                       this->dist(), numBlocks, blockSize);

        this->backward(d_x);
        this->backward(d_y);
        this->backward(d_z);

        launch_combine_density_vectors(this->grad(), d_x, d_y, d_z,
                                       this->dist(), numBlocks, blockSize);

        gpu_allocator.free(d_x);
        gpu_allocator.free(d_y);
        gpu_allocator.free(d_z);

        launch_place_particles(particles.pos(), particles.vel(), this->grad(),
                               delta, dot_delta, this->params().rl(), ts.a(),
                               ts.deltaT(), fscal, this->params().ng(),
                               this->dist(), numBlocks, blockSize);
    };
};

#endif