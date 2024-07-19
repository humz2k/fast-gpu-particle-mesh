#ifndef _FGPM_SIMPLE_GRID_HPP_
#define _FGPM_SIMPLE_GRID_HPP_

#include "allocators.hpp"
#include "common.hpp"
#include "initializer.hpp"
#include "serial_grid.hpp"
#include "solver.hpp"

template <class fft_t> class SimpleGrid : public SerialGrid<fft_t> {
  public:
    using SerialGrid<fft_t>::SerialGrid;

    void solve(){
        // not implemented
    };

    void solve_gradient() {

        int blockSize = BLOCKSIZE;
        int numBlocks = (this->size() + (blockSize - 1)) / blockSize;

        this->forward();

        fft_t* d_x;
        gpu_allocator.alloc(&d_x, sizeof(fft_t) * this->size());
        fft_t* d_y;
        gpu_allocator.alloc(&d_y, sizeof(fft_t) * this->size());
        fft_t* d_z;
        gpu_allocator.alloc(&d_z, sizeof(fft_t) * this->size());

        launch_kspace_solve_gradient(this->grid(), d_x, d_y, d_z, this->dist(),
                                     numBlocks, blockSize);

        this->backward(d_x);
        this->backward(d_y);
        this->backward(d_z);

        launch_combine_vectors(this->grad(), d_x, d_y, d_z, this->dist(),
                               numBlocks, blockSize);

        gpu_allocator.free(d_x);
        gpu_allocator.free(d_y);
        gpu_allocator.free(d_z);
    };

    void generate_displacement_ic(Cosmo& cosmo, Timestepper& ts,
                                  Particles<float3>& particles) {
        LOG_INFO("generating displacement ic");

        int blockSize = BLOCKSIZE;
        int numBlocks = (this->size() + (blockSize - 1)) / blockSize;

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