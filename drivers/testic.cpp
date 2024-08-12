#include "simple_grid.hpp"
#include "simple_particles.hpp"
#include "simulation.hpp"

template <template <class> class fft_wrapper_t, class fft_t>
class SineGrid : public SimpleGrid<fft_wrapper_t, fft_t> {
    using SimpleGrid<fft_wrapper_t, fft_t>::SimpleGrid;

  public:
    void generate_fourier_amplitudes(Cosmo& cosmo) {
        LOG_INFO("generating fourier amplitudes");

        int blockSize = BLOCKSIZE;
        int numBlocks = (this->size() + (blockSize - 1)) / blockSize;

        launch_generate_real_random(this->grid(), this->params().seed(),
                                    this->dist(), numBlocks, blockSize);

        this->forward();

        launch_scale_amplitudes_by_power_spectrum(this->grid(), cosmo.initial_pk(),
                                                  this->params().rl(), this->dist(),
                                                  numBlocks, blockSize);

        fft_t* h_grid;
        cpu_allocator.alloc(&h_grid, this->size() * sizeof(fft_t));
        gpuMemcpy(h_grid, this->grid(), this->size() * sizeof(fft_t),
                  gpuMemcpyDeviceToHost);

        for (size_t i = 0; i < this->size(); i++) {
            if (i == 100){
                auto kmodes = this->dist().kmodes(i,2*M_PI/((float)this->params().ng()));
                printf("setting k %g %g %g (%g) to not 0\n",kmodes.x,kmodes.y,kmodes.z,len(kmodes));
                continue;
            }
            h_grid[i].x = 0;
            h_grid[i].y = 0;
        }

        gpuMemcpy(this->grid(), h_grid, this->size() * sizeof(fft_t),
                  gpuMemcpyHostToDevice);
        cpu_allocator.free(h_grid);
    }
};

class SineParticles : public SerialParticles {
  public:
    SineParticles(const Params& params, Cosmo& cosmo, Timestepper& ts)
        : SerialParticles(params, cosmo, ts) {
        SineGrid<SerialFFT, complexDoubleDevice> ic_grid(this->params(),
                                                         this->params().np());
        ic_grid.generate_displacement_ic(cosmo, ts, *this);
    }
};

int main() {
    MPI_Init(NULL, NULL);

    Params params("test.params");
    Timestepper ts(params);
    Cosmo cosmo(params);
    cosmo.initial_pk().to_csv(params.output_prefix() +
                              "input_power_spectrum.csv");

    SineParticles particles(params, cosmo, ts);

    events.timers["dinit"].end();

    SimpleGrid<SerialFFT, complexFloatDevice> grid(params, params.ng());
    PowerSpectrum(particles, grid, params.pk_n_bins(), 0).to_csv(params.output_prefix() + "fold0.csv");
    PowerSpectrum(particles, grid, params.pk_n_bins(), 1).to_csv(params.output_prefix() + "fold1.csv");

    MPI_Finalize();
    return 0;
}