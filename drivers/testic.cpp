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

        fft_t* h_grid;
        cpu_allocator.alloc(&h_grid, this->size() * sizeof(fft_t));
        gpuMemcpy(h_grid, this->grid(), this->size() * sizeof(fft_t),
                  gpuMemcpyDeviceToHost);

        // float ng = this->dist().ng();

        for (size_t i = 0; i < this->size(); i++) {
            float sk = 0;
            // if (i == 10){
            //     sk = 1;
            // }

            float3 kmodes =
                this->dist().kmodes(i, (2.0f * M_PI) / this->params().rl());
            float k = len(kmodes);

            // if (k == 1){
            //     sk = 1;
            // }
            float center = 5;
            float delta = 0.1;

            if ((k <= (center + delta)) && (k >= (center - delta))) {
                sk = 1;
            }
            h_grid[i].x *= sk;
            h_grid[i].y *= sk;
        }

        gpuMemcpy(this->grid(), h_grid, this->size() * sizeof(fft_t),
                  gpuMemcpyHostToDevice);
        cpu_allocator.free(h_grid);

        // launch_scale_amplitudes_by_power_spectrum(this->grid(),
        // cosmo.initial_pk(),
        //                                         this->params().rl(),
        //                                         this->dist(), numBlocks,
        //                                         blockSize);
    }
};

class SineParticles : public SerialParticles {
  public:
    SineParticles(const Params& params, Cosmo& cosmo, Timestepper& ts)
        : SerialParticles(params, cosmo, ts) {
        /**
         * SerialParticles(...) will allocate the correct amount of space in
         * this->pos() and this->vel(); but will not initialize particles in the
         * arrays. So, we need to call generate_displacement_ic on some grid in
         * order to generate the initial conditions.
         */
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
    PowerSpectrum ic_power(particles, grid, params.pk_n_bins(), 1);
    ic_power.to_csv(params.output_prefix() + "init_power_spectrum.csv");

    MPI_Finalize();
    return 0;
}