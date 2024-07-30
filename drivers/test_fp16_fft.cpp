#include "allocators.hpp"
#include "event_logger.hpp"
#include "fp16_fft.hpp"
#include "logging.hpp"
#include "serial_fft.hpp"
#include <math.h>

template <class T> bool is_equal(T a_, T b_, double tolerance = 0.01) {
    double a = a_;
    double b = b_;
    if ((a == 0) || (b == 0)) {
        if (a == b)
            return true;
        return (fabs(a - b) < tolerance);
    }
    return fabs(1.0 - (a / b)) < tolerance;
}

template <class T>
bool test_delta_function(T* d_array, size_t ng, int direction,
                         double tolerance = 0.01) {
    T* h_array;
    cpu_allocator.alloc(&h_array, sizeof(T) * ng * ng * ng);
    gpuMemcpy(h_array, d_array, sizeof(T) * ng * ng * ng,
              gpuMemcpyDeviceToHost);

    double max_r = h_array[1].x;
    double min_r = h_array[1].x;
    double max_c = h_array[1].y;
    double min_c = h_array[1].y;
    for (size_t i = 1; i < ng * ng * ng; i++) {
        double r = h_array[i].x;
        double c = h_array[i].y;
        if (r > max_r) {
            max_r = r;
        }
        if (r < min_r) {
            min_r = r;
        }
        if (c > max_c) {
            max_c = c;
        }
        if (c < min_c) {
            min_c = c;
        }
    }
    double pole_r = h_array[0].x;
    double pole_c = h_array[0].y;
    bool passed = true;
    if (direction == GPUFFT_FORWARD) {
        if ((!is_equal(min_r, 1.0, tolerance)) ||
            (!is_equal(max_r, 1.0, tolerance)) ||
            (!is_equal(min_c, 0.0, tolerance)) ||
            (!is_equal(max_c, 0.0, tolerance)) ||
            (!is_equal(pole_c, 0.0, tolerance)) ||
            (!is_equal(pole_r, 1.0, tolerance))) {
            passed = false;
        }
    } else {
        if ((!is_equal(min_r, 0.0, tolerance)) ||
            (!is_equal(max_r, 0.0, tolerance)) ||
            (!is_equal(min_c, 0.0, tolerance)) ||
            (!is_equal(max_c, 0.0, tolerance)) ||
            (!is_equal(pole_c, 0.0, tolerance)) ||
            (!is_equal(pole_r, (double)(ng * ng * ng)))) {
            passed = false;
        }
    }
    LOG_MINIMAL("testing %s::%s:\n   - pole = %g + %gi\n   - real in [%g, "
                "%g]\n   - complex in [%gi, %gi]\n   - %s",
                typeid(T).name(),
                (direction == GPUFFT_FORWARD) ? "forward" : "backward", pole_r,
                pole_c, min_r, max_r, min_c, max_c,
                passed ? "\033[32mpass\033[0m" : "\033[31mfail\033[0m");
    cpu_allocator.free(h_array);
    return passed;
}

template <class fft_t, class T>
bool test_delta_function(size_t ng, int n_tests = 1) {
    // create fft object
    fft_t fft(ng);

    // allocate gpu memory
    T* d_array;
    gpu_allocator.alloc(&d_array, sizeof(T) * ng * ng * ng);

    bool passed = true;

    for (int test = 0; test < n_tests; test++) {
        // assign delta function
        T pole;
        pole.x = 1.0;
        pole.y = 0.0;
        gpuMemset(d_array, 0, sizeof(T) * ng * ng * ng);
        gpuMemcpy(d_array, &pole, sizeof(T), gpuMemcpyHostToDevice);

        // forward transform
        fft.forward(d_array);

        // check forward output
        passed &= test_delta_function(d_array, ng, GPUFFT_FORWARD);

        // backward transform
        fft.backward(d_array);

        // check backward output
        passed &= test_delta_function(d_array, ng, GPUFFT_INVERSE);
    }

    // free gpu memory
    gpu_allocator.free(d_array);
    return passed;
}

int main() {
    size_t ng = 32;

    test_delta_function<HalfFFT, complexHalfDevice>(ng, 5);

    test_delta_function<SerialFFT<complexFloatDevice>, complexFloatDevice>(ng,
                                                                           5);

    test_delta_function<SerialFFT<complexDoubleDevice>, complexDoubleDevice>(ng,
                                                                             5);

    test_delta_function<CastHalfFFT<complexFloatDevice>, complexFloatDevice>(ng,
                                                                             5);

    test_delta_function<CastHalfFFT<complexDoubleDevice>, complexDoubleDevice>(
        ng, 5);

    events.dump();
}