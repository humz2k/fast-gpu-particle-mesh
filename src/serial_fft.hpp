#ifndef _FGPM_SERIAL_FFT_HPP_
#define _FGPM_SERIAL_FFT_HPP_

#include "gpu.hpp"
#include "simulation.hpp"

template <class T> class SerialFFT : public FFT<T> {
  private:
    int m_ng;
    gpufftHandle m_plan;
    void fft(T* in, T* out, int direction);

  public:
    SerialFFT(int ng);
    ~SerialFFT();
    void forward(T* in, T* out);
    void backward(T* in, T* out);
    void forward(T* in);
    void backward(T* in);
};

#endif