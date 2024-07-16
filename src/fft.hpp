#ifndef _FGPM_FFT_HPP_
#define _FGPM_FFT_HPP_
#include "gpu.hpp"

template<class T>
class FFT{
    public:
        FFT();
        virtual ~FFT();
        virtual void forward(T* in, T* out, int ng) = 0;
        virtual void backward(T* in, T* out, int ng) = 0;
};

template<class T>
class SerialFFT : public FFT<T>{
    public:
        SerialFFT();
        ~SerialFFT();
};

#endif