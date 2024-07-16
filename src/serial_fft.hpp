#ifndef _FGPM_SERIAL_FFT_HPP_
#define _FGPM_SERIAL_FFT_HPP_

#include "simulation.hpp"
#include "gpu.hpp"
//#include <unordered_map>

template<class T>
class SerialFFT : public FFT<T>{
    private:
        int m_ng;
        //std::unordered_map<int,gpufftHandle> plans;

        //gpufftHandle get_plan(int ng);
        gpufftHandle m_plan;

        void fft(T* in, T* out, int direction);

    public:
        SerialFFT(int ng);
        ~SerialFFT();
        void forward(T* in, T* out);
        void backward(T* in, T* out);
};

#endif