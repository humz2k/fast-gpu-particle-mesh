#ifndef _FGPM_FP16_FFT_HPP_
#define _FGPM_FP16_FFT_HPP_

#include "gpu.hpp"
#include "simulation.hpp"
#include "allocators.hpp"
#include "cufft_wrapper.hpp"
#include "copy_grid.hpp"
#include <cassert>

class HalfFFT : public FFT<complexHalfDevice>{
    private:
        /** @brief ng of fft */
        int m_ng;
        /** @brief array of 3 ngs for XtMakePlanMany (all will be set to m_ng) */
        long long m_n[3];
        /** @brief rank = 3 for XtMakePlanMany */
        const int m_rank = 3;
        /** @brief fft plan */
        gpufftHandle m_plan;
        /** @brief work size of this plan */
        size_t m_work_size;

        const std::string fmt_type_name = std::string(typeid(complexHalfDevice).name());

        void fft(complexHalfDevice* in, complexHalfDevice* out, int direction){
            LOG_INFO("doing %s fft (ng = %d)",
                 (direction == GPUFFT_FORWARD) ? "forward" : "backward", m_ng);
            gpufftResult result;
            if ((result = gpufftXtExec(m_plan,in,out,direction)) != GPUFFT_SUCCESS){
                LOG_ERROR("Error in gpufftXtExec: %s", gpufftResult_to_string(result));
                exit(1);
            }
            gpuErrchk(gpuDeviceSynchronize());
            LOG_INFO("done %s fft (ng = %d)",
                 (direction == GPUFFT_FORWARD) ? "forward" : "backward", m_ng);
        }

    public:
        HalfFFT(int ng) : m_ng(ng), m_n{m_ng,m_ng,m_ng}{
            LOG_INFO("making half fft plan");
            gpufftResult result;
            if ((result = gpufftCreate(&m_plan)) != GPUFFT_SUCCESS){
                LOG_ERROR("Error in gpufftCreate: %s", gpufftResult_to_string(result));
                exit(1);
            }
            if ((result = gpufftXtMakePlanMany(m_plan,m_rank,m_n,NULL,1,1,CUDA_C_16F,NULL,1,1,CUDA_C_16F,1,&m_work_size,CUDA_C_16F)) != GPUFFT_SUCCESS){
                LOG_ERROR("Error in gpufftXtMakePlanMany: %s", gpufftResult_to_string(result));
                exit(1);
            }
        }

        ~HalfFFT(){
            gpufftResult result;
            if ((result = gpufftDestroy(m_plan)) != GPUFFT_SUCCESS){
                LOG_ERROR("Error in gpufftCreate: %s", gpufftResult_to_string(result));
                exit(1);
            }
        }

        void forward(complexHalfDevice* in, complexHalfDevice* out){
            events.timers["fft_forward_op_" + fmt_type_name].start();
            fft(in,out,GPUFFT_FORWARD);
            events.timers["fft_forward_op_" + fmt_type_name].end();
        }

        void backward(complexHalfDevice* in, complexHalfDevice* out){
            events.timers["fft_backward_op_" + fmt_type_name].start();
            fft(in,out,GPUFFT_INVERSE);
            events.timers["fft_backward_op_" + fmt_type_name].end();
        }

        void forward(complexHalfDevice* in){
            events.timers["fft_forward_ip_" + fmt_type_name].start();
            fft(in,in,GPUFFT_FORWARD);
            events.timers["fft_forward_ip_" + fmt_type_name].end();
        }

        void backward(complexHalfDevice* in){
            events.timers["fft_backward_ip_" + fmt_type_name].start();
            fft(in,in,GPUFFT_INVERSE);
            events.timers["fft_backward_ip_" + fmt_type_name].end();
        }
};

template<class T>
class CastHalfFFT : public FFT<T>{
    private:
        int m_ng;
        HalfFFT m_fft;
        int m_blockSize;
        int m_numBlocks;

        const std::string fmt_type_name = std::string(typeid(T).name()) + "_" + std::string(typeid(complexHalfDevice).name());

        void fft(T* in, T* out, int direction){
            complexHalfDevice* d_array;
            gpu_allocator.alloc(&d_array,sizeof(complexHalfDevice) * m_ng*m_ng*m_ng);
            launch_copy_grid(in,d_array,m_ng*m_ng*m_ng,m_numBlocks,m_blockSize);
            if (direction == GPUFFT_FORWARD)
                m_fft.forward(d_array);
            else
                m_fft.backward(d_array);
            launch_copy_grid(d_array,out,m_ng*m_ng*m_ng,m_numBlocks,m_blockSize);
            gpu_allocator.free(d_array);
        }
    public:
        CastHalfFFT(int ng, int blockSize = 32) : m_ng(ng), m_fft(m_ng), m_blockSize(blockSize), m_numBlocks((m_ng*m_ng*m_ng + (m_blockSize-1))/m_blockSize){}
        ~CastHalfFFT(){}

        void forward(T* in, T* out){
            events.timers["fft_forward_op_" + fmt_type_name].start();
            fft(in,out,GPUFFT_FORWARD);
            events.timers["fft_forward_op_" + fmt_type_name].end();
        }

        void backward(T* in, T* out){
            events.timers["fft_backward_op_" + fmt_type_name].start();
            fft(in,out,GPUFFT_INVERSE);
            events.timers["fft_backward_op_" + fmt_type_name].end();
        }

        void forward(T* in){
            events.timers["fft_forward_ip_" + fmt_type_name].start();
            fft(in,in,GPUFFT_FORWARD);
            events.timers["fft_forward_ip_" + fmt_type_name].end();
        }

        void backward(T* in){
            events.timers["fft_backward_ip_" + fmt_type_name].start();
            fft(in,in,GPUFFT_INVERSE);
            events.timers["fft_backward_ip_" + fmt_type_name].end();
        }
};

#endif // _FGPM_FP16_FFT_HPP_