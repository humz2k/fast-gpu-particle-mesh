#ifndef _FGPM_EVENT_LOGGER_
#define _FGPM_EVENT_LOGGER_

#include "common.hpp"
#include "logging.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

class Timer{
  private:
    CPUTimer_t m_start;
    CPUTimer_t m_end;
    std::vector<double> m_times;
  public:
    Timer() : m_start(0), m_end(0){}

    inline void start(){
      m_start = CPUTimer();
    }

    inline void end(){
      m_end = CPUTimer();
      m_times.push_back((double)(m_end-m_start) * 1e-6);
    }

    inline const std::vector<double>& times(){return m_times;}

    inline double mean(){
      double total = 0;
      for (auto i : m_times){
        total += i;
      }
      return total / m_times.size();
    }

    inline double max(){
      double v = 0;
      for (auto i : m_times){
        if (i > v){
          v = i;
        }
      }
      return v;
    }

    inline double min(){
      double v = m_times[0];
      for (auto i : m_times){
        if (i < v){
          v = i;
        }
      }
      return v;
    }

    inline size_t freq(){
      return m_times.size();
    }

    inline double total(){
      double total = 0;
      for (auto i : m_times){
        total += i;
      }
      return total;
    }
};

template <class T> class EventLogger {
  private:
    CPUTimer_t m_start;
    std::vector<CPUTimer_t> m_times;
    std::vector<T> m_events;

  public:
    EventLogger(CPUTimer_t start) : m_start(start) {}
    inline void log(T event) {
        m_times.push_back(CPUTimer(m_start));
        m_events.push_back(event);
    }
    inline void to_csv(std::string filename) {
        std::ofstream output(filename);

        output << "time,event\n";

        for (size_t i = 0; i < m_times.size(); i++) {
            output << m_times[i] << "," << m_events[i] << "\n";
        }

        output.close();
    }
};

class Events {
  private:
    CPUTimer_t m_start;

  public:
    EventLogger<size_t> gpu_allocation;
    std::unordered_map<std::string,Timer> timers;
    Events() : m_start(CPUTimer()), gpu_allocation(m_start) {}
    ~Events() {
      gpu_allocation.to_csv("gpu_allocations.csv");

      std::ofstream output("timers.csv");

      output << "timer,mean,min,max,freq,total\n";

      for (auto i : timers){
        output << i.first << "," << i.second.mean() << "," << i.second.min() << "," << i.second.max() << "," << i.second.freq() << "," << i.second.total() << "\n";
      }

      output.close();
    }
};

extern Events events;

#endif