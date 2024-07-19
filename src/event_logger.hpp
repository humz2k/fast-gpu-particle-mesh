#ifndef _FGPM_EVENT_LOGGER_
#define _FGPM_EVENT_LOGGER_

#include "common.hpp"
#include "logging.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

//#define LOG_FUNC_CALL() events.function_calls.log(__func__)

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
    //EventLogger<std::string> function_calls;
    Events() : m_start(CPUTimer()), gpu_allocation(m_start) {}
    ~Events() { gpu_allocation.to_csv("gpu_allocations.csv"); }
};

extern Events events;

#endif