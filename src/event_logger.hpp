#ifndef _FGPM_EVENT_LOGGER_
#define _FGPM_EVENT_LOGGER_

#include "common.hpp"
#include "logging.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class Timer
 * @brief Class for measuring and recording execution times.
 *
 * The Timer class provides methods for starting and stopping a timer, and for
 * recording and analyzing execution times.
 */
class Timer {
  private:
    CPUTimer_t m_start;          ///< Start time of the timer
    CPUTimer_t m_end;            ///< End time of the timer
    std::vector<double> m_times; ///< Vector to store recorded times

  public:
    /**
     * @brief Constructs a Timer object.
     */
    Timer() : m_start(0), m_end(0) {}

    /**
     * @brief Starts the timer.
     */
    inline void start() { m_start = CPUTimer(); }

    /**
     * @brief Stops the timer and records the elapsed time.
     */
    inline void end() {
        m_end = CPUTimer();
        m_times.push_back((double)(m_end - m_start) * 1e-6);
    }

    /**
     * @brief Gets the recorded times.
     * @return A reference to the vector of recorded times.
     */
    inline const std::vector<double>& times() { return m_times; }

    /**
     * @brief Calculates the mean of the recorded times.
     * @return The mean of the recorded times.
     */
    inline double mean() {
        double total = 0;
        for (auto i : m_times) {
            total += i;
        }
        return total / m_times.size();
    }

    /**
     * @brief Gets the maximum recorded time.
     * @return The maximum recorded time.
     */
    inline double max() {
        double v = 0;
        for (auto i : m_times) {
            if (i > v) {
                v = i;
            }
        }
        return v;
    }

    /**
     * @brief Gets the minimum recorded time.
     * @return The minimum recorded time.
     */
    inline double min() {
        double v = m_times[0];
        for (auto i : m_times) {
            if (i < v) {
                v = i;
            }
        }
        return v;
    }

    /**
     * @brief Gets the frequency of recorded times.
     * @return The number of recorded times.
     */
    inline size_t freq() { return m_times.size(); }

    /**
     * @brief Calculates the total of the recorded times.
     * @return The total of the recorded times.
     */
    inline double total() {
        double total = 0;
        for (auto i : m_times) {
            total += i;
        }
        return total;
    }
};

/**
 * @class EventLogger
 * @brief Class for logging events with timestamps.
 *
 * The EventLogger class provides methods for logging events with their
 * timestamps and exporting the logged events to a CSV file.
 *
 * @tparam T The type of events to log.
 */
template <class T> class EventLogger {
  private:
    CPUTimer_t m_start;              ///< Start time of the logger
    std::vector<CPUTimer_t> m_times; ///< Vector to store timestamps
    std::vector<T> m_events;         ///< Vector to store events

  public:
    /**
     * @brief Constructs an EventLogger object with a specified start time.
     *
     * @param start The start time of the logger.
     */
    EventLogger(CPUTimer_t start) : m_start(start) {}

    /**
     * @brief Logs an event with the current timestamp.
     *
     * @param event The event to log.
     */
    inline void log(T event) {
        m_times.push_back(CPUTimer(m_start));
        m_events.push_back(event);
    }

    /**
     * @brief Exports the logged events and their timestamps to a CSV file.
     *
     * @param filename The name of the CSV file.
     */
    inline void to_csv(std::string filename) {
        std::ofstream output(filename);

        output << "time,event\n";

        for (size_t i = 0; i < m_times.size(); i++) {
            output << m_times[i] << "," << m_events[i] << "\n";
        }

        output.close();
    }
};

/**
 * @class Events
 * @brief Class for managing and logging various events and timers in the
 * simulation.
 *
 * The Events class provides an interface for managing multiple timers and an
 * EventLogger for GPU allocations. It also handles exporting logged data to
 * CSV files upon destruction.
 */
class Events {
  private:
    CPUTimer_t m_start; ///< Start time for the event logger

  public:
    EventLogger<size_t> gpu_allocation; ///< Event logger for GPU allocations
    std::unordered_map<std::string, Timer> timers; ///< Map of named timers

    /**
     * @brief Constructs an Events object and initializes the GPU allocation
     * logger.
     */
    Events() : m_start(CPUTimer()), gpu_allocation(m_start) {}

    /**
     * @brief Destructor for Events. Exports logged data to CSV files.
     */
    ~Events() {}

    inline void dump(const std::string& prefix) {
        gpu_allocation.to_csv(prefix + "gpu_allocations.csv");

        std::ofstream output(prefix + "timers.csv");

        output << "timer,mean,min,max,freq,total\n";

        for (auto i : timers) {
            output << i.first << "," << i.second.mean() << "," << i.second.min()
                   << "," << i.second.max() << "," << i.second.freq() << ","
                   << i.second.total() << "\n";
        }

        output.close();
    }

    template <typename T> void print_element(T t, const int& width) {
        std::cout << std::left << std::setw(width) << std::setfill(' ') << t;
    }

    inline void dump() {
        int timer_width = 40;
        int data_width = 9;
        print_element("timer", timer_width);
        print_element("mean", data_width);
        print_element("min", data_width);
        print_element("max", data_width);
        print_element("freq", data_width);
        print_element("total", data_width);
        std::cout << std::endl;
        for (int i = 0; i < timer_width + data_width * 5; i++) {
            std::cout << "-";
        }
        std::cout << std::endl;

        for (auto i : timers) {
            print_element(i.first, timer_width);
            print_element(i.second.mean(), data_width);
            print_element(i.second.min(), data_width);
            print_element(i.second.max(), data_width);
            print_element(i.second.freq(), data_width);
            print_element(i.second.total(), data_width);
            std::cout << std::endl;
            // std::cout << i.first << "," << i.second.mean() << "," <<
            // i.second.min()
            //        << "," << i.second.max() << "," << i.second.freq() << ","
            //        << i.second.total() << std::endl;
        }
    }
};

/**
 * @brief Global instance of Events for logging purposes.
 */
extern Events events;

#endif