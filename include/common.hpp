#ifndef _FGPM_COMMON_HPP_
#define _FGPM_COMMON_HPP_

#include <math.h>
#include <sys/time.h>
#include <time.h>

#define USECPSEC 1000000ULL ///< Constant to convert seconds to microseconds.

typedef unsigned long long CPUTimer_t; ///< Type definition for CPU timer.

/**
 * @brief Returns the current CPU time in microseconds.
 *
 * This function returns the current CPU time in microseconds. If a start time
 * is provided, it returns the elapsed time since the start time.
 *
 * @param start The start time in microseconds. Defaults to 0.
 * @return The current CPU time in microseconds, or the elapsed time since the start time.
 */
static inline unsigned long long CPUTimer(unsigned long long start = 0) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

/**
 * @brief Converts redshift to scale factor.
 *
 * This function converts a given redshift value to the corresponding scale
 * factor.
 *
 * @tparam T The type of the input value (e.g., float, double).
 * @param z The redshift value.
 * @return The corresponding scale factor.
 */
template <class T> static inline T z2a(T z) {
    /*! NOTE: I guess this adds some cast operations when in double precision?*/
    return 1.0f / (1.0f + z);
}

/**
 * @brief Converts scale factor to redshift.
 *
 * This function converts a given scale factor to the corresponding redshift
 * value.
 *
 * @tparam T The type of the input value (e.g., float, double).
 * @param a The scale factor.
 * @return The corresponding redshift value.
 */
template <class T> static inline T a2z(T a) {
    /*! NOTE: I guess this adds some cast operations when in double precision?*/
    return (1.0f / (a)) - 1.0f;
}

#define MAX(a, b) (a > b ? a : b) ///< Macro to get the maximum of two values.

#define MIN(a, b) (a < b ? a : b) ///< Macro to get the minimum of two values.

/**
 * @brief Sets a to same sign as b
 */
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define BLOCKSIZE 64 ///< Block size used in CUDA kernels.

#endif