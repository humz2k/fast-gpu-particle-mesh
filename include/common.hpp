#ifndef _FGPM_COMMON_HPP_
#define _FGPM_COMMON_HPP_

#include <math.h>
#include <sys/time.h>
#include <time.h>

#define USECPSEC 1000000ULL

typedef unsigned long long CPUTimer_t;

static inline unsigned long long CPUTimer(unsigned long long start = 0) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

template <class T> static inline T z2a(T z) {
    /*! NOTE: I guess this adds some cast operations when in double precision?*/
    return 1.0f / (1.0f + z);
}

template <class T> static inline T a2z(T a) {
    /*! NOTE: I guess this adds some cast operations when in double precision?*/
    return (1.0f / (a)) - 1.0f;
}

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define BLOCKSIZE 64

#endif