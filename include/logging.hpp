#ifndef _LOGGING_H_
#define _LOGGING_H_

#include <stdio.h>
#include <string.h>

#define __FILENAME__                                                           \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define LOG_TEXT_RESET "\033[0m"
#define LOG_TEXT_BOLD "\033[1m"
#define LOG_TEXT_BLACK "\033[30m"              /* Black */
#define LOG_TEXT_RED "\033[31m"                /* Red */
#define LOG_TEXT_GREEN "\033[32m"              /* Green */
#define LOG_TEXT_YELLOW "\033[33m"             /* Yellow */
#define LOG_TEXT_BLUE "\033[34m"               /* Blue */
#define LOG_TEXT_MAGENTA "\033[35m"            /* Magenta */
#define LOG_TEXT_CYAN "\033[36m"               /* Cyan */
#define LOG_TEXT_WHITE "\033[37m"              /* White */
#define LOG_TEXT_BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define LOG_TEXT_BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define LOG_TEXT_BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define LOG_TEXT_BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define LOG_TEXT_BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define LOG_TEXT_BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define LOG_TEXT_BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define LOG_TEXT_BOLDWHITE "\033[1m\033[37m"   /* Bold White */

#ifndef LOGLEVEL
#define LOGLEVEL 3
#endif

#define _LOG(msg, msg_color, text_color, ...)                                  \
    {                                                                          \
        printf("log<%s%s%s>(%s%s:%d%s): %s", msg_color, msg, LOG_TEXT_RESET,   \
               LOG_TEXT_BOLDCYAN, __FILENAME__, __LINE__, LOG_TEXT_RESET,      \
               text_color);                                                    \
        printf(__VA_ARGS__);                                                   \
        printf("%s\n", LOG_TEXT_RESET);                                        \
    }

#define LOG_MINIMAL(...)                                                       \
    _LOG("minimal", LOG_TEXT_BOLD, LOG_TEXT_RESET, __VA_ARGS__)

#if LOGLEVEL > 0
#define LOG_ERROR(...)                                                         \
    _LOG("error", LOG_TEXT_BOLDRED, LOG_TEXT_RED, __VA_ARGS__)
#else
#define LOG_ERROR(...)
#endif

#if LOGLEVEL > 1
#define LOG_WARNING(...)                                                       \
    _LOG("warning", LOG_TEXT_BOLDYELLOW, LOG_TEXT_YELLOW, __VA_ARGS__)
#else
#define LOG_WARNING(...)
#endif

#if LOGLEVEL > 2
#define LOG_INFO(...)                                                          \
    _LOG("info", LOG_TEXT_BOLDGREEN, LOG_TEXT_GREEN, __VA_ARGS__)
#else
#define LOG_INFO(...)
#endif

#if LOGLEVEL > 3
#define LOG_DEBUG(...)                                                         \
    _LOG("debug", LOG_TEXT_BOLDBLUE, LOG_TEXT_BLUE, __VA_ARGS__)
#else
#define LOG_DEBUG(...)
#endif

#endif