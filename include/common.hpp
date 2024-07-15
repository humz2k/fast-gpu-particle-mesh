#ifndef _FGPM_COMMON_HPP_
#define _FGPM_COMMON_HPP_

template<class T>
static inline T z2a(T z){
    /*! NOTE: I guess this adds some cast operations when in double precision?*/
    return 1.0f/(1.0f+z);
}

template<class T>
static inline T a2z(T a){
    /*! NOTE: I guess this adds some cast operations when in double precision?*/
    return (1.0f/(a)) - 1.0f;
}

#endif