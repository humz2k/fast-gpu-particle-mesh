#ifndef _FGPM_OPERATORS_HPP_
#define _FGPM_OPERATORS_HPP_

#include <cuda_runtime.h>
#include <cufft.h>

/**
 * _________ _       _________ ______
 * \__   __/( (    /|\__   __// ___  \
 *    ) (   |  \  ( |   ) (   \/   \  \
 *    | |   |   \ | |   | |      ___) /
 *    | |   | (\ \) |   | |     (___ (
 *    | |   | | \   |   | |         ) \
 * ___) (___| )  \  |   | |   /\___/  /
 * \_______/|/    )_)   )_(   \______/
 */

__forceinline__ __host__ __device__ int3 operator+(int3 l, int3 r) {
    return make_int3(l.x + r.x, l.y + r.y, l.z + r.z);
}

__forceinline__ __host__ __device__ int3 operator+(int3 l, int r) {
    return make_int3(l.x + r, l.y + r, l.z + r);
}

__forceinline__ __host__ __device__ int3 operator+(int l, int3 r) {
    return make_int3(l + r.x, l + r.y, l + r.z);
}

__forceinline__ __host__ __device__ int3 operator-(int3 l, int3 r) {
    return make_int3(l.x - r.x, l.y - r.y, l.z - r.z);
}

__forceinline__ __host__ __device__ int3 operator-(int3 l, int r) {
    return make_int3(l.x - r, l.y - r, l.z - r);
}

__forceinline__ __host__ __device__ int3 operator-(int l, int3 r) {
    return make_int3(l - r.x, l - r.y, l - r.z);
}

__forceinline__ __host__ __device__ int3 operator*(int3 l, int3 r) {
    return make_int3(l.x * r.x, l.y * r.y, l.z * r.z);
}

__forceinline__ __host__ __device__ int3 operator*(int3 l, int r) {
    return make_int3(l.x * r, l.y * r, l.z * r);
}

__forceinline__ __host__ __device__ int3 operator*(int l, int3 r) {
    return make_int3(l * r.x, l * r.y, l * r.z);
}

__forceinline__ __host__ __device__ int3 operator/(int3 l, int3 r) {
    return make_int3(l.x / r.x, l.y / r.y, l.z / r.z);
}

__forceinline__ __host__ __device__ int3 operator/(int3 l, int r) {
    return make_int3(l.x / r, l.y / r, l.z / r);
}

__forceinline__ __host__ __device__ int3 operator/(int l, int3 r) {
    return make_int3(l / r.x, l / r.y, l / r.z);
}

__forceinline__ __host__ __device__ float3 to_float3(int3 v) {
    return make_float3(v.x, v.y, v.z);
}

/**
 *  _______  _        _______  _______ _________ ______
 * (  ____ \( \      (  ___  )(  ___  )\__   __// ___  \
 * | (    \/| (      | (   ) || (   ) |   ) (   \/   \  \
 * | (__    | |      | |   | || (___) |   | |      ___) /
 * |  __)   | |      | |   | ||  ___  |   | |     (___ (
 * | (      | |      | |   | || (   ) |   | |         ) \
 * | )      | (____/\| (___) || )   ( |   | |   /\___/  /
 * |/       (_______/(_______)|/     \|   )_(   \______/
 */

__forceinline__ __host__ __device__ float3 operator+(float3 l, float3 r) {
    return make_float3(l.x + r.x, l.y + r.y, l.z + r.z);
}

__forceinline__ __host__ __device__ float3 operator+(float3 l, float r) {
    return make_float3(l.x + r, l.y + r, l.z + r);
}

__forceinline__ __host__ __device__ float3 operator+(float l, float3 r) {
    return make_float3(l + r.x, l + r.z, l + r.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 l, float3 r) {
    return make_float3(l.x - r.x, l.y - r.y, l.z - r.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 l, float r) {
    return make_float3(l.x - r, l.y - r, l.z - r);
}

__forceinline__ __host__ __device__ float3 operator-(float l, float3 r) {
    return make_float3(l - r.x, l - r.z, l - r.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 l, float3 r) {
    return make_float3(l.x * r.x, l.y * r.y, l.z * r.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 l, float r) {
    return make_float3(l.x * r, l.y * r, l.z * r);
}

__forceinline__ __host__ __device__ float3 operator*(float l, float3 r) {
    return make_float3(l * r.x, l * r.y, l * r.z);
}

__forceinline__ __host__ __device__ float3 operator/(float3 l, float3 r) {
    return make_float3(l.x / r.x, l.y / r.y, l.z / r.z);
}

__forceinline__ __host__ __device__ float3 operator/(float3 l, float r) {
    return make_float3(l.x / r, l.y / r, l.z / r);
}

__forceinline__ __host__ __device__ float3 operator/(float l, float3 r) {
    return make_float3(l / r.x, l / r.y, l / r.z);
}

__forceinline__ __host__ __device__ float3 fmod(float3 l, float r) {
    return make_float3(fmod(l.x, r), fmod(l.y, r), fmod(l.z, r));
}

__forceinline__ __host__ __device__ float3 floor(float3 l) {
    return make_float3(floor(l.x), floor(l.y), floor(l.z));
}

__forceinline__ __host__ __device__ float3 cos(float3 v) {
    return make_float3(cos(v.x), cos(v.y), cos(v.z));
}

__forceinline__ __host__ __device__ float3 sin(float3 v) {
    return make_float3(sin(v.x), sin(v.y), sin(v.z));
}

__forceinline__ __host__ __device__ int3 to_int3(float3 v) {
    return make_int3(v.x, v.y, v.z);
}

__forceinline__ __host__ __device__ float len2(float3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__forceinline__ __host__ __device__ void operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

/**
 *  _______ _________ _______  _______
 * (       )\__   __/(  ____ \(  ____ \
 * | () () |   ) (   | (    \/| (    \/
 * | || || |   | |   | (_____ | |
 * | |(_)| |   | |   (_____  )| |
 * | |   | |   | |         ) || |
 * | )   ( |___) (___/\____) || (____/\
 * |/     \|\_______/\_______)(_______/
 */

__forceinline__ __host__ __device__ float len(float3 v) {
    return sqrtf(len2(v));
}

__forceinline__ __host__ __device__ double len2(double2 v) {
    return v.x * v.x + v.y * v.y;
}

__forceinline__ __host__ __device__ float len2(float2 v) {
    return v.x * v.x + v.y * v.y;
}

__forceinline__ __host__ __device__ double3 cos(double3 v) {
    return make_double3(cos(v.x), cos(v.y), cos(v.z));
}

__forceinline__ __host__ __device__ double3 sin(double3 v) {
    return make_double3(sin(v.x), sin(v.y), sin(v.z));
}

__forceinline__ __host__ __device__ double2 operator*(double2 l, double r) {
    return make_double2(l.x * r, l.y * r);
}

__forceinline__ __host__ __device__ float2 operator*(float2 l, double r) {
    return make_float2(l.x * r, l.y * r);
}

#endif