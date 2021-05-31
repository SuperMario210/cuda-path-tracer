//
// Created by Mario Ruiz on 5/17/21.
//

#ifndef PATHTRACER_UTIL_H
#define PATHTRACER_UTIL_H


#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "ray.cuh"
#include "../include/cutil_math.cuh"

#define PI          3.14159265358979323846264338327950288f
#define EPSILON     0.001f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}




#endif //PATHTRACER_UTIL_H
