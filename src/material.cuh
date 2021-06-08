//
// Created by SuperMario210 on 5/30/2021.
//

#ifndef CUDA_BASE_MATERIAL_CUH
#define CUDA_BASE_MATERIAL_CUH


#include "common.h"

__device__ __inline__ float3 diffuse(const float3 &n, curandState &rand_state) {
    // randomly generate point in sphere
    float z = curand_uniform(&rand_state) * 2.0f - 1.0f;
    float a = curand_uniform(&rand_state) * 2.0f * PI;
    float r = sqrtf(1.0f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);
    float3 dir = make_float3(x, y, z) + n;

    if (dir.x < EPSILON && dir.y < EPSILON && dir.z < EPSILON)
        return n;

    return normalize(dir);
}

__device__ __inline__ float3 refract(const float3 &v, const float3 &n, float ref_idx) {
    // Refracts a given ray
    float cos_t = fmin(dot(-v, n), 1.0f);
    float3 perp = (v + n * cos_t) * ref_idx;
    float3 parallel = n * -sqrt(fabs(1.0f - dot(perp, perp)));
    return normalize(perp + parallel);
}

__device__ __inline__ float reflectance(float cos_t, float ref_idx) {
    // Schlick approximation for reflectance
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf(1 - cos_t, 5);
}


#endif //CUDA_BASE_MATERIAL_CUH
