//
// Created by Mario Ruiz on 5/20/21.
//

#ifndef PATHTRACER_ENVIRONMENT_MAP_H
#define PATHTRACER_ENVIRONMENT_MAP_H


#include "common.h"

class EnvironmentMap
{
private:
    size_t width;
    size_t height;

    // CUDA Texture for this EnvironmentMap
    cudaArray_t data_array;
    cudaTextureObject_t texture_obj;

    // EnvironmentMap CDF
    size_t *marginal_lookup;
    size_t *conditional_lookup;

    void build_distribution(float *data);
    void send_to_device(const float *data);

public:
    explicit EnvironmentMap(const std::string &filename);
    ~EnvironmentMap();

    __device__ inline float3 sample(const float3 &dir)
    {
        auto phi = atan2f(dir.z, dir.x) * (0.5f / PI) + 0.5f;
        auto theta = acosf(dir.y) / PI;
        return make_float3(tex2D<float4>(texture_obj, phi, theta));
    }

    __device__ inline float3 sample_lights(curandState &rand_state)
    {
        size_t x = marginal_lookup[(size_t)(curand_uniform(&rand_state) * width - 0.5)];
        size_t y = conditional_lookup[x * height + (size_t)(curand_uniform(&rand_state) * height - 0.5)];

        float phi = 2.0f * x * PI / width;
        float theta = y * PI / height;
        float sin_t = -__sinf(theta);
        float sin_p = __sinf(phi);
        float cos_p = sqrtf(1.0f - sin_p * sin_p);

        return make_float3(sin_t * cos_p, -__cosf(theta), sin_t * sin_p);
    }

    __device__ inline float pdf(float3 &dir)
    {
        auto phi = atan2f(dir.z, dir.x) * (0.5f / PI) + 0.5f;
        auto theta = acosf(dir.y) / PI;
        return tex2D<float4>(texture_obj, phi, theta).w;
    }
};


#endif //PATHTRACER_ENVIRONMENT_MAP_H
