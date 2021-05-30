//
// Created by Mario Ruiz on 5/20/21.
//

#ifndef PATHTRACER_ENVIRONMENT_MAP_H
#define PATHTRACER_ENVIRONMENT_MAP_H


#include <curand_kernel.h>
#include "image.cuh"
#include "util.cuh"

class EnvironmentMap : public Image
{
private:
    float *marginal_cdf;
    float *conditional_cdf;
    size_t *marginal_lookup;
    size_t *conditional_lookup;

    __host__ void build_distribution();

    __host__ __device__ inline void direction_to_pixel(const float3 &direction, size_t &x, size_t &y) const
    {
        // kinda slow, is there a way to speed up?
        auto dir = normalize(direction);
        auto phi = atan2f(dir.z, dir.x) * (0.5f / PI) + 0.5f;
        auto theta = acosf(-dir.y) / PI;

        x = static_cast<size_t> (phi * float(width - 1));
        y = static_cast<size_t> (theta * float(height - 1));
    }

    __host__ __device__ inline float3 pixel_to_direction(size_t x, size_t y) const
    {
        auto phi = 2.0f * x * PI / width;
        auto theta = y * PI / height;
        auto sin_t = -sinf(theta);
        return {sin_t * cosf(phi), -cosf(theta), sin_t * sinf(phi)};
    }

public:
    EnvironmentMap() = default;
    __host__ explicit EnvironmentMap(const std::string &filename);
    __host__ ~EnvironmentMap();

    __device__ inline float3 sample(const float3 &direction) const
    {
        size_t x, y;
        direction_to_pixel(direction, x, y);
        return get_pixel(x, y);
    }

    __device__ inline float sample_lights(float3 &color, float3 &direction, curandState &rand_state) const
    {
        size_t x = marginal_lookup[(int)(curand_uniform(&rand_state) * width)];
        size_t y = conditional_lookup[x * height + (int)(curand_uniform(&rand_state) * height)];

        direction = pixel_to_direction(x, y);
        color = get_pixel(x, y);

        return pdf(x, y);
    }

    __device__ inline float pdf(const float3 &direction) const
    {
        size_t x = 0, y = 0;
        direction_to_pixel(direction, x, y);
        return pdf(x, y);
    }

    __device__ inline float pdf(size_t x, size_t y) const
    {
        float sin_theta = sinf(y * PI / height);
        if (sin_theta == 0) {
            return 0;
        }

        auto cond_cdf = &conditional_cdf[x * height];
        float pdf_x = (x == 0) ? marginal_cdf[0] : (marginal_cdf[x] - marginal_cdf[x - 1]);
        float pdf_y = (y == 0) ? cond_cdf[0] : (cond_cdf[y] - cond_cdf[y - 1]);

        return static_cast<float>(width * height) * (pdf_x * pdf_y) / (sin_theta * 2.0f * PI * PI);
    }

    __host__ EnvironmentMap *copy_to_device() const;
};


#endif //PATHTRACER_ENVIRONMENT_MAP_H
