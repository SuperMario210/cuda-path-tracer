//
// Created by Mario Ruiz on 5/11/21.
//

#ifndef PATHTRACER_IMAGE_H
#define PATHTRACER_IMAGE_H

#include <cstddef>
#include <string>
#include <cassert>
#include "../include/cutil_math.cuh"


class Image
{
public:
    size_t width;
    size_t height;
    float3 *data;

public:
    inline Image() = default;

    __host__ __device__ inline Image(size_t x, size_t y) : width(x), height(y), data(new float3[x * y])
    {

    }
    __host__ __device__ inline ~Image()
    {
        delete[] data;
    }

    __host__ __device__ inline float3 get_pixel(size_t x, size_t y) const
    {
        assert(x < width && y < height);
        return data[(height - y - 1) * width + x];
    }

    __host__ __device__ inline void set_pixel(size_t x, size_t y, const float3 &color) const
    {
        assert(x < width && y < height);
        data[(height - y - 1) * width + x] = color;
    }

    __host__ void tone_map(float exposure, float gamma);
    __host__ void save_png(const std::string &base_filename) const;
};


#endif //PATHTRACER_IMAGE_H
