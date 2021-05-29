//
// Created by Mario Ruiz on 5/11/21.
//

#ifndef PATHTRACER_IMAGE_H
#define PATHTRACER_IMAGE_H

#include <cstddef>
#include <string>
#include "../include/cutil_math.cuh"


class Image
{
public:
    size_t width;
    size_t height;
    float3 *data;

public:
    Image() = default;
    Image(size_t x, size_t y);
    ~Image();

    float3 get_pixel(size_t x, size_t y) const;
    void set_pixel(size_t x, size_t y, const float3 &color);
    void tone_map(float exposure, float gamma);
    void save_png(const std::string &base_filename) const;
};


#endif //PATHTRACER_IMAGE_H
