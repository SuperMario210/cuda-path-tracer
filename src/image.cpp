//
// Created by Mario Ruiz on 5/11/21.
//

#include <fstream>
#include <cassert>
#include <limits>
#include <cmath>
#include <iostream>

#include "image.h"
#include "../include/lodepng.h"

Image::Image(size_t x, size_t y) : width(x), height(y), data(new float3[x * y])
{
}

Image::~Image()
{
    delete[] data;
}

float3 Image::get_pixel(size_t x, size_t y) const
{
    assert(x >= 0 && x < width && y >= 0 && y < height);
    return data[(height - y - 1) * width + x];
}

void Image::set_pixel(size_t x, size_t y, const float3 &color)
{
//    assert(x >= 0 && x < width && y >= 0 && y < height);
    x = x < width ? x : width - 1;
    y = y < height ? y : height - 1;
    data[(height - y - 1) * width + x] = color;
}

void Image::tone_map(float exposure, float gamma) {
    for (int i = 0; i < width * height; i++) {
        data[i] = make_float3(1.0) - exp(-data[i] * exposure);
        data[i] = pow(data[i], 1.0f / gamma);
    }
}

void Image::save_png(const std::string &base_filename) const
{
    // Write image data to buffer
    const auto max_val = std::numeric_limits<uint16_t>::max();
    auto buffer = new uint16_t[3 * width * height];
    for (auto i = 0; i < width * height; i++) {
        buffer[i*3 + 0] = static_cast<uint16_t>(max_val * data[i].x);
        buffer[i*3 + 1] = static_cast<uint16_t>(max_val * data[i].y);
        buffer[i*3 + 2] = static_cast<uint16_t>(max_val * data[i].z);

        // Convert to big endian
        buffer[i*3 + 0] = (buffer[i*3 + 0] >> 8) | (buffer[i*3 + 0] << 8);
        buffer[i*3 + 1] = (buffer[i*3 + 1] >> 8) | (buffer[i*3 + 1] << 8);
        buffer[i*3 + 2] = (buffer[i*3 + 2] >> 8) | (buffer[i*3 + 2] << 8);
    }

    // Encode image data
    lodepng::State state;
    state.info_raw.colortype = LCT_RGB;
    state.info_raw.bitdepth = 16;
    state.info_png.color.colortype = LCT_RGB;
    state.info_png.color.bitdepth = 16;
    state.encoder.auto_convert = 0;
    std::vector<unsigned char> bytes;
    unsigned error = lodepng::encode(bytes, reinterpret_cast<unsigned char*>(buffer), width, height, state);

    // Save image data
    if (error) {
        std::cerr << "Error encoding PNG: " << lodepng_error_text(error) << "\n";
    } else {
        lodepng::save_file(bytes, base_filename + ".png");
    }

    delete[] buffer;
}
