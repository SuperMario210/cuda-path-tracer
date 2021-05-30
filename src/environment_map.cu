//
// Created by Mario Ruiz on 5/20/21.
//

#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

#include "environment_map.cuh"
#include "util.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"


static inline float luminance(const float3 &c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

EnvironmentMap::EnvironmentMap(const std::string &filename) : Image()
{
    // Load hdr
    int x, y, n;
    float *raw_data = stbi_loadf(filename.c_str(), &x, &y, &n, 0);
    assert(n == 3);

    // Copy raw image data
    width = x;
    height = y;
    data = new float3[x * y];
    for (size_t i = 0; i < x * y; i++) {
        data[i].x = raw_data[n * i + 0];
        data[i].y = raw_data[n * i + 1];
        data[i].z = raw_data[n * i + 2];
    }
    stbi_image_free(raw_data);

    build_distribution();
}

EnvironmentMap::~EnvironmentMap()
{
    delete[] marginal_cdf;
    delete[] conditional_cdf;
    delete[] marginal_lookup;
    delete[] conditional_lookup;
}

void EnvironmentMap::build_distribution()
{
    std::cout << "Building environment map...\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    marginal_cdf = new float[width];
    conditional_cdf = new float[width * height];
    marginal_lookup = new size_t[width];
    conditional_lookup = new size_t[width * height];

    // Compute conditional CDF
    float row_sum = 0;
    for (size_t x = 0; x < width; x++) {
        float col_sum = 0;

        // Compute conditional CDF for pixel y given column x
        for (size_t y = 0; y < height; y++) {
            float sin_theta = sinf((y + 0.5f) * PI / height);
            float weight = luminance(get_pixel(x, y)) * sin_theta;

            col_sum += weight;
            conditional_cdf[x * height + y] = col_sum;
        }

        // Normalize conditional CDF
        if (col_sum > 0) {
            for (size_t y = 0; y < height; y++) {
                conditional_cdf[x * height + y] /= col_sum;
            }
        }

        row_sum += col_sum;
        marginal_cdf[x] = row_sum;
    }

    // Normalize marginal CDF
    if (row_sum > 0) {
        for (size_t x = 0; x < width; x++) {
            marginal_cdf[x] /= row_sum;
        }
    }

    // Precompute indices for sampling
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            float r = (float) (y + 1) / height;
            auto cond_cdf = &conditional_cdf[x * height];
            auto y_ptr = std::lower_bound(cond_cdf, cond_cdf + height, r);
            conditional_lookup[x * height + y] = y_ptr - cond_cdf;
            if (conditional_lookup[x * height + y] >= height) {
                conditional_lookup[x * height + y] = height - 1;
            }
        }
    }

    for (size_t x = 0; x < width; x++) {
        float r = (float) (x + 1) / width;
        auto x_ptr = std::lower_bound(marginal_cdf, marginal_cdf + width, r);
        marginal_lookup[x] = x_ptr - marginal_cdf;
        if (marginal_lookup[x] >= width) {
            marginal_lookup[x] = width - 1;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Built environment map in " << ms_int.count() << " ms\n";
}

__host__ EnvironmentMap *EnvironmentMap::copy_to_device() const {
    // Allocate device memory to hold this EnvironmentMap's data
    EnvironmentMap *envmap_d;
    size_t buffer_size = sizeof(EnvironmentMap) + width * height * sizeof(float3) + (width * height + width) *
            (sizeof(float) + sizeof(size_t));
    gpuErrchk(cudaMalloc(&envmap_d, buffer_size));
    auto *buffer_ptr = reinterpret_cast<uint8_t*>(envmap_d);

    // Start copying over this EnvironmentMap's data to the device EnvironmentMap
    size_t num_bytes = sizeof(EnvironmentMap);
    EnvironmentMap envmap_h{};
    envmap_h.width = width;
    envmap_h.height = height;
    buffer_ptr += num_bytes;

    // Copy image data
    num_bytes = width * height * sizeof(float3);
    gpuErrchk(cudaMemcpy(buffer_ptr, data, num_bytes, cudaMemcpyHostToDevice));
    envmap_h.data = reinterpret_cast<float3 *>(buffer_ptr);
    buffer_ptr += num_bytes;

    // Copy marginal and conditional CDF data
    num_bytes = width * sizeof(float);
    gpuErrchk(cudaMemcpy(buffer_ptr, marginal_cdf, num_bytes, cudaMemcpyHostToDevice));
    envmap_h.marginal_cdf = reinterpret_cast<float *>(buffer_ptr);
    buffer_ptr += num_bytes;
    num_bytes = width * height * sizeof(float);
    gpuErrchk(cudaMemcpy(buffer_ptr, conditional_cdf, num_bytes, cudaMemcpyHostToDevice));
    envmap_h.conditional_cdf = reinterpret_cast<float *>(buffer_ptr);
    buffer_ptr += num_bytes;

    // Copy marginal and conditional lookup data
    num_bytes = width * sizeof(size_t);
    gpuErrchk(cudaMemcpy(buffer_ptr, marginal_lookup, num_bytes, cudaMemcpyHostToDevice));
    envmap_h.marginal_lookup = reinterpret_cast<size_t *>(buffer_ptr);
    buffer_ptr += num_bytes;
    num_bytes = width * height * sizeof(size_t);
    gpuErrchk(cudaMemcpy(buffer_ptr, conditional_lookup, num_bytes, cudaMemcpyHostToDevice));
    envmap_h.conditional_lookup = reinterpret_cast<size_t *>(buffer_ptr);
    buffer_ptr += num_bytes;

    gpuErrchk(cudaMemcpy(envmap_d, &envmap_h, sizeof(EnvironmentMap), cudaMemcpyHostToDevice));

    // Clear host EnvironmentMap so destructor is not run
    envmap_h.data = nullptr;
    envmap_h.marginal_cdf = nullptr;
    envmap_h.conditional_cdf = nullptr;
    envmap_h.marginal_lookup = nullptr;
    envmap_h.conditional_lookup = nullptr;

    return envmap_d;
}


