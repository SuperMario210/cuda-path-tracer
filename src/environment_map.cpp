//
// Created by Mario Ruiz on 5/20/21.
//

#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

#include "environment_map.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"


static inline float luminance(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

EnvironmentMap::EnvironmentMap(const std::string &filename)
{
    // Load hdr
    int x, y, n;
    auto data = stbi_loadf(filename.c_str(), &x, &y, &n, 4);
    width = x;
    height = y;

    // Calculate necessary data and send to device
    build_distribution(data);
    create_texture(data);

    stbi_image_free(data);
}

EnvironmentMap::~EnvironmentMap()
{

    cudaDestroyTextureObject(texture_obj);
    cudaFreeArray(data_array);
}

void EnvironmentMap::build_distribution(const float *data)
{
    std::cout << "Building environment map...\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    auto d_marginal_cdf = new float[width];
    auto d_conditional_cdf = new float[width * height];
    auto d_marginal_lookup = new size_t[width];
    auto d_conditional_lookup = new size_t[width * height];

    // Compute conditional CDF
    float row_sum = 0;
    for (size_t x = 0; x < width; x++) {
        float col_sum = 0;

        // Compute conditional CDF for pixel y given column x
        for (size_t y = 0; y < height; y++) {
            size_t i = 4 * ((height - y - 1) * width + x);
            float sin_theta = sinf((y + 0.5f) * PI / height);
            float weight = luminance(data[i + 0], data[i + 1], data[i + 2]) * sin_theta;

            col_sum += weight;
            d_conditional_cdf[x * height + y] = col_sum;
        }

        // Normalize conditional CDF
        if (col_sum > 0) {
            for (size_t y = 0; y < height; y++) {
                d_conditional_cdf[x * height + y] /= col_sum;
            }
        }

        row_sum += col_sum;
        d_marginal_cdf[x] = row_sum;
    }

    // Normalize marginal CDF
    if (row_sum > 0) {
        for (size_t x = 0; x < width; x++) {
            d_marginal_cdf[x] /= row_sum;
        }
    }

    // Precompute indices for sampling
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            float r = (float) (y + 1) / height;
            auto cond_cdf = &d_conditional_cdf[x * height];
            auto y_ptr = std::lower_bound(cond_cdf, cond_cdf + height, r);
            d_conditional_lookup[x * height + y] = y_ptr - cond_cdf;
            if (d_conditional_lookup[x * height + y] >= height) {
                d_conditional_lookup[x * height + y] = height - 1;
            }
        }
    }

    for (size_t x = 0; x < width; x++) {
        float r = (float) (x + 1) / width;
        auto x_ptr = std::lower_bound(d_marginal_cdf, d_marginal_cdf + width, r);
        d_marginal_lookup[x] = x_ptr - d_marginal_cdf;
        if (d_marginal_lookup[x] >= width) {
            d_marginal_lookup[x] = width - 1;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Built environment map in " << ms_int.count() << " ms\n";

    gpuErrchk(cudaMalloc(&marginal_cdf, width * sizeof(float)));
    gpuErrchk(cudaMemcpy(marginal_cdf, d_marginal_cdf, width * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&conditional_cdf, width * height * sizeof(float)));
    gpuErrchk(cudaMemcpy(conditional_cdf, d_conditional_cdf, width * height * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&marginal_lookup, width * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(marginal_lookup, d_marginal_lookup, width * sizeof(size_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&conditional_lookup, width * height * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(conditional_lookup, d_conditional_lookup, width * height * sizeof(size_t), cudaMemcpyHostToDevice));


    std::cout << "Delete \n";

    delete[] d_marginal_cdf;
    delete[] d_conditional_cdf;
    delete[] d_marginal_lookup;
    delete[] d_conditional_lookup;

    std::cout << "Done \n";

}

void EnvironmentMap::create_texture(const float *data) {
    // Create cuda array to hold texture data
    const size_t width_bytes = width * 4 * sizeof(float);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    gpuErrchk(cudaMallocArray(&data_array, &channelDesc, width, height));
    gpuErrchk(cudaMemcpy2DToArray(data_array, 0, 0, data, width_bytes, width_bytes, height, cudaMemcpyHostToDevice));

    // Specify texture
    cudaResourceDesc resource_desc{};
    memset(&resource_desc, 0, sizeof(resource_desc));
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = data_array;

    // Specify texture object parameters
    cudaTextureDesc texture_desc{};
    memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.addressMode[0] = cudaAddressModeWrap;
    texture_desc.addressMode[1] = cudaAddressModeWrap;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 1;

    // Create texture object
    texture_obj = 0;
    cudaCreateTextureObject(&texture_obj, &resource_desc, &texture_desc, nullptr);
}

EnvironmentMapData EnvironmentMap::get_data() const {
    return EnvironmentMapData {
        texture_obj,
        marginal_cdf,
        conditional_cdf,
        marginal_lookup,
        conditional_lookup
    };
}


