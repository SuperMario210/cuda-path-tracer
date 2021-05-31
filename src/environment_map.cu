//
// Created by Mario Ruiz on 5/20/21.
//

#include <chrono>
#include <algorithm>
#include <texture_indirect_functions.h>

#include "environment_map.cuh"

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

    cudaFree(marginal_lookup);
    cudaFree(conditional_lookup);
}

void EnvironmentMap::build_distribution(float *data)
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

    // Compute PDFs for each pixel
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            size_t i = 4 * ((height - y - 1) * width + x) + 3;

            float sin_theta = sinf(y * PI / height);
            if (sin_theta == 0) {
                data[i] = 0;
                continue;
            }

            auto cond_cdf = &d_conditional_cdf[x * height];
            float pdf_x = (x == 0) ? d_marginal_cdf[0] : (d_marginal_cdf[x] - d_marginal_cdf[x - 1]);
            float pdf_y = (y == 0) ? cond_cdf[0] : (cond_cdf[y] - cond_cdf[y - 1]);

            data[i] = static_cast<float>(width * height) * (pdf_x * pdf_y) / (sin_theta * 2.0f * PI * PI);
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

    gpuErrchk(cudaMalloc(&marginal_lookup, width * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(marginal_lookup, d_marginal_lookup, width * sizeof(size_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&conditional_lookup, width * height * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(conditional_lookup, d_conditional_lookup, width * height * sizeof(size_t), cudaMemcpyHostToDevice));

    delete[] d_marginal_cdf;
    delete[] d_conditional_cdf;
    delete[] d_marginal_lookup;
    delete[] d_conditional_lookup;
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