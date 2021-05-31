//
// Created by Mario Ruiz on 5/20/21.
//

#ifndef PATHTRACER_ENVIRONMENT_MAP_H
#define PATHTRACER_ENVIRONMENT_MAP_H


#include <curand_kernel.h>
#include "image.h"
#include "util.cuh"

struct EnvironmentMapData
{
    size_t width;
    size_t height;
    cudaTextureObject_t texture_obj;
    size_t *marginal_lookup;
    size_t *conditional_lookup;
};

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
    void create_texture(const float *data);

public:
    explicit EnvironmentMap(const std::string &filename);
    ~EnvironmentMap();

    EnvironmentMapData get_data() const;
};


#endif //PATHTRACER_ENVIRONMENT_MAP_H
