//
// Created by SuperMario210 on 5/29/2021.
//

#ifndef CUDA_PATH_TRACER_KERNEL_CUH
#define CUDA_PATH_TRACER_KERNEL_CUH

#include "common.h"
#include "environment_map.cuh"
#include "bvh.cuh"

__host__ void launch_render_kernel(BVH *bvh, EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width,
                                   size_t height, size_t samples_per_pixel, dim3 grid, dim3 block);

#endif //CUDA_PATH_TRACER_KERNEL_CUH
