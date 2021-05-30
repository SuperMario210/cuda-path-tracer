//
// Created by SuperMario210 on 5/29/2021.
//

#ifndef CUDA_PATH_TRACER_KERNEL_CUH
#define CUDA_PATH_TRACER_KERNEL_CUH

#include "../include/cutil_math.cuh"
#include "environment_map.cuh"

__host__ void launch_render_kernel(EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width, size_t height,
                                   size_t samples_per_pixel, dim3 grid, dim3 block);

#endif //CUDA_PATH_TRACER_KERNEL_CUH
