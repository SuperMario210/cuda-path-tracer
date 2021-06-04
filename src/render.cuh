//
// Created by SuperMario210 on 5/29/2021.
//

#ifndef CUDA_PATH_TRACER_KERNEL_CUH
#define CUDA_PATH_TRACER_KERNEL_CUH

#include "common.h"
#include "environment_map.cuh"
#include "bvh.cuh"

struct Path
{
public:
    uint pixel_x;           // y coordinate on the final image
    uint pixel_y;           // x coordinate on the final image
    float4 origin;          // (x, y, z, min_t)
    float4 normal;          // what direction is this ray coming from
    float4 direction;       // (x, y, z, max_t)
    float4 throughput;      // color throughput of this path
};

__host__ void launch_render_kernel(BVH *bvh, EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width,
                                   size_t height, size_t samples_per_pixel, dim3 grid, dim3 block, Path *paths, Path *next_paths);

#endif //CUDA_PATH_TRACER_KERNEL_CUH
