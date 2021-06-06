//
// Created by SuperMario210 on 5/29/2021.
//

#ifndef CUDA_PATH_TRACER_KERNEL_CUH
#define CUDA_PATH_TRACER_KERNEL_CUH

#include <device_atomic_functions.h>
#include "common.h"
#include "environment_map.cuh"
#include "bvh.cuh"

#define MAX_PATHS   0x100000

struct PathQueue
{
    uint pixel_index[MAX_PATHS];
    float4 origin[MAX_PATHS];
    float4 normal[MAX_PATHS];
    float4 direction[MAX_PATHS];
    float4 throughput[MAX_PATHS];
    uint depth[MAX_PATHS];
};

uint atomicAdd(uint *addr, uint val);

struct Queue
{
    uint index[MAX_PATHS];
    uint size;

    __device__ __inline__ void add(uint idx) {
        // TODO: per warp add
        uint i = atomicAdd(&size, 1);
        index[i] = idx;
    }
};

//struct Path
//{
//public:
//    uint pixel_x;           // y coordinate on the final image
//    uint pixel_y;           // x coordinate on the final image
//    float4 origin;          // (x, y, z, min_t)
//    float4 normal;          // what direction is this ray coming from
//    float4 direction;       // (x, y, z, max_t)
//    float4 throughput;      // color throughput of this path
//};

__host__ void launch_render_kernel(BVH *bvh, EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width,
                                   size_t height, size_t samples_per_pixel, dim3 grid, dim3 block, PathQueue *paths, Queue *new_paths, Queue *diffuse_paths);

#endif //CUDA_PATH_TRACER_KERNEL_CUH
