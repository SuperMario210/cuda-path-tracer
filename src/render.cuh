//
// Created by SuperMario210 on 5/29/2021.
//

#ifndef CUDA_PATH_TRACER_KERNEL_CUH
#define CUDA_PATH_TRACER_KERNEL_CUH

#include <cooperative_groups.h>
#include "common.h"
#include "environment_map.cuh"
#include "bvh.cuh"
#include "scene.cuh"

#define MAX_PATHS   0x200000
#define IS_ACTIVE   0x1
#define IS_NEW_PATH 0x2
#define IS_DIFFUSE  0x4
#define IS_MIRROR   0x8
#define IS_GLOSSY   0x10
#define IS_GLASS    0x20


struct PathData
{
    uint pixel_index[MAX_PATHS];
    float4 origin[MAX_PATHS];
    float4 normal[MAX_PATHS];
    float4 direction[MAX_PATHS];
    float4 throughput[MAX_PATHS];
    float4 material[MAX_PATHS];
    uint depth[MAX_PATHS];
    uint flags[MAX_PATHS];

    __device__ __inline__ bool get_flag(const uint index, const uint flag)
    {
        return flags[index] & flag;
    }

    __device__ __inline__ void set_flag(const uint index, const uint flag)
    {
        flags[index] |= flag;
    }

    __device__ __inline__ void clear_flag(const uint index, const uint flag)
    {
        flags[index] &= ~flag;
    }
};

__host__ void render_scene(Scene *scene, size_t width, size_t height, size_t samples_per_pixel);

#endif //CUDA_PATH_TRACER_KERNEL_CUH
