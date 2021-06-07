//
// Created by SuperMario210 on 5/29/2021.
//

#ifndef CUDA_PATH_TRACER_KERNEL_CUH
#define CUDA_PATH_TRACER_KERNEL_CUH

#include <cooperative_groups.h>
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

struct Queue
{
    uint index[MAX_PATHS];
    uint size;

    __device__ __inline__ void add(uint idx) {
        auto g = cooperative_groups::coalesced_threads();
        int warp_res;
        if(g.thread_rank() == 0)
            warp_res = atomicAdd(&size, g.size());

        int i = g.shfl(warp_res, 0) + g.thread_rank();
        index[i] = idx;
    }

    __host__ void clear() {
        cudaMemset(&size, 0, sizeof(uint));
    }

    __host__ uint get_size() {
        uint h_size;
        cudaMemcpy(&h_size, &size, sizeof(uint), cudaMemcpyDeviceToHost);
        return h_size;
    }
};

__host__ void launch_render_kernel(BVH *bvh, EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width,
                                   size_t height, size_t samples_per_pixel, dim3 grid, dim3 block, PathQueue *paths, Queue *new_paths, Queue *diffuse_paths);

#endif //CUDA_PATH_TRACER_KERNEL_CUH
