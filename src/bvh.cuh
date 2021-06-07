//
// Created by Mario Ruiz on 5/18/21.
//

#ifndef PATHTRACER_BVH_H
#define PATHTRACER_BVH_H


#include <vector>
#include "object.cuh"

struct BVHNode {
    AABB aabb;
    size_t index;
    uint8_t num_children = 0;
    uint8_t axis = 0;

    __device__ inline bool intersect(Ray r, float3 inv_dir, float t_min, float t_max) const
    {
        float3 t0 = (aabb.min - r.origin) * inv_dir;
        float3 t1 = (aabb.max - r.origin) * inv_dir;

        float3 tsmaller = fminf(t0, t1);
        float3 tbigger  = fmaxf(t0, t1);

        t_min = fmaxf(t_min, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
        t_max = fminf(t_max, fminf(tbigger.x, fminf(tbigger.y, tbigger.z)));

        return (t_min < t_max);
    }
};

class BVH
{
public:
    float4 *triangles;
    float4 *nodes;

    cudaTextureObject_t nodes_texture;

    void build(size_t start, size_t end, size_t &index, std::vector<Triangle> &h_triangles, std::vector<BVHNode> &h_nodes);
    void send_to_device(const std::vector<Triangle> &h_triangles, const std::vector<BVHNode> &h_nodes);

public:
    BVH(std::vector<Triangle> objects);
    ~BVH();
};


#endif //PATHTRACER_BVH_H
