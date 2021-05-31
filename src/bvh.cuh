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
private:
    BVHNode *nodes;
    Triangle *triangles;

public:
    BVH(std::vector<Triangle> objects);
    ~BVH();

    void build(size_t start, size_t end, size_t &index, std::vector<Triangle> &h_triangles, std::vector<BVHNode> &h_nodes);

    __device__ bool intersect(const Ray &r, Intersection &intersect) const
    {
        bool hit = false;
        float3 inv_dir = 1 / r.direction;
//        bool is_dir_neg[3] = {inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0};

        size_t stack[32];
        size_t stack_index = 0;
        stack[stack_index++] = 0;

        while (stack_index > 0) {
            size_t index = stack[--stack_index];
            const BVHNode *node = &nodes[index];

            if (node->intersect(r, inv_dir, EPSILON, intersect.t)) {
                if (node->num_children > 0) {
                    for (auto i = 0; i < node->num_children; i++) {
                        if (triangles[node->index + i].intersect(r, intersect)) {
                            hit = true;
                        }
                    }
                } else {
                    stack[stack_index++] = node->index;
                    stack[stack_index++] = index + 1;

//                    stack[stack_index++] = is_dir_neg[node->axis] ? index + 1 : node->index;
//                    stack[stack_index++] = is_dir_neg[node->axis] ? node->index : index + 1;
                }
            }
        }

        return hit;
    }

};


#endif //PATHTRACER_BVH_H
