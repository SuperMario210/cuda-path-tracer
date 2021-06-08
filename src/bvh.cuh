//
// Created by Mario Ruiz on 5/18/21.
//

#ifndef PATHTRACER_BVH_H
#define PATHTRACER_BVH_H


#include <vector>
#include "object.cuh"

#define STACK_SIZE              64

/**
 * Optimized min/max methods from Understanding the Efficiency of Ray Traversal on GPUs (Aila et al)
 * https://research.nvidia.com/publication/understanding-efficiency-ray-traversal-gpus
 */
__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){ return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }
__device__ __inline__ void swap(int& a, int& b){ int temp = a; a = b; b = temp;}

struct Triangle
{
    float3 v0, v1, v2;

    Triangle(const float3 &v0, const float3 &v1, const float3 &v2) : v0(v0), v1(v1), v2(v2)
    {

    }

    AABB bounding_box() const
    {
        float3 min = fminf(fminf(v0, v1), v2);
        float3 max = fmaxf(fmaxf(v0, v1), v2);

        if (max.x - min.x == 0) max.x += EPSILON;
        if (max.y - min.y == 0) max.y += EPSILON;
        if (max.z - min.z == 0) max.z += EPSILON;

        return {min, max};
    }
};

struct BVHNode {
    AABB aabb;
    size_t index;
    uint8_t num_children = 0;
    uint8_t axis = 0;
};

class BVH
{
public:
    float4 *triangles;
    float4 *nodes;
    float4 material;
    uint material_type;

    cudaTextureObject_t nodes_texture;

    void build(size_t start, size_t end, size_t &index, std::vector<Triangle> &h_triangles, std::vector<BVHNode> &h_nodes);
    void send_to_device(const std::vector<Triangle> &h_triangles, const std::vector<BVHNode> &h_nodes);

    /**
     * Optimized AABB intersection method from Understanding the Efficiency of Ray Traversal on GPUs
     */
    __device__ __inline__
    bool intersect_aabb(float3 inv_dir, float3 ood, float x0, float x1, float y0, float y1, float z0, float z1,
                        float t_max, float &t0) const
    {
        float lox = x0 * inv_dir.x - ood.x;
        float hix = x1 * inv_dir.x - ood.x;
        float loy = y0 * inv_dir.y - ood.y;
        float hiy = y1 * inv_dir.y - ood.y;
        float loz = z0 * inv_dir.z - ood.z;
        float hiz = z1 * inv_dir.z - ood.z;
        t0 = spanBeginKepler(lox, hix, loy, hiy, loz, hiz, EPSILON);
        float t1 = spanEndKepler  (lox, hix, loy, hiy, loz, hiz, t_max);
        return (t1 >= t0);
    }

    /**
     * Sven Woop's triangle intersection algorithm (http://www.sven-woop.de/papers/2004-GH-SaarCOR.pdf)
     */
    __device__ __inline__
    void intersect_triangle(float4 v00, float4 v11, float4 v22, float3 o, float3 d, float &t_max, float3 &n,
                            float4 &mat, uint &mat_type) const
    {
        float Oz = v00.w - o.x*v00.x - o.y*v00.y - o.z*v00.z;
        float invDz = 1.0f / (d.x*v00.x + d.y*v00.y + d.z*v00.z);
        float t = Oz * invDz;

        if (t > EPSILON && t < t_max) {
            float Ox = v11.w + o.x*v11.x + o.y*v11.y + o.z*v11.z;
            float Dx = d.x*v11.x + d.y*v11.y + d.z*v11.z;
            float u = Ox + t*Dx;

            if (u >= 0.0f) {
                float Oy = v22.w + o.x*v22.x + o.y*v22.y + o.z*v22.z;
                float Dy = d.x*v22.x + d.y*v22.y + d.z*v22.z;
                float v = Oy + t*Dy;

                if (v >= 0.0f && u + v <= 1.0f) {
                    // Record intersection
                    t_max = t;
                    n = cross(make_float3(v11.x, v11.y, v11.z), make_float3(v22.x, v22.y, v22.z));
                    mat = material;
                    mat_type = material_type;
                }
            }
        }
    }

public:
    BVH(std::vector<Triangle> objects, const float4 &mat, const uint mat_type);
    ~BVH();

    /**
     * This BVH traversal kernel is heavily based on the one from the paper Understanding the Efficiency of Ray
     * Traversal on GPUs (https://research.nvidia.com/publication/understanding-efficiency-ray-traversal-gpus).
     */
    __device__ __inline__
    void intersect(const float3 &o, const float3 &d, float &t_max, float3 &n, float4 &mat, uint &mat_type) const
    {
        int stack[STACK_SIZE];
        int stack_index = 0;
        int node_addr = 0;
        float3 inv_dir = 1.0f / d;
        float3 ood  = inv_dir * o;

        while(stack_index >= 0) {
            // Traverse the BVH until we find a leaf node
            while (node_addr >= 0 && stack_index >= 0) {
                // Fetch child nodes' bounding boxes
                const float4 n0xy = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 0);
                const float4 n1xy = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 1);
                const float4 nz   = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 2);
                const float4 tmp  = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 3);
                int2 cnodes       = *(int2*)&tmp;

                // Intersect the ray against the child nodes' bounding boxes
                float t0_left, t0_right;
                bool hit_left = intersect_aabb(inv_dir, ood, n0xy.x, n0xy.y, n0xy.z, n0xy.w, nz.x, nz.y, t_max, t0_left);
                bool hit_right = intersect_aabb(inv_dir, ood, n1xy.x, n1xy.y, n1xy.z, n1xy.w, nz.z, nz.w, t_max,
                                                t0_right);

                // Handle intersection with child nodes
                if (hit_left || hit_right) {
                    node_addr = (hit_left) ? cnodes.x : cnodes.y;
                    if (hit_left && hit_right) {
                        if (t0_right < t0_left) {
                            swap(node_addr, cnodes.y);
                        }

                        stack[++stack_index] = cnodes.y;
                    }
                } else {
                    node_addr = stack[stack_index--];
                }
            }

            // Intersect ray with leaf nodes
            while (node_addr < 0 && stack_index >= 0) {
                // Loop through all triangles in this leaf node
                for (int tri_addr = ~node_addr;; tri_addr += 3) {
                    const float4 v00 = triangles[tri_addr + 0];
                    const float4 v11 = triangles[tri_addr + 1];
                    const float4 v22 = triangles[tri_addr + 2];

                    // Check if we hit the end marker for this leaf node (no more triangles left)
                    if (__float_as_int(v00.x) == 0x80000000) {
                        break;
                    }

                    intersect_triangle(v00, v11, v22, o, d, t_max, n, mat, mat_type);
                }

                node_addr = stack[stack_index--];
            }
        }
    }
};


#endif //PATHTRACER_BVH_H
