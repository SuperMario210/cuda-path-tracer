//
// Created by Mario Ruiz on 5/18/21.
//

#ifndef PATHTRACER_BVH_H
#define PATHTRACER_BVH_H


#include <vector>
#include "object.cuh"

#define STACK_SIZE              64

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
public:
    float3 v0, v1, v2;

    __host__ Triangle(const float3 &v0, const float3 &v1, const float3 &v2) : v0(v0), v1(v1), v2(v2)
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
    float4 material = make_float4(0.65);
    uint material_type = 0x8;

    cudaTextureObject_t nodes_texture;

    void build(size_t start, size_t end, size_t &index, std::vector<Triangle> &h_triangles, std::vector<BVHNode> &h_nodes);
    void send_to_device(const std::vector<Triangle> &h_triangles, const std::vector<BVHNode> &h_nodes);

public:
    BVH(std::vector<Triangle> objects);
    ~BVH();

    __device__ __inline__
    void intersect(const float3 &o, const float3 &d, float &t_max, float3 &n, float4 &mat, uint &mat_type) const
    {
        // Traversal stack in CUDA thread-local memory.
        int traversalStack[STACK_SIZE];

        // Live state during traversal, stored in registers.
        int stackPtr = 0;
        int node_addr = 0;
        float3 idir = 1.0f / d;
        float3 ood  = idir * o;

        // Traversal loop.
        while(stackPtr >= 0) {
            while (node_addr >= 0 && stackPtr >= 0) {
                // Fetch AABBs of the two child nodes.
                const float4 n0xy = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
                const float4 n1xy = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
                const float4 nz   = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
                const float4 tmp  = tex1Dfetch<float4>(nodes_texture, node_addr * 4 + 3); // child_index0, child_index1
                int2 cnodes       = *(int2*)&tmp;

                // Intersect the ray against the child nodes.
                const float c0lox = n0xy.x * idir.x - ood.x;
                const float c0hix = n0xy.y * idir.x - ood.x;
                const float c0loy = n0xy.z * idir.y - ood.y;
                const float c0hiy = n0xy.w * idir.y - ood.y;
                const float c0loz = nz.x   * idir.z - ood.z;
                const float c0hiz = nz.y   * idir.z - ood.z;
                const float c1loz = nz.z   * idir.z - ood.z;
                const float c1hiz = nz.w   * idir.z - ood.z;
                const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, EPSILON);
                const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, t_max);
                const float c1lox = n1xy.x * idir.x - ood.x;
                const float c1hix = n1xy.y * idir.x - ood.x;
                const float c1loy = n1xy.z * idir.y - ood.y;
                const float c1hiy = n1xy.w * idir.y - ood.y;
                const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, EPSILON);
                const float c1max = spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, t_max);

                const bool traverseChild0 = (c0max >= c0min);
                const bool traverseChild1 = (c1max >= c1min);

                // Otherwise => fetch child pointers.
                if (traverseChild0 || traverseChild1) {
                    node_addr = (traverseChild0) ? cnodes.x : cnodes.y;

                    // Both children were intersected => push the farther one.
                    if (traverseChild0 && traverseChild1) {
                        if (c1min < c0min) {
                            swap(node_addr, cnodes.y);
                        }

                        traversalStack[++stackPtr] = cnodes.y;
                    }
                } else {
                    node_addr = traversalStack[stackPtr--];
                }
            }

            while (node_addr < 0 && stackPtr >= 0) {
                for (int tri_addr = ~node_addr;; tri_addr += 3) {
                    const float4 v00 = triangles[tri_addr + 0];
                    const float4 v11 = triangles[tri_addr + 1];
                    const float4 v22 = triangles[tri_addr + 2];

                    // End marker (negative zero) => all triangles processed.
                    if (__float_as_int(v00.x) == 0x80000000) {
                        break;
                    }

                    // Woop triangle intersection
                    float Oz = v00.w - o.x*v00.x - o.y*v00.y - o.z*v00.z;
                    float invDz = 1.0f / (d.x*v00.x + d.y*v00.y + d.z*v00.z);
                    float t = Oz * invDz;

                    if (t > EPSILON && t < t_max) {
                        // Compute and check barycentric u.
                        float Ox = v11.w + o.x*v11.x + o.y*v11.y + o.z*v11.z;
                        float Dx = d.x*v11.x + d.y*v11.y + d.z*v11.z;
                        float u = Ox + t*Dx;

                        if (u >= 0.0f) {
                            // Compute and check barycentric v.
                            float Oy = v22.w + o.x*v22.x + o.y*v22.y + o.z*v22.z;
                            float Dy = d.x*v22.x + d.y*v22.y + d.z*v22.z;
                            float v = Oy + t*Dy;

                            if (v >= 0.0f && u + v <= 1.0f) {
                                // Record intersection.
                                t_max = t;
                                n = cross(make_float3(v11.x, v11.y, v11.z), make_float3(v22.x, v22.y, v22.z));
                                mat = material;
                                mat_type = material_type;
                            }
                        }
                    }
                } // triangle

                node_addr = traversalStack[stackPtr--];
            } // leaf
        } // traversal
    }
};


#endif //PATHTRACER_BVH_H
