#include "camera.cuh"
#include "render.cuh"

#define IMPORTANCE_SAMPLING
#define MAX_DEPTH               16
#define STACK_SIZE              64
#define ENTRYPOINT_SENTINEL     0x76543210
#define FULL_MASK               0xffffffff

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

__device__ bool intersect_bvh(BVH *bvh, const Ray &r, Intersection &intersect) {
    ///////////////////////////////////////////
    //// KEPLER KERNEL
    ///////////////////////////////////////////

    // BVH layout Compact2 for Kepler
    int traversalStack[STACK_SIZE];

    // Live state during traversal, stored in registers.
    float   origx, origy, origz;    // Ray origin.
    float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.
    float   idirx, idiry, idirz;    // 1 / ray direction
    float   oodx, oody, oodz;       // ray origin / ray direction

    int*    stackPtr;               // Current position in traversal stack.
    int     leafAddr;               // If negative, then first postponed leaf, non-negative if no leaf (innernode).
    int     nodeAddr;
    int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.
    float3  hitNormal;              // normal of the closest intersection.


    // Initialize (stores local variables in registers)
    {
        origx = r.origin.x;
        origy = r.origin.y;
        origz = r.origin.z;
        dirx = r.direction.x;
        diry = r.direction.y;
        dirz = r.direction.z;
        tmin = EPSILON;

        // ooeps is very small number, used instead of raydir xyz component when that component is near zero
        float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
        idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : copysignf(ooeps, dirx)); // inverse ray direction
        idiry = 1.0f / (fabsf(diry) > ooeps ? diry : copysignf(ooeps, diry)); // inverse ray direction
        idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : copysignf(ooeps, dirz)); // inverse ray direction
        oodx = origx * idirx;  // ray origin / ray direction
        oody = origy * idiry;  // ray origin / ray direction
        oodz = origz * idirz;  // ray origin / ray direction

        // Setup traversal + initialisation

        traversalStack[0] = ENTRYPOINT_SENTINEL; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
        stackPtr = &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
        leafAddr = 0;   // No postponed leaf.
        nodeAddr = 0;   // Start from the root.
        hitIndex = -1;  // No triangle intersected so far.
        hitT = intersect.t; // tmax
    }

    // Traversal loop.

    while (nodeAddr != ENTRYPOINT_SENTINEL)
    {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        bool searchingLeaf = true; // required for warp efficiency
        while (nodeAddr >= 0 && nodeAddr != ENTRYPOINT_SENTINEL)
        {
            // Fetch AABBs of the two child nodes.

            // nodeAddr is an offset in number of bytes (char) in gpuNodes array

            float4 n0xy = tex1Dfetch<float4>(bvh->nodes_texture, nodeAddr * 4); // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            float4 n1xy = tex1Dfetch<float4>(bvh->nodes_texture, nodeAddr * 4 + 1); // childnode 1, xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            float4 nz = tex1Dfetch<float4>(bvh->nodes_texture, nodeAddr * 4 + 2); // childnode 0 and 1, z-bounds (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            float4 tmp = tex1Dfetch<float4>(bvh->nodes_texture, nodeAddr * 4 + 3); // contains indices to 2 childnodes in case of innernode, see below
            int2 cnodes = *(int2*)&tmp; // cast first two floats to int
            // (childindex = size of array during building, see CudaBVH.cpp)

            // compute ray intersections with BVH node bounding box

            /// RAY BOX INTERSECTION
            // Intersect the ray against the child nodes.
            float c0lox = n0xy.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
            float c0hix = n0xy.y * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
            float c0loy = n0xy.z * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
            float c0hiy = n0xy.w * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
            float c0loz = nz.x   * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
            float c0hiz = nz.y   * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
            float c1loz = nz.z   * idirz - oodz; // nz.z   = c1.lo.z, child 1 minbound z
            float c1hiz = nz.w   * idirz - oodz; // nz.w   = c1.hi.z, child 1 maxbound z
            float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
            float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)
            float c1lox = n1xy.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 minbound x
            float c1hix = n1xy.y * idirx - oodx; // n1xy.y = c1.hi.x, child 1 maxbound x
            float c1loy = n1xy.z * idiry - oody; // n1xy.z = c1.lo.y, child 1 minbound y
            float c1hiy = n1xy.w * idiry - oody; // n1xy.w = c1.hi.y, child 1 maxbound y
            float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
            float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

            // ray box intersection boundary tests:
            bool traverseChild0 = (c0min <= c0max);
            bool traverseChild1 = (c1min <= c1max);

            // Neither child was intersected => pop stack.
            if (!traverseChild0 && !traverseChild1) {
                nodeAddr = *stackPtr; // fetch next node by popping the stack
                stackPtr--; // popping decrements stackPtr by 4 bytes (because stackPtr is a pointer to char)
            }

                // Otherwise, one or both children intersected => fetch child pointers.
            else {
                // set nodeAddr equal to intersected childnode index (or first childnode when both children are intersected)
                nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                // Both children were intersected => push the farther one on the stack.
                if (traverseChild0 && traverseChild1) // store closest child in nodeAddr, swap if necessary
                {
                    if (c1min < c0min) swap(nodeAddr, cnodes.y);
                    stackPtr++;
                    *stackPtr = cnodes.y; // push furthest node on the stack
                }
            }

            // First leaf => postpone and continue traversal.
            // leafnodes have a negative index to distinguish them from inner nodes
            // if nodeAddr less than 0 -> nodeAddr is a leaf
            if (nodeAddr < 0 && leafAddr >= 0)
            {
                searchingLeaf = false; // required for warp efficiency
                leafAddr = nodeAddr;
                nodeAddr = *stackPtr;  // pops next node from stack
                stackPtr--;
            }

            // All SIMD lanes have found a leaf => process them.

            // to increase efficiency, check if all the threads in a warp have found a leaf before proceeding to the
            // ray/triangle intersection routine
            if (!__any_sync(FULL_MASK, searchingLeaf))
                break;    // break from while loop and go to code below, processing leaf nodes

        }

        ///////////////////////////////////////////
        /// TRIANGLE INTERSECTION
        //////////////////////////////////////

        // Process postponed leaf nodes.
        while (leafAddr < 0) { /// if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode)
            // Intersect the ray against each triangle using Sven Woop's algorithm.
            // Woop ray triangle intersection: Woop triangles are unit triangles. Each ray
            // must be transformed to "unit triangle space", before testing for intersection

            // triAddr is index in triWoop array (and bitwise complement of leafAddr)
            for (int triAddr = ~leafAddr;; triAddr += 3) { // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered
                // Read first 16 bytes of the triangle.
                // fetch first precomputed triangle edge
                float4 v00 = bvh->triangles[triAddr];

                // End marker 0x80000000 (negative zero) => all triangles in leaf processed --> terminate
                if (__float_as_int(v00.x) == 0x80000000)
                    break;

                // Compute and check intersection t-value (hit distance along ray).
                float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;   // Origin z
                float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);  // inverse Direction z
                float t = Oz * invDz;

                if (t > tmin && t < hitT) {
                    // Compute and check barycentric u.
                    // fetch second precomputed triangle edge
                    float4 v11 = bvh->triangles[triAddr + 1];
                    float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;  // Origin.x
                    float Dx = dirx * v11.x + diry * v11.y + dirz * v11.z;  // Direction.x
                    float u = Ox + t * Dx; /// parametric equation of a ray (intersection point)

                    if (u >= 0.0f && u <= 1.0f) {
                        // Compute and check barycentric v.
                        // fetch third precomputed triangle edge
                        float4 v22 = bvh->triangles[triAddr + 2];
                        float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f) {
                            // We've got a hit!
                            // Record intersection.
                            hitT = t;
                            hitIndex = triAddr; // store triangle index for shading

//                            // Closest intersection not required => terminate.
//                            if (anyHit)  // only true for shadow rays
//                            {
//                                nodeAddr = EntrypointSentinel;
//                                break;
//                            }

                            // compute normal vector by taking the cross product of two edge vectors
                            // because of Woop transformation, only one set of vectors works
                            hitNormal = cross(make_float3(v11.x, v11.y, v11.z), make_float3(v22.x, v22.y, v22.z));
                        }
                    }
                }
            } // end triangle intersection

            // Another leaf was postponed => process it as well.
            leafAddr = nodeAddr;
            if (nodeAddr < 0)    // nodeAddr is an actual leaf when < 0
            {
                nodeAddr = *stackPtr;  // pop stack
                stackPtr--;
            }
        } // end leaf/triangle intersection loop
    } // end traversal loop (AABB and triangle intersection)

    // Remap intersected triangle index, and store the result.
    if (hitIndex != -1) {
        intersect.t = hitT;
        intersect.position = r.at(hitT);
        intersect.set_normal(r, normalize(hitNormal));
        return true;
    } else {
        return false;
    }
}

__device__ float3 path_trace(Ray r, curandState &rand_state, EnvironmentMap *envmap, BVH *bvh)
{
    Material plane_mat(LAMBERTIAN, make_float3(0.65, 0.1, 0.1), 0);
    Plane plane(make_float3(0, -0.283, 0), make_float3(0, 1, 0), &plane_mat);

//    Material sphere1_mat(GLASS, make_float3(1), 1.5);
//    Sphere sphere1(make_float3(-2.025, 0.5, 0), 0.5, &sphere1_mat);
//
//    Material sphere2_mat(LAMBERTIAN, make_float3(0.7), 0);
//    Sphere sphere2(make_float3(-0.675, 0.5, 0), 0.5, &sphere2_mat);
//
//    Material sphere3_mat(MIRROR, make_float3(0.5), 0);
//    Sphere sphere3(make_float3(0.675, 0.5, 0), 0.5, &sphere3_mat);
//
//    Material sphere4_mat(GLOSSY, make_float3(0.15, 0.25, 0.4), 0.05);
//    Sphere sphere4(make_float3(2.025, 0.5, 0), 0.5, &sphere4_mat);

    Material bvh_mat(LAMBERTIAN, make_float3(0.65, 0.65, 0.65), 0);
//    Material bvh_mat(GLASS, make_float3(1), 1.5);

    float3 color = make_float3(1);
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        // Intersect scene
        Intersection intersect;
        bool hit0 = plane.intersect(r, intersect);
        bool hit1 = intersect_bvh(bvh, r, intersect);
        if (hit1) {
            intersect.material = &bvh_mat;
        }

//        bool hit1 = sphere1.intersect(r, intersect);
//        bool hit2 = sphere2.intersect(r, intersect);
//        bool hit3 = sphere3.intersect(r, intersect);
//        bool hit4 = sphere4.intersect(r, intersect);
//
//        if (hit0 || hit1 || hit2 || hit3 || hit4) {
        if (hit0 || hit1) {

#ifdef IMPORTANCE_SAMPLING

            float3 attenuation;
            bool importance_sample = false;
            Ray r_new = intersect.material->brdf(r, intersect, attenuation, rand_state, importance_sample);

            if (importance_sample) {
                if (curand_uniform(&rand_state) < 0.5) {
                    r_new.direction = envmap->sample_lights(rand_state);
                }

                float env_pdf = envmap->pdf(r_new.direction);
                float diff_pdf = intersect.material->pdf(r, intersect, r_new);
                float mixed_pdf = (env_pdf + diff_pdf) * 0.5f;
                color *= diff_pdf / mixed_pdf;
            }

            color *= attenuation;
            r = r_new;

#else

            float3 attenuation;
            bool importance_sample = false;
            r = intersect.material->brdf(r, intersect, attenuation, rand_state, importance_sample);
            color *= attenuation;

#endif

        } else {
            return color * envmap->sample(r.direction);
        }
    }

    return make_float3(0);
}

__global__ void render_kernel(BVH *bvh, EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width,
                              size_t height, size_t samples_per_pixel)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = (height - y - 1) * width + x;

    curandState rand_state;
    curand_init(1984 + i, 0, 0, &rand_state);

    float3 accum_color = make_float3(0);
    for (int samp = 0; samp < samples_per_pixel; samp++) {
        float u = (float(x) + curand_uniform(&rand_state) - .5f) / float(width - 1);
        float v = (float(y) + curand_uniform(&rand_state) - .5f) / float(height - 1);

        Ray ray = camera->cast_ray(u, v, rand_state);
        accum_color += path_trace(ray, rand_state, envmap, bvh);
    }

    image_data[i] = accum_color / samples_per_pixel;
}

__global__ void intersect_scene(BVH *bvh, PathData *paths)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_ACTIVE)) return;

    // Load ray data
    const uint path_index = index;
    float4 o = paths->origin[path_index];
    float4 d = paths->direction[path_index];
    float3 n;
    float t_min  = o.w;
    float t_max  = d.w;

    // Intersect planes
    const float3 normal = make_float3(0, 1, 0);
    const float3 position = make_float3(0, -0.283, 0);

    float det = dot(make_float3(d), normal);
    if (det > EPSILON || det < -EPSILON) {
        auto t = dot(position - make_float3(o), normal) / det;
        if (t > t_min && t < t_max) {
            t_max = t;
            n = normal;
        }
    }

    // Traversal stack in CUDA thread-local memory.
    int traversalStack[STACK_SIZE];

    // Live state during traversal, stored in registers.
    int stackPtr = 0;
    int node_addr = 0;
    float3 idir = 1.0f / make_float3(d);
    float3 ood  = idir * make_float3(o);

    // Traversal loop.
    while(stackPtr >= 0) {
        while (node_addr >= 0 && stackPtr >= 0) {
            // Fetch AABBs of the two child nodes.
            const float4 n0xy = tex1Dfetch<float4>(bvh->nodes_texture, node_addr * 4 + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            const float4 n1xy = tex1Dfetch<float4>(bvh->nodes_texture, node_addr * 4 + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            const float4 nz   = tex1Dfetch<float4>(bvh->nodes_texture, node_addr * 4 + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            const float4 tmp  = tex1Dfetch<float4>(bvh->nodes_texture, node_addr * 4 + 3); // child_index0, child_index1
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
            const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, t_min);
            const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, t_max);
            const float c1lox = n1xy.x * idir.x - ood.x;
            const float c1hix = n1xy.y * idir.x - ood.x;
            const float c1loy = n1xy.z * idir.y - ood.y;
            const float c1hiy = n1xy.w * idir.y - ood.y;
            const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, t_min);
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
                const float4 v00 = bvh->triangles[tri_addr + 0];
                const float4 v11 = bvh->triangles[tri_addr + 1];
                const float4 v22 = bvh->triangles[tri_addr + 2];

                // End marker (negative zero) => all triangles processed.
                if (__float_as_int(v00.x) == 0x80000000) {
                    break;
                }

                // Woop triangle intersection
                float Oz = v00.w - o.x*v00.x - o.y*v00.y - o.z*v00.z;
                float invDz = 1.0f / (d.x*v00.x + d.y*v00.y + d.z*v00.z);
                float t = Oz * invDz;

                if (t > t_min && t < t_max) {
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
                        }
                    }
                }
            } // triangle

            node_addr = traversalStack[stackPtr--];
        } // leaf
    } // traversal

    paths->direction[path_index].w = t_max;
    paths->normal[path_index] = make_float4(normalize(n));
}

__device__ __inline__ float3 diffuse(const float3 &n, curandState &rand_state) {
    // randomly generate point in sphere
    float z = curand_uniform(&rand_state) * 2.0f - 1.0f;
    float a = curand_uniform(&rand_state) * 2.0f * PI;
    float r = sqrtf(1.0f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);
    float3 dir = make_float3(x, y, z) + n;

    if (dir.x < EPSILON && dir.y < EPSILON && dir.z < EPSILON)
        return n;

    return normalize(dir);
}

__device__ bool g_is_working = false;

__global__ void logic_kernel(PathData *paths, EnvironmentMap *envmap, float3 *image_data, uint samples_per_pixel)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_ACTIVE)) return;

    paths->flags[index] = 0;
    g_is_working = true;

    // TERMINATE PATH
    if (paths->direction[index].w == FLT_MAX || paths->depth[index]++ >= MAX_DEPTH) {
        uint pixel_index = paths->pixel_index[index];
        float3 color = make_float3(paths->throughput[index]) * envmap->sample(make_float3(paths->direction[index])) / samples_per_pixel;
        atomicAdd(&image_data[pixel_index].x, color.x);
        atomicAdd(&image_data[pixel_index].y, color.y);
        atomicAdd(&image_data[pixel_index].z, color.z);

        paths->set_flag(index, IS_NEW_PATH);
    } else {
        // TODO: check actual material
        paths->set_flag(index, IS_DIFFUSE);
    }
}

__device__ uint g_path_count = 0;

__global__ void generate_primary_paths(Camera *camera, uint width, uint height, uint samples_per_pixel, uint path_count,
                                       PathData *paths, int seed, bool override)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
//    if (index >= MAX_PATHS || !paths->get_flag(index, IS_NEW_PATH)) return;

    if (index >= MAX_PATHS || (!override && !paths->get_flag(index, IS_NEW_PATH))) return;

    const uint global_index = atomicAdd(&g_path_count, 1) / samples_per_pixel;
    if (global_index >= width * height) return;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);

    const uint x = (global_index) % width;
    const uint y = (global_index / width) % height;

    const float u = (float(x) + curand_uniform(&rand_state) - .5f) / float(width - 1);
    const float v = (float(y) + curand_uniform(&rand_state) - .5f) / float(height - 1);

    Ray r = camera->cast_ray(u, v, rand_state);
    paths->pixel_index[index] = (height - y - 1) * width + x;
    paths->origin[index] = make_float4(r.origin, EPSILON);
    paths->direction[index] = make_float4(r.direction, FLT_MAX);
    paths->throughput[index] = make_float4(1);
    paths->depth[index] = 0;
    paths->set_flag(index, IS_ACTIVE);
}

__global__ void generate_diffuse_paths(EnvironmentMap *envmap, PathData *paths, int seed)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_DIFFUSE)) return;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);

    float3 norm = make_float3(paths->normal[index]);
    float3 out_dir = (curand_uniform(&rand_state) < 0.5) ? diffuse(norm, rand_state) : envmap->sample_lights(rand_state);

    float4 dir = paths->direction[index];
    dir *= dir.w;
    dir.w = 0;

    paths->origin[index] += dir;
    paths->direction[index] = make_float4(out_dir, FLT_MAX);

    float env_pdf = envmap->pdf(out_dir);
    float diff_pdf = max(dot(norm, out_dir) / PI, 0.0f);
    float mixed_pdf = (env_pdf + diff_pdf) * 0.5f;
    paths->throughput[index] *= make_float4(0.65f * diff_pdf / mixed_pdf);
    paths->set_flag(index, IS_ACTIVE);
}


__host__ void launch_render_kernel(BVH *bvh, EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width,
                                   size_t height, size_t samples_per_pixel, dim3 grid, dim3 block, PathData *paths)
{
//    render_kernel <<< grid, block >>>(bvh, envmap, camera, image_data, width, height, samples_per_pixel);
//    return;

    const uint block_size = 64 * 2;
    const uint grid_size = (MAX_PATHS + block_size - 1) / block_size;

    uint path_count = 0;
    int i = 1021;

    bool is_working = false;
    bool override = true;
    do {
        generate_primary_paths<<<grid_size, block_size>>>(camera, width, height, samples_per_pixel,
                                                          path_count, paths, i++, override);

        generate_diffuse_paths<<<grid_size, block_size>>>(envmap, paths, rand());

        intersect_scene<<<grid_size, block_size>>>(bvh, paths);

        logic_kernel<<<grid_size, block_size>>>(paths, envmap, image_data, samples_per_pixel);
        override = false;

        bool temp = false;
        gpuErrchk(cudaMemcpyFromSymbol(&is_working, g_is_working, sizeof(bool)));
        gpuErrchk(cudaMemcpyToSymbol(g_is_working, &temp, sizeof(bool)));
    } while (is_working);
}