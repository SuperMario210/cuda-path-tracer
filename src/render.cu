#include "camera.cuh"
#include "render.cuh"

#define IMPORTANCE_SAMPLING
#define RUSSIAN_ROULETTE
#define MIN_DEPTH               3
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

__global__ void logic_kernel(PathData *paths, EnvironmentMap *envmap, float3 *image_data, uint samples_per_pixel, int seed)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_ACTIVE)) return;

    paths->flags[index] = 0;
    g_is_working = true;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);


    // Background intersection
    if (paths->direction[index].w == FLT_MAX) {
        uint pixel_index = paths->pixel_index[index];
        float3 color = make_float3(paths->throughput[index]) * envmap->sample(make_float3(paths->direction[index])) / samples_per_pixel;
        atomicAdd(&image_data[pixel_index].x, color.x);
        atomicAdd(&image_data[pixel_index].y, color.y);
        atomicAdd(&image_data[pixel_index].z, color.z);

        paths->set_flag(index, IS_NEW_PATH);
        return;
    }

    // Material Intersection


#ifdef RUSSIAN_ROULETTE

    if (paths->depth[index]++ >= MIN_DEPTH) {
        float4 throughput = paths->throughput[index];
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (curand_uniform(&rand_state) > p) {
            paths->set_flag(index, IS_NEW_PATH);
            return;
        }
        paths->throughput[index] *= 1 / p;
    }

#endif

    paths->set_flag(index, IS_DIFFUSE);
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

        logic_kernel<<<grid_size, block_size>>>(paths, envmap, image_data, samples_per_pixel, rand());
        override = false;

        bool temp = false;
        gpuErrchk(cudaMemcpyFromSymbol(&is_working, g_is_working, sizeof(bool)));
        gpuErrchk(cudaMemcpyToSymbol(g_is_working, &temp, sizeof(bool)));
    } while (is_working);
}