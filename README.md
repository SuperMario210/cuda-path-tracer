# CUDA Path Tracer

## Project Description

This program is a path tracer that runs on the GPU using CUDA.  Given an input scene file, resolution, and sample count, the program renders the scene to an output PNG file.  

The program supports the following four materials:
* Lambertian: a perfectly diffuse material that scatters rays
* Mirror: a perfectly reflective material that reflects rays
* Glass: a perfectly dielectric material that refracts rays
* Glossy: a blend between the Lambertian and Mirror materials that results in a shiny looking surface

The program also supports the use of HDR environment maps for lighting.  These background maps are importance sampled for faster convergence using less rays.

Finally, the program supports three different types of object primitives: spheres, planes and triangles.  When rendering triangles, a BVH is built on the CPU using a binned SAH algorithm.  As a result, the program can handle large amounts of triangles (dragon model contains ~800k triangles) without issue. 

## Build Instructions

Use the included `CMakeLists.txt` file to build the project.  Use the following commands:
```
mkdir bin
cd bin
cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build .
```
The `cuda-path-tracer` executable should now be built in the `bin` directory.

## How to run

You need to run the executable from within the `bin` directory in order for all the file paths to work correctly.  The executable takes the following arguments:

`./cuda-path-tracer scene_file x_res y_res samples_per_pixel image_file`
* `scene_file`: the path to the scene file to render
* `x_res`: the x resolution of the resulting render (1920 is recommended)
* `y_res`: the y resolution of the resulting render (1080 is recommended)
* `samples_per_pixel`: the number of rays per pixel (512 gives nice results)
* `image_file`: the path to the file in which to store the resulting render

Some commands to copy and paste:

Balls scene (~):
```
./cuda-path-tracer ../scene/balls.scene 1920 1080 512 ../img/balls_new
```

City scene (~):
```
./cuda-path-tracer ../scene/city.scene 1920 1080 512 ../img/city_new
```

Dragon scene (~):
```
./cuda-path-tracer ../scene/dragon.scene 1920 1080 512 ../img/dragon_new
```

## Sample Renders
![](img/balls.png)
1920 x 1080p, 2048 samples per pixel

![](img/dragon.png)
1920 x 1080p, 2048 samples per pixel

![](img/city.png)
1920 x 1080p, 2048 samples per pixel

![](img/suzanne.png)
1920 x 1080p, 2048 samples per pixel

![](img/scream.png)
1920 x 1080p, 2048 samples per pixel

## Performance Analysis

The GPU version of the path tracer provides a huge performance increase over the CPU version.  The following tests were run on `titan.caltech.edu` using the build specifications described above.   

For simple scenes such as the balls scene, at a 960 x 540p resolution with 32 samples per pixel, the CPU version rendered the final image in 11461 ms while the GPU version rendered the exact same image in 79 ms for a 145x increase in performance. 

For more complex scenes such as the dragon scene, this improvement becomes even more pronouced.  At a 960 x 540p resolution with 32 samples per pixel, the CPU version rendered the final dragon image in 64612 ms while the GPU version rendered the exact same image in 244 ms for a 265x increase in performance. 

At higher resolutions and sample counts, the GPU's advantage becomes even more pronounced.

One thing that could still be improved when it comes to performance is combining the different material kernels into a single kernel.  To decrease warp divergence and register usage, I completely restructured the path tracer to use the "wavefront formulation" described in [this paper](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf).  In the linked paper, the authors use a separate kernel for each type of material in the scene.  However, this only really provides a benefit when rendering extremely complex materials.  In my simple scenes, it would probably be better to just use a single kernel for all materials despite the small amount of divergence that would occur.

Another small performance optimization could be using warp-aggregated atomic operations.  Currently, atomic additions are used in several places in the renderer in order to accumulate output colors and increment index counters.  By using warp level primitives to aggregate these increments and only do a single addition per warp, a small performance improvement could be achieved.

Finally, using higher quality BVH construction algorithms could also provide a good performance improvement.  Something like [NVIDIA's SBVH](and using warp-aggregated atomic operations) could result in much higher quality BVH's that require less intersection tests and as a result, give faster render times.