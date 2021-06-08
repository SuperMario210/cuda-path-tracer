# CUDA Path Tracer

## Build Instructions

Use the included `CMakeLists.txt` file to build the project.  Use the following commands:
```
mkdir bin
cd bin
cmake ../
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
1920 x 1080p, 512 samples per pixel

![](img/city.png)
1920 x 1080p, 512 samples per pixel

![](img/dragon.png)
1920 x 1080p, 512 samples per pixel