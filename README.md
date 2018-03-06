# parallelizationstudy

A performance study of parallelisation on
* Block Matrix Multiplication by [Evan Purkhiser](https://github.com/EvanPurkhiser/CS-Matrix-Multiplication)
* Lucas-Kanade Optical Flow algorithm using OpenCV

# Setup

Install openCV as per the official method. Only C/C++ installation is sufficient.

### Modification in OpenCV
* run `cmake-gui` in the `opencv/build` directory and remove all parallel-based optimisations as listed in [pre-condition section](https://docs.opencv.org/trunk/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html)
* Disable `BUILD_SHARED_LIBS` so that profiling with gprof works
* Click on `configure` and then `generate`
* Run `make -jX` where X is twice the number of your CPU cores ( `make -j8` for a quad core processor)
* Run `sudo make install`

### Installation Of ffmpeg
`sudo apt-get install ffmpeg`

### Clone this project
You're done setting up after this step.

# Usage

### Block Matrix Multiplication

```bash
# From root of the cloned repository
cd BMM/

# Run the script that automates compilation and execution
./automate.sh
```

### Lukas Kanade Optical Flow
* Run directly as shown below for single core execution
* For multi-core execution, first add the `-fopenmp` flag to line #5 @ CMakeLists.txt and then follow instructions as shown below

```bash
# From root of the cloned repository
cd OpticalFlow/

# For Processing over Full HD images
# ffmpeg -i videoplayback.webm -r 10 "inputs/%d.jpg"

# For Processing over 4K images
ffmpeg -i videoplayback1.webm -r 10 "inputs/%d.jpg"

# Run the script that automates compilation and execution
./automate.sh
```

# Results

### Overall Speedup
|Parallelization Technique|Lukas Kanade Optical Flow|Block Matrix Multiplication    |
| ----------------------- |-------------------------| ------------------------------|
| OpenMP                  | 4.5 (500 4K Images)     | 5.3 (Matrix Size 4096X4096)   |
| OpenACC                 | Not Applicable          | 13.2 (Matrix Size 4096X4096)  |

### Block Matrix Multiplication Execution Times in seconds (ET) and Speedups
|Matrix Size  |Single Core ET|Multi Core ET|GPU ET |OpenMP Speedup|OpenACC Speedup|
|-------------|--------------|-------------|-------|--------------|---------------|
| 512X512     | 0.812        | 0.151       | 0.231 |5.4           |3.5            |
| 1024X1024   | 7.1855       | 1.300       | 1.140 |5.5           |6.3            |
| 2048X2048   | 56.094       | 11.095      | 8.827 |5.1           |6.4            |
| 4096X4096   | 861.111      | 163.065     | 65.317|5.3           |13.2           |

# Additional Notes
* Any video can be extracted into a series of frames by using ffmpeg
* The command `ffmpeg -i videoplayback.webm -r 10 "inputs/%d.jpg"` should be run from the root of the repository
    * The name of the input video file should be given after the `-i` flag
    * The fps (Frames Per Second) rate at which the video should be extracted should be given after the `-r` flag
    * The number of images present in inputs folder should be reflected in the `NUM_INPUTS` definition in the opticalflow.cpp file
