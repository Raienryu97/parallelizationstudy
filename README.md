# parallelizationstudy

A performance study of parallelisation on the Lucas-Kanade Optical Flow algorithm.

# Setup

Install openCV as per the official method. Only C/C++ installation is sufficient.

### Modification in OpenCV
* run `cmake-gui` in the `opencv/build` directory and remove all parallel-based optimisations as listed in [pre-condition section](https://docs.opencv.org/trunk/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html)
* Disable `BUILD_SHARED_LIBS` so that profiling with gprof works
* Click on `configure` and then `generate`
* Run `make -jX` where X is twice the number of your CPU cores ( `make -j8` for a quad core processor)
* Run `sudo make install`

### Clone this project
You're done setting up after this step.

# Usage
* To view the working of optical flow
  * `./automate.sh`

# Additional Notes
* Any video can be extracted into a series of frames by using ffmpeg
* The command `ffmpeg -i videoplayback.webm -r 10 "inputs/%d.jpg"` should be run from the root of the repository
    * The name of the input video file should be given after the `-i` flag
    * The fps (Frames Per Second) rate at which the video should be extracted should be given after the `-r` flag
    * The number of images present in inputs folder should be reflected in the `NUM_INPUTS` definition in the opticalflow.cpp file
