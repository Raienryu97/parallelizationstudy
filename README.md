# parallelizationstudy

A performance study of parallelisation on
* Block Matrix Multiplication by [Evan Purkhiser](https://github.com/EvanPurkhiser/CS-Matrix-Multiplication)
* Discrete Cosine Transform used in JPEG Image Compression
* k-means clustering algorithm

OpenMP, OpenACC, OpenCL and CUDA are the parallelisation tools that are used in this project.

If you're interested in the results obtained, skip directly to the results section at the end of this readme.

# Setup

Make sure you have gcc 4.8 or above and have cmake installed.

### Hardware Dependencies
This project will work completely only if you have an NVIDIA GPU since OpenACC and CUDA do not work on others. 

In case you have a machine with a non NVIDIA GPU, then you can comment out the compilation and execution of CUDA and OpenACC variants for the benchmarks in the [automate.sh](automate.sh) file to get partial execution.

### Install PGI Compiler
This is for OpenACC. You can also use [other compilers](https://www.openacc.org/tools) for OpenACC (if you are sure it would work), but we have used the PGI Compiler for our testing. You can get the latest community edition of PGI Compiler for free at their [downloads](https://www.pgroup.com/products/community.htm) page.
We have tested the project on the 17.10 release.

### Install OpenCL
There are two steps involved in OpenCL installation
* Install OpenCL drivers for your hardware accelerator (usually bundled in the driver package for the GPU)
* Download and install OpenCL sdk following proper instructions specific to your platform

You can check for proper OpenCL installation by running the command `clinfo` in the terminal. 

If it fails with a segmentation fault, then your installation may be faulty.

### Install CUDA 
* Install the latest display driver from NVIDIA's website for your GPU
* Download cuda-toolkit from NVIDIA's website and install it. In this step,
you can skip installation of the display driver if it asks since you have already
done it in the previous step.

While installing the display driver, you might have to disable the open source display
driver (nouveau). Disabling nouveau differs based on the Linux distribution you
are using.

### Get the project files
Either download the project as a zip or clone it to get the files locally.

You're done setting up after this step.

# Usage

Make sure that the [automate.sh](automate.sh) file has permissions to be run as an executable and then run it.

```bash
# From root of the cloned repository
./automate.sh
```
You can change the size of the data set by modifying the variables in the [automate.sh](automate.sh) file. 

The following is an example to convert Matrix Size for Blocked Matrix Multiplication and DCT from 256 to 2048.

Old Code :
```bash
#!/bin/bash

projectRoot=`pwd`;
executablesDir=`pwd`/Executables;

MATRIX_SIZE=256;

num_loops=10000;
num_data=256;
```

Replace it as :
```bash
#!/bin/bash

projectRoot=`pwd`;
executablesDir=`pwd`/Executables;

MATRIX_SIZE=2048;

num_loops=10000;
num_data=256;
```

# Results
The following results were observed on a machine with
* Intel Xeon E3-1241 CPU
* NVIDIA Quadro K620 GPU

Speedup here is defined as the ratio of execution time of base variant to the execution time of accelerated variant.
* A speedup of greater than 1 indicates that the accelerated variant is faster than the base code
* A speedup of 1 indicates that the accelerated variant runs in the same time as the base code
* A speedup of lesser than 1 indicates that the accelerated variant is slower than the base code

### Overall Speedup
|Parallelization Technique|Block Matrix Multiplication|Discrete Cosine Transform|k-means clustering algorithm|
| ------------- | ------------- | ------------- | ------------- |
| OpenMP        | 5.454         | 3.491         | 1.376         |
| OpenACC       | 5.388         | 0.068         | 5.919         |
| OpenCL        | 246.383       | 8.286         | 6.014         |
| CUDA          | 309.763       | 10.053        | 11.136        |

The following three tables have the execution times (in seconds) for each of the benchmark on all of the parellilsation tools/APIs used in this project.

### Block Matrix Multiplication Execution Times in seconds (ET)
|Matrix Size  |Single Core ET|OMP ET       |OACC ET |OCL ET           |CUDA ET        |
|-------------|--------------|-------------|--------|-----------------|---------------|
| 256X256     | 0.079        | 0.014       | 0.111  | 0.001           | 0.001         |
| 512X512     | 0.664        | 0.117       | 0.285  | 0.004           | 0.003         |
| 1024X1024   | 5.472        | 1.03        | 1.254  | 0.024           | 0.020         |
| 2048X2048   | 44.607       | 8.384       | 9.234  | 0.187           | 0.151         |
| 4096X4096   | 363.662      | 66.677      | 67.496 | 1.476           | 1.174         |

### Discrete Cosine Transform Execution Times in seconds (ET)
|Image Size   |Single Core ET|OMP ET       |OACC ET |OCL ET           |CUDA ET        |
|-------------|--------------|-------------|--------|-----------------|---------------|
| 256X256     | 0.003        | 0.002       | 0.119  | 0.001           | 0.0005        |
| 512X512     | 0.012        | 0.004       | 0.245  | 0.003           | 0.002         |
| 1024X1024   | 0.051        | 0.015       | 0.766  | 0.007           | 0.006         |
| 2048X2048   | 0.186        | 0.056       | 2.84   | 0.024           | 0.019         |
| 4096X4096   | 0.754        | 0.216       | 11.168 | 0.091           | 0.075         |

### k-means Clustering Algorithm Execution Times in seconds (ET)
|Data Size    |Single Core ET|OMP ET       |OACC ET |OCL ET           |CUDA ET        |
|-------------|--------------|-------------|--------|-----------------|---------------|
| 1024        | 0.92         | 1.277       | 0.464  | 0.332           | 0.224         |
| 2048        | 1.937        | 1.447       | 0.625  | 0.431           | 0.289         |
| 4096        | 3.965        | 2.093       | 0.877  | 0.682           | 0.507         |
| 8192        | 7.903        | 5.632       | 1.462  | 1.332           | 0.802         |
| 16384       | 15.679       | 11.397      | 2.649  | 2.607           | 1.408         |
