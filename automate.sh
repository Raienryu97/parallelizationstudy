#!/bin/bash

projectRoot=`pwd`;
executablesDir=`pwd`/Executables;

MATRIX_SIZE=256;

num_loops=10000;
num_data=256;


CYAN='\e[0;36m'
NOCOLOR="\033[0m"

rm -f ${executablesDir}/*;

# Compiling Blocked Matrix Multiplication
echo ""
echo -e "${CYAN} Compiling Executables For Blocked Matrix Multiplication of Matrix Size  ${MATRIX_SIZE} X ${MATRIX_SIZE}${NOCOLOR}"
echo ""

# Base Variant
cd $projectRoot/BMM/base
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" MatrixMultiplication.c
if cc MatrixMultiplication.c -std=c99 -o $executablesDir/bmm_singlecore; then
	echo "Compiled Single Core Variant of Blocked Matrix Multiplication"
else 
	echo "Error compiling Single Core Variant of Blocked Matrix Multiplication"
	exit
fi

# OpenMP (Multicore) Variant
cd $projectRoot/BMM/omp
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" MatrixMultiplication.c
if cc MatrixMultiplication.c -std=c99 -fopenmp -o $executablesDir/bmm_multicore; then
	echo "Compiled Multi Core Variant of Blocked Matrix Multiplication"
else
	echo "Error compiling Multi Core Variant of Blocked Matrix Multiplication"
	exit
fi

# OpenACC Variant
cd $projectRoot/BMM/oacc
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" MatrixMultiplication.c
if pgcc MatrixMultiplication.c -ta=tesla:cc50 -o $executablesDir/bmm_oacc; then
	echo "Compiled OpenACC Variant of Blocked Matrix Multiplication"
else
	echo "Error compiling OpenACC Variant of Blocked Matrix Multiplication"
	exit
fi

# OpenCL Variant
cd $projectRoot/BMM/ocl
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" MatrixMultiplication.cpp
cmake . &>/dev/null
if make -j8 &>/dev/null; then
	echo "Compiled OpenCL Variant of Blocked Matrix Multiplication"
	mv bmm_ocl $executablesDir;
	cp MatMul.cl $executablesDir;
else
	echo "Error compiling OpenCL Variant of Blocked Matrix Multiplication"
	exit
fi

# CUDA Variant
cd $projectRoot/BMM/cuda
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" MatrixMultiplication.cu
if nvcc MatrixMultiplication.cu -o $executablesDir/bmm_cuda; then
	echo "Compiled CUDA Variant of Blocked Matrix Multiplication"
else
	echo "Error compiling CUDA Variant of Blocked Matrix Multiplication"
	exit
fi

# Execute Blocked Matrix Multiplication
echo ""
echo -e "${CYAN} Running Executables For Blocked Matrix Multiplication${NOCOLOR}"
echo ""
cd $executablesDir
./bmm_singlecore
./bmm_multicore
./bmm_oacc
./bmm_ocl
./bmm_cuda
echo ""

# Compiling Discrete Cosine Transform
echo ""
echo -e "${CYAN} Compiling Executables For Discrete Cosine Transform for Image Size ${MATRIX_SIZE} X ${MATRIX_SIZE}${NOCOLOR}"
echo ""

# Base Variant
cd $projectRoot/DCT/base
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" DCT.cpp
if g++ DCT.cpp -std=gnu++0x -o $executablesDir/dct_singlecore; then
	echo "Compiled Single Core Variant of Discrete Cosine Transform"
else 
	echo "Error compiling Single Core Variant of Discrete Cosine Transform"
	exit
fi

# OpenMP (Multicore) Variant
cd $projectRoot/DCT/omp
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" DCT.cpp
if g++ DCT.cpp -std=gnu++0x -fopenmp -o $executablesDir/dct_multicore; then
	echo "Compiled Multi Core Variant of Discrete Cosine Transform"
else
	echo "Error compiling Multi Core Variant of Discrete Cosine Transform"
	exit
fi

# OpenACC Variant
cd $projectRoot/DCT/oacc
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" DCT.c
if pgcc DCT.c -ta=tesla:cc50 -o $executablesDir/dct_oacc; then
	echo "Compiled OpenACC Variant of Discrete Cosine Transform"
else
	echo "Error compiling OpenACC Variant of Discrete Cosine Transform"
	exit
fi

# OpenCL Variant
cd $projectRoot/DCT/ocl
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" DCT.cpp
cmake . &>/dev/null
if make -j8 &>/dev/null; then
	echo "Compiled OpenCL Variant of Blocked Matrix Multiplication"
	mv dct_ocl $executablesDir;
	cp DCT.cl $executablesDir;
else
	echo "Error compiling OpenCL Variant of Blocked Matrix Multiplication"
	exit
fi

# CUDA Variant
cd $projectRoot/DCT/cuda
sed -i "/#define MATRIX_SIZE/c\#define MATRIX_SIZE ${MATRIX_SIZE}" DCT.cu
if nvcc -std=c++11  DCT.cu -o $executablesDir/dct_cuda; then
	echo "Compiled CUDA Variant of Discrete Cosine Transform"
else
	echo "Error compiling CUDA Variant of Discrete Cosine Transform"
	exit
fi

# Execute Blocked Matrix Multiplication
echo ""
echo -e "${CYAN} Running Executables For Discrete Cosine Transform${NOCOLOR}"
echo ""
cd $executablesDir
./dct_singlecore
./dct_multicore
./dct_oacc
./dct_ocl
./dct_cuda
echo ""

# Compiling kmeans
echo ""
echo -e "${CYAN} Compiling Executables For KMeans Program of Data Size $num_data ${NOCOLOR}"
echo ""

# Base Variant
cd $projectRoot/kmeans/base
sed -i "/#define num_loops/c\#define num_loops ${num_loops}" base_kmeans.cpp
sed -i "/#define num_data/c\#define num_data ${num_data}" base_kmeans.cpp
if g++ base_kmeans.cpp -o $executablesDir/kmeans_singlecore; then
	echo "Compiled Single Core Variant of KMeans"
else 
	echo "Error compiling Single Core Variant of KMeans"
	exit
fi

# OpenMP (Multicore) Variant
cd $projectRoot/kmeans/omp
sed -i "/#define num_loops/c\#define num_loops ${num_loops}" omp_kmeans.cpp
sed -i "/#define num_data/c\#define num_data ${num_data}" omp_kmeans.cpp
if g++ omp_kmeans.cpp -fopenmp -o $executablesDir/kmeans_multicore; then
	echo "Compiled Multi Core Variant of KMeans"
else
	echo "Error compiling Multi Core Variant of KMeans"
	exit
fi

# OpenACC Variant
cd $projectRoot/kmeans/oacc
sed -i "/#define num_loops/c\#define num_loops ${num_loops}" oacc_kmeans.cpp
sed -i "/#define num_data/c\#define num_data ${num_data}" oacc_kmeans.cpp
if pgc++ oacc_kmeans.cpp -ta=tesla:cc50 -o $executablesDir/kmeans_oacc; then
	echo "Compiled OpenACC Variant of KMeans"
else
	echo "Error compiling OpenACC Variant of KMeans"
	exit
fi

# OpenCL Variant
cd $projectRoot/kmeans/ocl
sed -i "/#define num_loops/c\#define num_loops ${num_loops}" ocl_kmeans.cpp
sed -i "/#define num_data/c\#define num_data ${num_data}" ocl_kmeans.cpp
cmake . &>/dev/null
if make -j8 &>/dev/null; then
	echo "Compiled OpenCL Variant of KMeans"
	mv kmeans_ocl $executablesDir;
	cp kmeans.cl $executablesDir;
else
	echo "Error compiling OpenCL Variant of Kmeans"
	exit
fi

# CUDA Variant
cd $projectRoot/kmeans/cuda
sed -i "/#define num_loops/c\#define num_loops ${num_loops}" kmeans_cuda.cu
sed -i "/#define num_data/c\#define num_data ${num_data}" kmeans_cuda.cu
if nvcc kmeans_cuda.cu -o $executablesDir/kmeans_cuda; then
	echo "Compiled CUDA Variant of KMeans"
else
	echo "Error compiling CUDA Variant of KMeans"
	exit
fi


# Execute KMeans
echo ""
echo -e "${CYAN} Running Executables For KMeans Program${NOCOLOR}"
echo ""
cd $executablesDir
./kmeans_singlecore
./kmeans_multicore
./kmeans_oacc
./kmeans_ocl
./kmeans_cuda
echo ""