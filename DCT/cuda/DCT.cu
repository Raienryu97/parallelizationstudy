#include <iostream>
#include <stdio.h>
#include <random>
#include <sys/time.h>
#include "DCT.h"

using namespace std;

#define MATRIX_SIZE 256
#define BLOCK_SIZE 8

__global__ void DCT1(float * inputImage, float * temp, float * dct8x8Mat, int matrixSize){
	
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;

	float sum = 0.0f;

	int x = i % BLOCK_SIZE;
	//int y = j % BLOCK_SIZE;

	for (int k = 0;k < BLOCK_SIZE;k++) {
		sum += dct8x8Mat[x*BLOCK_SIZE + k] * inputImage[k*matrixSize + j];
	}

	temp[i*matrixSize + j] = sum;
}

__global__ void DCT2(float * temp, float * dctCoeffMatrix, float * dct8x8TMat, int matrixSize){

	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	float sum = 0.0f;

	//int x = i % BLOCK_SIZE;
	int y = j % BLOCK_SIZE;

	for (int k = 0;k < BLOCK_SIZE;k++) {
		sum += temp[i*matrixSize + k] * dct8x8TMat[k*BLOCK_SIZE + y];
	}

	dctCoeffMatrix[i*matrixSize + j] = sum;
}

int main(){

	float *d_img, *d_temp, *d_dctCoeffMatrix, *d_dct8x8Mat, *d_dct8x8TMat;
	struct timeval start,end;
	double elapsedTime;

	const int32_t matrixSize = MATRIX_SIZE;
	mt19937 rng(time(NULL));
	uniform_int_distribution<int> gen(0, 255);

    float *h_img = (float *)malloc(matrixSize * matrixSize * sizeof(float));
    float *h_temp = (float *)malloc(matrixSize * matrixSize * sizeof(float));
    float *h_dctCoeffMatrix = (float *)malloc(matrixSize * matrixSize * sizeof(float));

	for(int i=0;i<matrixSize;i++){
		for(int j=0;j<matrixSize;j++){
			*(h_img + i * matrixSize + j) = static_cast<float>(gen(rng));
			*(h_img + i * matrixSize + j) = *(h_img + i * matrixSize + j) - 128;
		}
	}

	cudaMalloc(&d_img, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
	cudaMalloc(&d_temp, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
	cudaMalloc(&d_dctCoeffMatrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
	cudaMalloc(&d_dct8x8Mat, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	cudaMalloc(&d_dct8x8TMat, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

	// Keep track of when we start doing work
    gettimeofday(&start, NULL);

	cudaMemcpy(d_img, h_img, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dct8x8Mat, dct8x8Matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dct8x8TMat, dct8x8MatrixTranspose, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE/threads.x,MATRIX_SIZE/threads.y);

    DCT1<<<grid,threads>>>(d_img, d_temp, d_dct8x8Mat, MATRIX_SIZE);
	cudaDeviceSynchronize();


    DCT2<<<grid,threads>>>(d_temp, d_dctCoeffMatrix, d_dct8x8TMat, MATRIX_SIZE);
	cudaDeviceSynchronize();

	cudaMemcpy(h_dctCoeffMatrix, d_dctCoeffMatrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Keep track of when we finish our work
    gettimeofday(&end, NULL);
    // Calculate the time it took to do the above task
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    elapsedTime /= 1000;
    if(elapsedTime >= 0.001)
    	printf("CUDA            : %.3f seconds\n",elapsedTime);
    else
    	printf("CUDA            : %.4f seconds\n",elapsedTime);

    cudaFree(d_img);
    cudaFree(d_temp);
    cudaFree(d_dct8x8Mat);
    cudaFree(d_dct8x8TMat);
    cudaFree(d_dctCoeffMatrix);

	return 0;
}