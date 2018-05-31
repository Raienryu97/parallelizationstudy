#include <iostream>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define MATRIX_SIZE 256
#define BLOCK_SIZE  16

using namespace std;

__global__ void matMul(float *x, float *y, float *z, int matrixSize){

	float zTemp = 0.0f;

	__shared__ float xblkMat[BLOCK_SIZE*BLOCK_SIZE], yblkMat[BLOCK_SIZE*BLOCK_SIZE];

	const int global_x = threadIdx.x + blockIdx.x * blockDim.x;
	const int global_y = threadIdx.y + blockIdx.y * blockDim.y;

	const int blocked_x = blockIdx.x;
	const int blocked_y = blockIdx.y;

	const int blocked_x_id = threadIdx.x;
	const int blocked_y_id = threadIdx.y;

	const int numBlocks = matrixSize / BLOCK_SIZE;

	int xStart = blocked_y * matrixSize * BLOCK_SIZE;
	int yStart = blocked_x * BLOCK_SIZE;

	for (int block = 0; block < numBlocks; block++) {
		xblkMat[blocked_x_id + (blocked_y_id*BLOCK_SIZE)] = x[xStart + ((blocked_y_id*matrixSize) + blocked_x_id)];
		yblkMat[blocked_x_id + (blocked_y_id*BLOCK_SIZE)] = y[yStart + ((blocked_y_id*matrixSize) + blocked_x_id)];

		__syncthreads();
		
		for (int k = 0;k < BLOCK_SIZE;k++) {
			zTemp += xblkMat[k + (blocked_y_id * BLOCK_SIZE)] * yblkMat[blocked_x_id + (k * BLOCK_SIZE)];
		}

		__syncthreads();

		xStart += BLOCK_SIZE;
		yStart += BLOCK_SIZE;
	}

	z[global_x + (global_y * matrixSize)] = zTemp;
 
}

int main(){
	float *x,*y,*z;
	struct timeval start;
	struct timeval end;
	double elapsedTime;
	double numOps;
	float gFLOPS;

	cudaMallocManaged(&x, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
	cudaMallocManaged(&y, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
	cudaMallocManaged(&z, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

	for(int i=0;i<MATRIX_SIZE;i++){
		for(int j=0;j<MATRIX_SIZE;j++){
			*(x + i*MATRIX_SIZE + j) = 1.0f;
			*(y + i*MATRIX_SIZE + j) = 1.0f;
			*(z + i*MATRIX_SIZE + j) = 0.0f;
 		}
	}

	// Keep track of when we start doing work
    gettimeofday(&start, NULL);

    dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE/threads.x,MATRIX_SIZE/threads.y);
	
	matMul<<<grid,threads>>>(x,y,z,MATRIX_SIZE);
	cudaDeviceSynchronize();

    // Keep track of when we finish our work
    gettimeofday(&end, NULL);
    // Calculate the time it took to do the above task
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    elapsedTime /= 1000;
    
    //Calculate the GFLOPS obtained and print it along with the execution time
	numOps = 2 * pow(MATRIX_SIZE, 3);
	gFLOPS = float(1.0e-9 * numOps / elapsedTime);
	printf("CUDA            : %.3f seconds ( %f GFLOPS )\n",elapsedTime,gFLOPS);

	/*cout << "X[23][65] : " << *(x + 23*MATRIX_SIZE + 65) << endl;
	cout << "Y[23][65] : " << *(y + 23*MATRIX_SIZE + 65) << endl;
	cout << "Z[23][65] : " << *(z + 23*MATRIX_SIZE + 65) << endl;*/

	cudaFree(x);
	cudaFree(y);
	cudaFree(z);

	return 0;
}

