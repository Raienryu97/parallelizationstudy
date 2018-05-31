#define BLOCK_SIZE 8

__kernel void DCT1(	__global float* inputImage,
					__global float* temp,
					__global float* dct8x8Mat,
					__const int matrixSize
)
{

	int i = get_global_id(0);
	int j = get_global_id(1);

	float sum = 0.0f;

	int x = i % BLOCK_SIZE;
	int y = j % BLOCK_SIZE;

	//printf("[%d, %d] , [%d, %d] \t", i, j, x, y);


	for (int k = 0;k < BLOCK_SIZE;k++) {
		sum += dct8x8Mat[x*BLOCK_SIZE + k] * inputImage[k*matrixSize + j];
	}

	temp[i*matrixSize + j] = sum;
}

__kernel void DCT2(__global float* temp,
				   __global float* dctCoeffMat,
				   __global float* dct8x8TMat,
				   __const int matrixSize
)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	float sum = 0.0f;

	int x = i % BLOCK_SIZE;
	int y = j % BLOCK_SIZE;

	for (int k = 0;k < BLOCK_SIZE;k++) {
		sum += temp[i*matrixSize + k] * dct8x8TMat[k*BLOCK_SIZE + y];
	}

	dctCoeffMat[i*matrixSize + j] = sum;
}