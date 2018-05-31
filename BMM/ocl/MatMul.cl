#define BLOCK_SIZE 16

__kernel void MatMul (__global float* x,
	                  __global float* y,
					  __global float* z,
					  __local float* xblkMat,
					  __local float* yblkMat,
					  __const int matrixSize)
{
	float zTemp = 0.0f;

	const int global_x = get_global_id(0);
	const int global_y = get_global_id(1); 

	const int blocked_x = get_group_id(0);
	const int blocked_y = get_group_id(1);

	const int blocked_x_id = get_local_id(0);
	const int blocked_y_id = get_local_id(1);

	const int numBlocks = matrixSize / BLOCK_SIZE;

	int xStart = blocked_y * matrixSize * BLOCK_SIZE;
	int yStart = blocked_x * BLOCK_SIZE;

	for (int block = 0; block < numBlocks; block++) {
		xblkMat[blocked_x_id + (blocked_y_id*BLOCK_SIZE)] = x[xStart + ((blocked_y_id*matrixSize) + blocked_x_id)];
		yblkMat[blocked_x_id + (blocked_y_id*BLOCK_SIZE)] = y[yStart + ((blocked_y_id*matrixSize) + blocked_x_id)];

		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (int k = 0;k < BLOCK_SIZE;k++) {
			zTemp += xblkMat[k + (blocked_y_id * BLOCK_SIZE)] * yblkMat[blocked_x_id + (k * BLOCK_SIZE)];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		xStart += BLOCK_SIZE;
		yStart += BLOCK_SIZE;
	}

	z[global_x + (global_y * matrixSize)] = zTemp;
}