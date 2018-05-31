#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include "limits.h"
#include "sys/time.h"

#define BLOCK_SIZE 16
#define MATRIX_SIZE 256

float A[MATRIX_SIZE][MATRIX_SIZE],
      B[MATRIX_SIZE][MATRIX_SIZE],
      C[MATRIX_SIZE][MATRIX_SIZE];

#pragma acc routine seq
int min(int a, int b)
{
	return a < b ? a : b;
}

int main(int argc, char*  argv[])
{
	struct timeval start;
	struct timeval end;
	double elapsedTime;
	double numOps;
	float gFLOPS;
	
	// Initalize array A and B with '1's and C with '0's
	for (int i = 0; i < MATRIX_SIZE; ++i)
		for (int k = 0; k < MATRIX_SIZE; ++k)
			A[i][k] = B[i][k] = 1.0;
	memset(C, 0, sizeof(C[0][0] * MATRIX_SIZE * MATRIX_SIZE));
	
	// Initalize our matix looping variables once
	int k, j, i, jj, kk;

    // Keep track of when we start doing work
    gettimeofday(&start, NULL);
			
	// Do block matrix multiplication
    #pragma acc data copyin(A[:][:], B[:][:]) copy(C[:][:])
	for (k = 0; k < MATRIX_SIZE; k += BLOCK_SIZE)
		for (j = 0; j < MATRIX_SIZE; j += BLOCK_SIZE)
        	#pragma acc kernels loop gang, vector(128)
			for (i = 0; i < MATRIX_SIZE; ++i)
				for (jj = j; jj < min(j + BLOCK_SIZE, MATRIX_SIZE); ++jj)
              		for (kk = k; kk < min(k + BLOCK_SIZE, MATRIX_SIZE); ++kk)
						C[i][jj] += A[i][kk] * B[kk][jj];
    
    // Keep track of when we finish our work
    gettimeofday(&end, NULL);

    // Calculate the time it took to do the above task
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    elapsedTime /= 1000;

	//Calculate the GFLOPS obtained and print it along with the execution time
	numOps = 2 * pow(MATRIX_SIZE, 3);
	gFLOPS = (float)(1.0e-9 * numOps / elapsedTime);

    printf("OpenACC         : %.3f seconds ( %f GFLOPS )\n", elapsedTime,gFLOPS);

	return 0;
}