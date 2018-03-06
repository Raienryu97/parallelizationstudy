#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "limits.h"
#include "sys/time.h"
#include <omp.h>

#define TRIALS 2
#define BLOCK_MIN 256
#define BLOCK_MAX 256
#define MATRIX_SIZE 4096

int A[MATRIX_SIZE][MATRIX_SIZE],
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
	// Initalize array A and B with '1's
	for (int i = 0; i < MATRIX_SIZE; ++i)
		for (int k = 0; k < MATRIX_SIZE; ++k)
			A[i][k] = B[i][k] = 1;

	// Initalize our matix looping variables once
	int k, j, i, jj, kk;

	// Run TRIALS number of trials for each block size
	for (int trial = 0; trial < TRIALS; ++trial)
	{
		printf("Trial %d: \n", trial);
    // Keep track of when we start doing work
    gettimeofday(&start, NULL);
		// Iterate through the block sizes
		for (int block_size = BLOCK_MIN; block_size <= BLOCK_MAX; block_size = block_size*2)
		{
			memset(C, 0, sizeof(C[0][0] * MATRIX_SIZE * MATRIX_SIZE));
			// Do block matrix multiplication
      #pragma acc data copyin(A[:][:], B[:][:]) copy(C[:][:])
			for (k = 0; k < MATRIX_SIZE; k += block_size)
				for (j = 0; j < MATRIX_SIZE; j += block_size)
        #pragma acc kernels loop gang, vector(128)
        #pragma omp parallel for collapse(3)
					for (i = 0; i < MATRIX_SIZE; ++i)
						for (jj = j; jj < min(j + block_size, MATRIX_SIZE); ++jj)
              for (kk = k; kk < min(k + block_size, MATRIX_SIZE); ++kk)
								C[i][jj] += A[i][kk] * B[kk][jj];
			fflush(stdout);
		}
    printf("C[2][2]: %d\n",C[2][2]);
    // Keep track of when we finish our work
    gettimeofday(&end, NULL);
    // Calculate the time it took to do the above task
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Elapsed: %.3f seconds\n", elapsedTime / 1000);
		puts("");
	}

	return 0;
}
