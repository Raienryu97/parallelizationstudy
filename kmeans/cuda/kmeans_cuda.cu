#include <stdio.h>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cudalibxt.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <sys/time.h>


#define num_loops 10000
#define num_data 256
#define data_max_value 10000
#define BLOCK_SIZE  1024


__global__ void kernel_kmeans(int *data, int *centroids, int numdata)
{
	int row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
   // int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float d_c0, d_c1, d_c2;
	float2 pt;
	float2 ctr0,ctr1,ctr2;

	pt.x = data[(row * 3) + 0];
	pt.y = data[(row * 3) + 1];
	
	ctr0.x = centroids[(0 * 2) + 0];
	ctr0.y = centroids[(0 * 2) + 1];
	
	ctr1.x = centroids[(1 * 2) + 0];
	ctr1.y = centroids[(1 * 2) + 1];
	
	ctr2.x = centroids[(2 * 2) + 0];
	ctr2.y = centroids[(2 * 2) + 1];

	d_c0 = hypot(pt.x-ctr0.x,pt.y-ctr0.y);
	d_c1 = hypot(pt.x-ctr1.x,pt.y-ctr1.y);
	d_c2 = hypot(pt.x-ctr2.x,pt.y-ctr2.y);

	if ((int)d_c0 < (int)d_c1 && (int)d_c0 < (int)d_c2)
		data[(3 * row) + 2] = 0;
	else if ((int)d_c1 < (int)d_c0 && (int)d_c1 < (int)d_c2)
		data[(3 * row) + 2] = 1;
	else if ((int)d_c2 < (int)d_c0 && (int)d_c2 < (int)d_c1)
		data[(3 * row) + 2] = 2;





}



int main()
{
	struct timeval start;
	struct timeval end;
	double elapsedTime;
	//int h_data[num_data][3];
	//int h_centroids[3][2];
	srand(time(NULL));
	int *h_data;
	int *h_centroids;
	//srand(time(NULL));
	h_data = (int*)malloc(num_data * 3 * sizeof(int));
	h_centroids = (int*)malloc(3 * 2 * sizeof(int));
	
	for (int i = 0; i < num_data; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			
			h_data[ i* 3 + j]  = (rand() % data_max_value) + 1;
			//h_data[ i][j]  = 1;//(rand() % data_max_value) + 1;

		}
	}

	for (int i = 0; i < 3; i++)
	{
		int index = rand() % num_data;
		h_centroids[i*2+0] = h_data[index*3+0];
		h_centroids[i*2+1] = h_data[index*3+1];
		//cout << "centroid:" << i << "is" << h_centroids[i*2+0] << "," << centroids[i*2+1] << endl;
		//printf("centroid %d is %d,%d\n",i,h_centroids[i*2+0],h_centroids[i*2+1]);
		//printf("\n");
	}

	int *d_data;
	int *d_centroids;

	cudaMalloc(&d_data , num_data*3*sizeof(int));
	cudaMalloc(&d_centroids , 3*2*sizeof(int));

	cudaMemcpy(d_data, h_data, num_data*3*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids, h_centroids, 3*2*sizeof(int), cudaMemcpyHostToDevice);

	//dim3 threads(BLOCK_SIZE);
    //dim3 grid(num_data/threads.x);

   
    gettimeofday(&start, NULL);
    
    for (int loop = 0; loop < num_loops;loop++) 
    {


    	cudaMemcpy(d_centroids, h_centroids, 3*2*sizeof(int), cudaMemcpyHostToDevice);

    	kernel_kmeans<<<num_data/BLOCK_SIZE,BLOCK_SIZE>>>(d_data,d_centroids,num_data);

    	cudaMemcpy(h_data, d_data, num_data*3*sizeof(int), cudaMemcpyDeviceToHost);

    	float c0x_avg = 0, c0y_avg = 0, c1x_avg = 0, c1y_avg = 0, c2x_avg = 0, c2y_avg = 0;
			int c0_count = 0, c1_count = 0, c2_count = 0;


		// moving the centroid step

		for (int i = 0; i < num_data; i++)
		{
			if (h_data[i*3+2] == 0)
			{
				c0x_avg = c0x_avg + h_data[i*3+0];
				c0y_avg = c0y_avg + h_data[i*3+1];
				c0_count++;
			}
			else if (h_data[i*3+2] == 1)
			{
				c1x_avg = c1x_avg + h_data[i*3+0];
				c1y_avg = c1y_avg + h_data[i*3+1];
				c1_count++;
			}
			else if (h_data[i*3+2] == 2)
			{
				c2x_avg = c2x_avg + h_data[i*3+0];
				c2y_avg = c2y_avg + h_data[i*3+1];
				c2_count++;
			}
			else {
				// No minimum was found, maybe equal ?
			}
		}
		if(c0_count == 0){
			c0x_avg = h_centroids[0*2+0];
			c0y_avg = h_centroids[0*2+1];
			c0_count = 1;
		}
		else if(c1_count == 0){
			c1x_avg = h_centroids[1*2+0];
			c1y_avg = h_centroids[1*2+1];
			c1_count = 1;
		}
		else if(c2_count == 0){
			c2x_avg = h_centroids[2*2+0];
			c2y_avg = h_centroids[2*2+1];
			c2_count = 1;
		}
		h_centroids[0*2+0] = c0x_avg / c0_count;
		h_centroids[0*2+1] = c0y_avg / c0_count;
		h_centroids[1*2+0] = c1x_avg / c1_count;
		h_centroids[1*2+1] = c1y_avg / c1_count;
		h_centroids[2*2+0] = c2x_avg / c2_count;
		h_centroids[2*2+1] = c2y_avg / c2_count;
		
	}

	cudaFree(d_data);
	cudaFree(d_centroids);
/*
	cout << "Centroid 1 : (" << h_centroids[0*2+0] << " , " << h_centroids[0*2+1] << ")" << endl;
	cout << "Centroid 2 : (" << h_centroids[1*2+0] << " , " << h_centroids[1*2+1] << ")" << endl;
	cout << "Centroid 3 : (" << h_centroids[2*2+0] << " , " << h_centroids[2*2+1] << ")" << endl;
*/
	/*printf("Centroid 1 : ( %d),(%d)\n",h_centroids[0*2+0],h_centroids[0*2+1]);
	printf("Centroid 2 : ( %d),(%d)\n",h_centroids[1*2+0],h_centroids[1*2+1]);
	printf("Centroid 3 : ( %d),(%d)\n",h_centroids[2*2+0],h_centroids[2*2+1]);*/

	gettimeofday(&end, NULL);
    // Calculate the time it took to do the above task
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    elapsedTime /= 1000;
    
    printf("CUDA            : %.3f seconds\n",elapsedTime);

	return 0;


}






