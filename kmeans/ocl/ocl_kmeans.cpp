#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <CL/cl.h>
#include <math.h>

#define num_loops 10000
#define num_data 256
#define data_max_value 10000

using namespace std;

string getPlatformName(cl_platform_id id) {
	size_t size = 0;
	string platformName;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);
	platformName.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char*>(platformName.data()), nullptr);
	return platformName;
}

string getDeviceName(cl_device_id id) {
	size_t size = 0;
	string deviceName;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);
	deviceName.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*>(deviceName.data()), nullptr);
	return deviceName;
}

size_t* getDeviceMaxWorkItemSizes(cl_device_id id) {
	size_t workItems[3];
	clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItems), &workItems, nullptr);
	return workItems;
}

string getProgramBuildInfo(cl_program program, cl_device_id id) {
	size_t size = 0;
	string buildInfo;
	clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
	buildInfo.resize(size);
	clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, size, const_cast<char*>(buildInfo.data()), nullptr);
	return buildInfo;
}

void checkError(cl_int error) {
	if (error != CL_SUCCESS) {
		cerr << "OpenCL call failed with error: " << error << endl;
		getchar();
		exit(1);
	}
}

string loadKernel(const char* kernelName) {
	ifstream in(kernelName);
	string result((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
	return result;
}

cl_program createProgram(const string& source, cl_context context) {
	size_t lengths[1] = { source.size() };
	const char * sources[1] = { source.data() };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
	checkError(error);

	return program;
}

int main() {
	//Get the number of platforms available
	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, nullptr, &numPlatforms);

	//Get the platform IDs
	cl_platform_id optimalPlatform;
	vector <cl_platform_id> platformIDs(numPlatforms);
	clGetPlatformIDs(numPlatforms, platformIDs.data(), nullptr);

	//Get the number of devices available per platform
	vector <cl_uint> numDevices(numPlatforms);
	for (cl_uint i = 0; i < numPlatforms; i++)
		clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices[i]);

	//Get the device IDs
	vector < vector < cl_device_id > > deviceIDs;
	deviceIDs.resize(numPlatforms);
	for (cl_uint i = 0; i < numPlatforms; i++)
		deviceIDs[i].resize(numDevices[i]);
	for (cl_uint i = 0; i < numPlatforms; i++) {
		for (cl_uint j = 0; j < numDevices[i]; j++) {
			clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, numDevices[i], deviceIDs[i].data(), nullptr);
		}
	}

	//Print out the names of the devices available along with the platforms
	optimalPlatform = platformIDs[0];
	int optimalPlatformId = 0;
	for (cl_uint i = 0; i < numPlatforms; i++) {
		if (getPlatformName(platformIDs[i]).find("NVIDIA") != string::npos) {
			optimalPlatform = platformIDs[i];
			optimalPlatformId = i;
		}
		//cout << endl << "Devices on " << getPlatformName(platformIDs[i]) << "Platform" << endl;
		for (cl_uint j = 0; j < numDevices[i]; j++) {
			//cout << "[" << j + 1 << "] : " << getDeviceName(deviceIDs[i][j]) << endl;
		}
	}

	const cl_context_properties contextProperties[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (optimalPlatform),
		0, 0
	};


	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext(contextProperties, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, &error);
	checkError(error);

	cl_program program = createProgram(loadKernel("kmeans.cl"), context);
	//clBuildProgram(program1, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, nullptr);
	//cout << getProgramBuildInfo(program1, deviceIDs[0][0]);
	
	checkError(clBuildProgram(program, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, nullptr));
	//clBuildProgram(program, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, nullptr);
	//cout << getProgramBuildInfo(program, deviceIDs[0][0]);

	cl_kernel kernel = clCreateKernel(program, "kmeans", &error);
	checkError(error);
	
	int *h_data;
	int *h_centroids;
	srand(time(NULL));
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
		//cout << "centroid:" << i << "is" << h_centroids[i*2+0] << "," << h_centroids[i*2+1]<<endl;
		//printf("\n");
	}
	for (unsigned int id = 0; id < numDevices[optimalPlatformId]; id++) {
		
		auto start = chrono::high_resolution_clock::now();
		
		cl_mem d_data, d_centroids;
		
		size_t bytes = num_data * 3 * sizeof(int32_t);
		size_t centroidSize = 3 * 2* sizeof(int32_t);
		
		d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
		checkError(error);
		d_centroids = clCreateBuffer(context, CL_MEM_READ_ONLY, centroidSize, nullptr, &error);
		checkError(error);

		cl_command_queue queue = clCreateCommandQueue(context, deviceIDs[optimalPlatformId][id], 0, &error);
		checkError(error);

		//Write input vectors to device
		checkError(clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0, bytes, h_data, 0, nullptr, nullptr));
		checkError(clEnqueueWriteBuffer(queue, d_centroids, CL_TRUE, 0, centroidSize, h_centroids, 0, nullptr, nullptr));
		
		int data_size = num_data;
		const size_t totSize = 256;
		const size_t global_size =ceil(num_data/(float)totSize)*totSize;

		checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data));
		checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_centroids));
		checkError(clSetKernelArg(kernel, 2, sizeof(int), &data_size));

		for (int loop = 0; loop < num_loops;loop++) {
			
			//cout <<"The 4042st element:"<<h_data[4042*3+2]<<endl;
			
			checkError(clEnqueueWriteBuffer(queue, d_centroids, CL_TRUE, 0, centroidSize, h_centroids, 0, nullptr, nullptr));
		
			checkError(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &totSize, 0, NULL, NULL));
			clFinish(queue);

			//Read the results back from the device
			checkError(clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, bytes, h_data, 0, NULL, NULL));
			//cout <<"The 4042st element:"<<h_data[4042*3+2]<<endl;

			float c0x_avg = 0, c0y_avg = 0, c1x_avg = 0, c1y_avg = 0, c2x_avg = 0, c2y_avg = 0;
			int c0_count = 0, c1_count = 0, c2_count = 0;

			//cout << "The last element" <<h_data[2046*3+2] << endl;
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

		clReleaseMemObject(d_data);
		clReleaseMemObject(d_centroids);
		clReleaseCommandQueue(queue);

		/*cout << "Centroid 1 : (" << h_centroids[0*2+0] << " , " << h_centroids[0*2+1] << ")" << endl;
		cout << "Centroid 2 : (" << h_centroids[1*2+0] << " , " << h_centroids[1*2+1] << ")" << endl;
		cout << "Centroid 3 : (" << h_centroids[2*2+0] << " , " << h_centroids[2*2+1] << ")" << endl;*/


		auto finish = chrono::high_resolution_clock::now();
		chrono::duration<double> elapsed = finish - start;
		printf("OpenCL          : %.3f seconds\n", elapsed.count());
	}

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseContext(context);
	
	return 0;
}