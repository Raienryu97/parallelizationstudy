#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <CL/cl.h>
#include <math.h>
#include "sys/time.h"

#define MATRIX_SIZE 2048
#define BLOCK_SIZE  16

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

int min(int a, int b)
{
	return a < b ? a : b;
}

int main() {
	struct timeval start,end;
	double elapsedTime;
	//Get the number of platforms available
	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, nullptr, &numPlatforms);

	//Get the platform IDs
	cl_platform_id optimalPlatform;
	vector <cl_platform_id> platformIDs(numPlatforms);
	clGetPlatformIDs(numPlatforms, platformIDs.data(), nullptr);

	//Get the number of devices available per platform
	vector <cl_uint> numDevices(numPlatforms);
	for (cl_uint i = 0;i < numPlatforms; i++)
		clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices[i]);

	//Get the device IDs
	vector < vector < cl_device_id > > deviceIDs;
	deviceIDs.resize(numPlatforms);
	for (cl_uint i = 0; i < numPlatforms;i++)
		deviceIDs[i].resize(numDevices[i]);
	for (cl_uint i = 0; i < numPlatforms;i++) {
		for (cl_uint j = 0; j < numDevices[i]; j++) {
			clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, numDevices[i], deviceIDs[i].data(), nullptr);
		}
	}

	//Print out the names of the devices available along with the platforms
	optimalPlatform = platformIDs[0];
	int optimalPlatformId = 0;
	for (cl_uint i = 0;i < numPlatforms;i++) {
		if (getPlatformName(platformIDs[i]).find("NVIDIA") != string::npos) {
			optimalPlatform = platformIDs[i];
			optimalPlatformId = i;
		}
		cout << endl << "Devices on " << getPlatformName(platformIDs[i]) << "Platform" << endl;
		for (cl_uint j = 0; j < numDevices[i]; j++) {
			cout << "[" << j + 1 << "] : " << getDeviceName(deviceIDs[i][j]) << endl;
		}
	}

	const cl_context_properties contextProperties[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (optimalPlatform),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext(contextProperties, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, &error);
	checkError(error);

	cl_program program = createProgram(loadKernel("MatMul.cl"), context);

	checkError(clBuildProgram(program, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, nullptr));

	cl_kernel kernel = clCreateKernel(program, "MatMul", &error);
	checkError(error);

	//Allocate Test Data
	const int32_t matrixSize = MATRIX_SIZE;
	float *h_x = (float *)malloc(matrixSize * matrixSize * sizeof(float));
	float *h_y = (float *)malloc(matrixSize * matrixSize * sizeof(float));
	float *h_z = (float *)malloc(matrixSize * matrixSize * sizeof(float));
	float *cpu_out = (float *)malloc(matrixSize * matrixSize * sizeof(float));
	double numOps;
	float gFLOPS;

	cout << endl << "About to multiply two matrices of sizes " << matrixSize << " X " << matrixSize << endl;
	cout << endl << "Execution Time and GFLOPS count on :" << endl << endl ;

	for (unsigned int id = 0; id < numDevices[optimalPlatformId]; id++) {

		cl_mem d_x, d_y;      //device input buffers
		cl_mem d_z;           //device output buffers

		size_t bytes = matrixSize * matrixSize * sizeof(float);

		d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &error);
		checkError(error);
		d_y = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &error);
		checkError(error);
		d_z = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &error);
		checkError(error);

		for (int i = 0;i < matrixSize; i++) {
			for (int j = 0; j < matrixSize; j++) {
				*(h_x + i * matrixSize + j) = 1.0;
				*(h_y + i * matrixSize + j) = 1.0;
				*(cpu_out + i * matrixSize + j) = 0.0;
			}
		}

		size_t localSize[2], globalSize[2];
		localSize[0] = BLOCK_SIZE;
		localSize[1] = BLOCK_SIZE;
		globalSize[0] = matrixSize;
		globalSize[1] = matrixSize;

		//Generate command queue
		cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceIDs[optimalPlatformId][id], 0, &error);
		checkError(error);

		//Write input vectors to device
		checkError(clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, bytes, h_x, 0, nullptr, nullptr));
		checkError(clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, bytes, h_y, 0, nullptr, nullptr));

		//Set kernel parameters
		checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x));
		checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y));
		checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_z));
		checkError(clSetKernelArg(kernel, 3, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, nullptr));
		checkError(clSetKernelArg(kernel, 4, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, nullptr));
		checkError(clSetKernelArg(kernel, 5, sizeof(int32_t), &matrixSize));


		gettimeofday(&start, NULL);

		//Execute the kernel and wait for execution to finish
		checkError(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL));
		clFinish(queue);

		//Read the results back from the device
		checkError(clEnqueueReadBuffer(queue, d_z, CL_TRUE, 0, bytes, h_z, 0, NULL, NULL));

		gettimeofday(&end, NULL);
		elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
		elapsedTime /= 1000.0;

		//Calculate the GFLOPS obtained and print it along with the execution time
		numOps = 2 * pow(matrixSize, 3);
		gFLOPS = float(1.0e-9 * numOps / elapsedTime);
		cout << "[" << id + 1 << "] " << getDeviceName(deviceIDs[optimalPlatformId][id]) << ": " << elapsedTime << " seconds ( " << gFLOPS << " GFLOPS )" << endl;
		//Release OpenCL resources
		clReleaseMemObject(d_x);
		clReleaseMemObject(d_y);
		clReleaseMemObject(d_z);
		clReleaseCommandQueue(queue);

	}
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseContext(context);

	//Run the same code on CPU without OpenCL and time it

	gettimeofday(&start, NULL);

	for(int k=0; k< MATRIX_SIZE; k+= BLOCK_SIZE)
		for(int j=0;j<MATRIX_SIZE;j+=BLOCK_SIZE)
			for(int i=0;i<MATRIX_SIZE;i++)
				for(int jj=j; jj<min(j + BLOCK_SIZE, MATRIX_SIZE);jj++)
					for(int kk=k; kk<min(k + BLOCK_SIZE, MATRIX_SIZE);kk++)
						*(cpu_out + i * matrixSize + jj) += *(h_x + i * matrixSize + kk) * *(h_y + kk * matrixSize + j);

	gettimeofday(&end, NULL);
	elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
  elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
	elapsedTime /= 1000.0;
	//Calculate the GFLOPS obtained and print it along with the execution time
	numOps = 2 * pow(matrixSize, 3);
	gFLOPS = float(1.0e-9 * numOps / elapsedTime);
	cout << endl << "Single thread CPU : " << elapsedTime << " seconds ( " << gFLOPS << " GFLOPS )" <<  endl;

	int count = 0;
	for (int i = 0;i < matrixSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			if (*(cpu_out + i * matrixSize + j) != *(h_z + i * matrixSize + j)) {
				count++;
			}
		}
	}
	cout << endl << "Found " << count << " errors in the output matrix of GPU" << endl;

	getchar();
	return 0;
}
