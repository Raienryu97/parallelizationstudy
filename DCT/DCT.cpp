#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <CL/cl.h>
#include <omp.h>
#include "sys/time.h"
#include "DCT.h"

using namespace std;

#define MATRIX_SIZE 2048
#define BLOCK_SIZE 8

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

string getProgramBuildInfo(cl_program program, cl_device_id id) {
	size_t size = 0;
	string buildInfo;
	clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
	buildInfo.resize(size);
	clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, size, const_cast<char*>(buildInfo.data()), nullptr);
	return buildInfo;
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

void checkError(cl_int error, string message) {
	if (error != CL_SUCCESS) {
		cerr << "OpenCL call failed with error: " << error << endl;
		cerr << message << endl;
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

	cl_program program = createProgram(loadKernel("DCT.cl"), context);
	//clBuildProgram(program, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, nullptr);
	//cout << getProgramBuildInfo(program, deviceIDs[0][0]);
	checkError(clBuildProgram(program, numDevices[optimalPlatformId], deviceIDs[optimalPlatformId].data(), nullptr, nullptr, nullptr));

	cl_kernel kernel = clCreateKernel(program, "DCT1", &error);
	checkError(error);

	cl_kernel kernel1 = clCreateKernel(program, "DCT2", &error);
	checkError(error);

	const int32_t matrixSize = MATRIX_SIZE;

	cl_float *h_img = (float *)malloc(matrixSize * matrixSize * sizeof(cl_float));
	cl_float *h_imgCopy = (float *)malloc(matrixSize * matrixSize * sizeof(cl_float));
	cl_float *dctCoeffMatrixGPU = (float *)malloc(matrixSize * matrixSize * sizeof(cl_float));
	cl_float *dctCoeffMatrix = (float *)malloc(matrixSize * matrixSize * sizeof(cl_float));
	cl_float *temp = (float *)malloc(matrixSize * matrixSize * sizeof(cl_float));
	cl_float *tempGPU = (float *)malloc(matrixSize * matrixSize * sizeof(cl_float));
	
	mt19937 rng(time(NULL));
	uniform_int_distribution<int> gen(0, 255);

	/*cl_float lolTest[8][8] = {
		{154, 123, 123, 123, 123, 123, 123, 136},
		{192, 180, 136, 154, 154, 154, 136, 110},
		{254, 198, 154, 154, 180, 154, 123, 123},
		{239, 180, 136, 180, 180, 166, 123, 123},
		{180, 154, 136, 167, 166, 149, 136, 136},
		{128, 136, 123, 136, 154, 180, 198, 154},
		{123, 105, 110, 149, 136, 136, 180, 166},
		{110, 136, 123, 123, 123, 136, 154, 136}
	};*/

	for (int i = 0; i < matrixSize; i++) {
		for (int j = 0;j < matrixSize; j++) {
			*(h_img + i * matrixSize + j) = static_cast<cl_float>(gen(rng));
			// Lets make pixel values lie in [-127, 128] instead of [0,255]
			*(h_imgCopy + i * matrixSize + j) = *(h_img + i * matrixSize + j) - 128;
			//*(h_imgCopy + i * matrixSize + j) = lolTest[i][j] - 128;
		}
	}

	cout << endl << "About to run DCT on matrix of size " << matrixSize << " X " << matrixSize << endl;
	cout << endl << "Execution Time on :" << endl << endl;

	for (unsigned int id = 0; id < numDevices[optimalPlatformId]; id++) {

		cl_mem d_x;     //device input buffer
		cl_mem d_y,d_z;     //device output buffer
		cl_mem d_tempGPU;
		cl_mem d_dct8x8Mat;
		cl_mem d_dct8x8TMat;

		size_t bytes = matrixSize * matrixSize * sizeof(float);
		size_t bytes1 = BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_float);

		d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &error);
		checkError(error);
		d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &error);
		checkError(error);
		d_z = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &error);
		checkError(error);
		d_tempGPU = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &error);
		checkError(error);
		d_dct8x8Mat = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes1, nullptr, &error);
		checkError(error);
		d_dct8x8TMat = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes1, nullptr, &error);
		checkError(error);

		size_t localSize[2], globalSize[2];
		localSize[0] = BLOCK_SIZE;
		localSize[1] = BLOCK_SIZE;
		globalSize[0] = matrixSize;
		globalSize[1] = matrixSize;

		//Generate command queue
		cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceIDs[optimalPlatformId][id], 0, &error);
		checkError(error);

		//Write input vectors to device
		checkError(clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, bytes, h_imgCopy, 0, nullptr, nullptr));
		checkError(clEnqueueWriteBuffer(queue, d_dct8x8Mat, CL_TRUE, 0, bytes1, dct8x8Matrix, 0, nullptr, nullptr));

		//Set kernel parameters
		checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x));
		checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y));
		checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_dct8x8Mat));
		checkError(clSetKernelArg(kernel, 3, sizeof(int32_t), &matrixSize));


		gettimeofday(&start, NULL);

		//Execute the kernel and wait for execution to finish
		checkError(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL));
		clFinish(queue);

		//Read the results back from the device
		checkError(clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, bytes, tempGPU, 0, NULL, NULL));

		//Generate command queue
		cl_command_queue queue1 = clCreateCommandQueueWithProperties(context, deviceIDs[optimalPlatformId][id], 0, &error);
		checkError(error);

		//Write input vectors to device
		checkError(clEnqueueWriteBuffer(queue1, d_tempGPU, CL_TRUE, 0, bytes, tempGPU, 0, nullptr, nullptr));
		checkError(clEnqueueWriteBuffer(queue1, d_dct8x8TMat, CL_TRUE, 0, bytes1, dct8x8MatrixTranspose, 0, nullptr, nullptr));

		//Set kernel parameters
		checkError(clSetKernelArg(kernel1, 0, sizeof(cl_mem), &d_tempGPU));
		checkError(clSetKernelArg(kernel1, 1, sizeof(cl_mem), &d_z));
		checkError(clSetKernelArg(kernel1, 2, sizeof(cl_mem), &d_dct8x8TMat));
		checkError(clSetKernelArg(kernel1, 3, sizeof(int32_t), &matrixSize));

		//Execute the kernel and wait for execution to finish
		checkError(clEnqueueNDRangeKernel(queue1, kernel1, 2, NULL, globalSize, localSize, 0, NULL, NULL));
		clFinish(queue1);

		//Read the results back from the device
		checkError(clEnqueueReadBuffer(queue, d_z, CL_TRUE, 0, bytes, dctCoeffMatrixGPU, 0, NULL, NULL));

		gettimeofday(&end, NULL);
		elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
		elapsedTime /= 1000.0;

		//Calculate the GFLOPS obtained and print it along with the execution time
		cout << "[" << id + 1 << "] " << getDeviceName(deviceIDs[optimalPlatformId][id]) << ": " << elapsedTime << " seconds " << endl;
		//Release OpenCL resources
		clReleaseMemObject(d_x);
		clReleaseMemObject(d_y);
		clReleaseMemObject(d_z);
		clReleaseMemObject(d_tempGPU);
		clReleaseMemObject(d_dct8x8Mat);
		clReleaseMemObject(d_dct8x8TMat);
		clReleaseCommandQueue(queue);
		clReleaseCommandQueue(queue1);

	}
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseKernel(kernel1);
	clReleaseContext(context);

	gettimeofday(&start, NULL);

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < matrixSize; i += BLOCK_SIZE) {
		for (int j = 0; j < matrixSize; j += BLOCK_SIZE) {

			
			float sum = 0;

			// DCTMatrix X Image
			for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
				for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
					for (int k = 0; k < BLOCK_SIZE; k++) {
						sum += dct8x8Matrix[ii%BLOCK_SIZE][k] * *(h_imgCopy + k * matrixSize + jj);
					}
					*(temp + ii * matrixSize + jj) = sum;
					sum = 0;
					//cout << "[ " << ii << ", " << jj << " ]" << "\t";
				}
			}

			
			sum = 0;

			// (DCTMatrix X Image) X (DCTTransposeMatrix)
			for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
				for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
					for (int k = 0; k < BLOCK_SIZE; k++) {
						sum += *(temp + ii * matrixSize + k) * dct8x8MatrixTranspose[k][jj%BLOCK_SIZE];
					}
					*(dctCoeffMatrix + ii * matrixSize + jj) = sum;
					sum = 0;
				}
			}
		}
	}

	gettimeofday(&end, NULL);
	elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
  	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
	elapsedTime /= 1000.0;
	cout << endl << "Multicore CPU Execution : " << elapsedTime << " seconds" << endl;

	gettimeofday(&start, NULL);

	for (int i = 0; i < matrixSize; i += BLOCK_SIZE) {
		for (int j = 0; j < matrixSize; j += BLOCK_SIZE) {

			
			float sum = 0;

			// DCTMatrix X Image
			for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
				for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
					for (int k = 0; k < BLOCK_SIZE; k++) {
						sum += dct8x8Matrix[ii%BLOCK_SIZE][k] * *(h_imgCopy + k * matrixSize + jj);
					}
					*(temp + ii * matrixSize + jj) = sum;
					sum = 0;
					//cout << "[ " << ii << ", " << jj << " ]" << "\t";
				}
			}

			
			sum = 0;

			// (DCTMatrix X Image) X (DCTTransposeMatrix)
			for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
				for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
					for (int k = 0; k < BLOCK_SIZE; k++) {
						sum += *(temp + ii * matrixSize + k) * dct8x8MatrixTranspose[k][jj%BLOCK_SIZE];
					}
					*(dctCoeffMatrix + ii * matrixSize + jj) = sum;
					sum = 0;
				}
			}
		}
	}

	gettimeofday(&end, NULL);
	elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
  	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
	elapsedTime /= 1000.0;
	cout << endl << "Single thread CPU : " << elapsedTime << " seconds" << endl;

	int count = 0;
	for (int i = 0;i < matrixSize;i++) {
		for (int j = 0; j < matrixSize;j++) {
			if (*(dctCoeffMatrix + i * matrixSize + j) != *(dctCoeffMatrixGPU + i * matrixSize + j)) {
				cout << "[ " << i << " , " << j << " ]" << *(dctCoeffMatrix + i * matrixSize + j) << " : " <<  *(dctCoeffMatrixGPU + i * matrixSize + j) << endl;
				count++;
			}
		}
	}

	cout << endl << "Found " << count << " errors in the output matrix of GPU" << endl;
	return 0;
}
