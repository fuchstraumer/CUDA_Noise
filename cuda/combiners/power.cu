#include "power.cuh"
#include "..\..\cpp\modules\combiners\Power.h"


__global__ void powerKernel(float* output, float* input0, float* input1, const int width, const int height) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	float prev0, prev1;
	prev0 = input0[(j * width) + i];
	prev1 = input1[(j * width) + i];

	// Raise prev0 to the power of prev1 and write to the output.
	output[(j * width) + i] = powf(prev0, prev1);
}

void powerLauncher(float* output, float* input0, float* input1, const int width, const int height) {

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	int blockSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, powerKernel, 0, 0); 
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	powerKernel<<<grid, block >>>(output, input0, input1, width, height); //Call Kernel
	// Check for successful kernel launch
	cudaAssert(cudaGetLastError());
	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING
	// If this completes, kernel is done and "output" contains correct data.
}


