#include "Add.cuh"

__global__ void AddKernel(float* output, float* input0, float* input1, const int width, const int height) {
	// Get current pixel.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	// Reading from a surface still requires the byte offset, so multiply the x coordinate by the size of a float in bytes.
	// surf2Dread also writes the value at the point to a pre-existing variable, so declare soemthing like "prev" and pass
	// it as a reference (&prev) to the surf2Dread function.
	float prev0 = input0[(j * width) + i];
	float prev1 = input1[(j * width) + i];
	// Add values and store in output.
	output[(j * width) + i] = prev0 + prev1;
}

void AddLauncher(float* output, float* input0, float* input1, const int width, const int height){
#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	int blockSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, AddKernel, 0, 0);
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	AddKernel<<<grid, block>>>(output, input0, input1, width, height);
	// Check for succesfull kernel launch
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
}