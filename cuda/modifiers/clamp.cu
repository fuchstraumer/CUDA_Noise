#include "clamp.cuh"

__global__ void ClampKernel(float* output, float* input, const int width, const int height, const float lower_value, const float upper_value) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	
	// Get previous value.
	float prev = input[(j * width) + i];

	// Compare and clamp to range appropriately.
	if (prev < lower_value) {
		output[(j * width) + i] = lower_value;
	}
	else if (prev > upper_value) {
		output[(j * width) + i] = upper_value;
	}
	else {
		output[(j * width) + i] = prev;
	}

}

void ClampLauncher(float* output, float* input, const int width, const int height, const float lower_value, const float upper_value) {

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(32, 32, 1);
	dim3 grid(width / block.x , height / block.y, 1);
	ClampKernel<<<grid, block>>>(output, input, width, height, lower_value, upper_value);
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