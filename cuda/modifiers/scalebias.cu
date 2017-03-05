#include "scalebias.cuh"

__global__ void scalebiasKernel(float* output, float* input, const int width, const int height, float scale, float bias) {
	// Get current pixel.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}

	output[(j * width) + i] = (input[(j * width) + i] * scale) + bias; // for default value for scale is 1 and bias is 0;
}

void scalebiasLauncher(float* output, float* input, const int width, const int height, float scale, float bias){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(32, 32, 1);
	dim3 grid((width - 1) / block.x + 1, (height - 1) / block.y + 1, 1);
	scalebiasKernel<<<grid, block>>>(output, input, width, height, scale, bias);
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