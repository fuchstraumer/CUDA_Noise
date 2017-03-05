#include "checkerboard.cuh"

__global__ void CheckerboardKernel(float* output, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	float result = (i & 1 ^ j & 1) ? -1.0f : 1.0f;
	output[(j * width) + i] = result;
}

void CheckerboardLauncher(float *output, const int width, const int height) {

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	dim3 block(32, 32, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	CheckerboardKernel<<<block, grid>>>(output, width, height);
	// Confirm launch is good
	cudaAssert(cudaGetLastError());
	// Synchronize device to complete kernel
	cudaAssert(cudaDeviceSynchronize());

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

}