#include "checkerboard.cuh"

__global__ void CheckerboardKernel(float* output, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int center_x = width / 2;
	const int center_y = height / 2;
	if (i >= width || j >= height) {
		return;
	}
	float dist = (i - center_x)*(i - center_x) + (j - center_y)*(j - center_y);
	dist = sqrtf(dist);
	float dist_smaller = dist - floorf(dist);
	float dist_larger = 1.0f - dist;

	float result = dist_smaller < dist_larger ? dist_smaller : dist_larger;
	result = 1.0f - (result * 4.0f);
	//float result = (i + j % 2 == 0) ? -0.25f : 1.0f;
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