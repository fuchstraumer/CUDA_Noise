#include "normalize.cuh"
#include <thrust/execution_policy.h>
__global__ void NormalizeKernel(float* output, const float* input, const int width, const int height, const float max, const float min) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}

	// Normalize previous value to 0.0 - 1.0 range and write to output.
	output[(j * width) + i] = -1.0f * (input[(j * width) + i] - min) / (min - max);
}

void NormalizeLauncher(float * output, float * input, const int width, const int height){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	thrust::device_vector<float> in(input, input + (width * height));
	float max = thrust::reduce(thrust::device,input, input + (width * height), -1e10f, thrust::maximum<float>());
	float min = thrust::reduce(thrust::device,in.begin(), in.end(), 1e10f, thrust::minimum<float>());

	dim3 block(32, 32, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	NormalizeKernel<<<block, grid>>>(output, input, width, height, max, min);
	// Confirm launch is good
	cudaAssert(cudaGetLastError());
	// Synchronize device to complete kernel
	cudaAssert(cudaDeviceSynchronize());

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Normalize Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING
}