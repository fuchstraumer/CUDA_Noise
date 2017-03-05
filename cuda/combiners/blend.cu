#include "blend.cuh"
#include "../cutil_math.cuh"

__global__ void BlendKernel(float *output, const float* in0, const float* in1, const float* control, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	output[(j * width) + i] = lerp(in0[(j * width) + i], in1[(j * width) + i], (control[(j * width) + i] + 1.0f) / 2.0f);
}

void BlendLauncher(float * output, const float * in0, const float * in1, const float * weight, const int width, const int height){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(32, 32, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	BlendKernel<<<grid,block>>>(output, in0, in1, weight, width, height);
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