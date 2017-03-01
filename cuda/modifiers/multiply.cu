#include "power.cuh"


__global__ void multiplyKernel(cudaSurfaceObject_t out, cudaSurfaceObject_t in, const int width, const int height, float factor) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= width || j >= height) {
		return;
	}

	float prev;
	surf2Dread(&prev, input, i * sizeof(float), j);

	float final_value;
	final_value = prev * factor;



	surf2Dwrite(final_value, out, i * sizeof(float), j);

}

void multiplyLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t in, const int width, const int height, float factor) {
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	// Setup dimensions of kernel launch. 
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);


	multiplyKernel << <block, grid >> >(out, in, width, height, factor);


	// Check for succesfull kernel launch
	cudaAssert(cudaGetLastError());
	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());
#ifdef CUDA_TIMING_TESTS
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_TIMING_TESTS
	// If this completes, kernel is done and "output" contains correct data.
}