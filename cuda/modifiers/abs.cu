#include "abs.cuh"


__global__ void absKernel(cudaSurfaceObject_t out, cudaSurfaceObject_t in, const int width, const int height) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}

	float prev;
	surf2Dread(&prev, input, i * sizeof(float), j);

	float final_value;

	if (prev <= 0)
	{
		final_value = -prev;
	}

	else
	{
		final_value = prev;
	}
	

	surf2Dwrite(final_value, out, i * sizeof(float), j);

}

void absLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t in, const int width, const int height) {
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif // CUDA_TIMING_TESTS

	// Setup dimensions of kernel launch. 

	// Use occupancy calc to find optimal sizes.
	int blockSize, minGridSize;
#ifdef CUDA_TIMING_TESTS
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)absKernel, 0, 0); //???
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	absKernel << <block, grid >> >(out, in, width, height);
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