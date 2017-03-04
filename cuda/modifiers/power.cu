#include "power.cuh"


__global__ void powerKernel(cudaSurfaceObject_t output, cudaSurfaceObject_t input0, cudaSurfaceObject_t input1, const int width, const int height) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	float prev0, prev1;
	surf2Dread(&prev0, input0, i * sizeof(float), j);
	surf2Dread(&prev1, input1, i * sizeof(float), j);
	float final_value;
	final_value = powf(prev0, prev1);

	surf2Dwrite(final_value, output, i * sizeof(float), j);

}

void powerLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input0, cudaSurfaceObject_t input1, const int width, const int height) {

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
	powerKernel<<<grid, block >>>(output, input0, input1, width, height, p); //Call Kernel
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