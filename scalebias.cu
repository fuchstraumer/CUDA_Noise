#include "scalebias.cuh"

__global__ void scalebiasKernel(cudaSurfaceObject_t output, cudaSurfaceObject_t input, const int width, const int height, float scale, float bias) {
	// Get current pixel.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	// Reading from a surface still requires the byte offset, so multiply the x coordinate by the size of a float in bytes.
	// surf2Dread also writes the value at the point to a pre-existing variable, so declare soemthing like "prev" and pass
	// it as a reference (&prev) to the surf2Dread function.
	float prev;
	surf2Dread(&prev, input, i * sizeof(float), j);

	float out;
	out = prev* scale + bias; // for default value for scale is 1 and bias is 0;

	// Store value in destination surface.
	surf2Dwrite(out, output, i * sizeof(float), j);
}

void scalebiasLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input, const int width, const int height, float scale, float bias){
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	// Setup dimensions of kernel launch. 
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	scalebiasKernel << <numBlocks, threadsPerBlock >> >(output, input, width, height, scale, bias);
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
}