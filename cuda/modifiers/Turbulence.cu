#include "Turbulence.cuh"
#include "../generators/FBM.cuh"
/*
	
	Turbulence process:

	1. Get current pixel position.
	2. Offset pixel position using turbulence device functions.
	3. Before reading with pixel position, make sure its in range of surfaceObject
	4. Read from input with offset position, and use this value to set the (i,j) position to this new value in output.

*/

__global__ void TurbulenceKernel(float* out, const float* input, const int width, const int height, const noise_t noise_type, const int roughness, const int seed, const float strength, const float freq) {
	// Get current pixel.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	// Return if out of bounds.
	if (i >= width || j >= height) {
		return;
	}
	// Position that will be displaced
	int2 displace;
	displace.x = i;
	displace.y = j;
	float x_distort, y_distort;
	if (noise_type == noise_t::PERLIN) {
		x_distort = FBM2d(make_float2(i, j), freq, 1.50f, 0.60f, seed, roughness) * strength;
		y_distort = FBM2d(make_float2(i, j), freq, 1.50f, 0.60f, seed, roughness) * strength;
	}
	else {
		x_distort = FBM2d_Simplex(make_float2(i, j), freq, 1.50f, 0.60f, seed, roughness) * strength;
		y_distort = FBM2d_Simplex(make_float2(i, j), freq, 1.50f, 0.60f, seed, roughness) * strength;
	}
	
	displace.x += x_distort;
	displace.y += y_distort;
	displace.x %= width;
	displace.y %= height;
	displace.x = clamp(displace.x, 0, width - 1);
	displace.y = clamp(displace.y, 0, height - 1);
	// Get offset value.
	// Add it to previous value and store the result in the output array.
	out[(j * width) + i] = input[(displace.y * width) + displace.x];
}

void TurbulenceLauncher(float* out, const float* input, const int width, const int height, const noise_t noise_type, const int roughness, const int seed, const float strength, const float freq){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	TurbulenceKernel<<<numBlocks, threadsPerBlock>>>(out, input, width, height, noise_type, roughness, seed, strength, freq);
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

	// If this completes, kernel is done and "output" contains correct data.
}
