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
	float2 distort;
	if (noise_type == noise_t::PERLIN) {
		distort.x = FBM2d(make_float2(i, j), freq, 2.20f, 0.90f, seed, roughness) * strength;
		distort.y = FBM2d(make_float2(i, j), freq, 2.20f, 0.90f, seed, roughness) * strength;
	}
	else {
		distort.x = FBM2d_Simplex(make_float2(i, j), freq, 1.50f, 0.60f, seed, roughness) * strength;
		distort.y = FBM2d_Simplex(make_float2(i - seed, j + seed), freq, 1.50f, 0.60f, seed, roughness) * strength;
	}
	// Get offset value.
	// Add it to previous value and store the result in the output array.
	int i_offset, j_offset;
	i_offset = i + distort.x;
	j_offset = j + distort.y;
	i_offset %= width;
	j_offset %= height;
	if (i_offset < 0) {
		i_offset = width + i_offset;
	}
	if (j_offset < 0) {
		j_offset = height + j_offset;
	}

	out[(j * width) + i] = input[(j_offset * width) + i_offset];
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
