#include "Turbulence.cuh"
#include "..\..\cpp\modules\modifiers\Turbulence.h"

/*
	
	Turbulence process:

	1. Get current pixel position.
	2. Offset pixel position using turbulence device functions.
	3. Before reading with pixel position, make sure its in range of surfaceObject
	4. Read from input with offset position, and use this value to set the (i,j) position to this new value in output.

*/

__global__ void TurbulenceKernel(cudaSurfaceObject_t out, cudaSurfaceObject_t input, int width, int height, noise_t noise_type, int roughness, int seed, float strength) {
	// Get current pixel.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	// Return if out of bounds.
	if (i >= width || j >= height) {
		return;
	}
	
	// Position that will be displaced
	float2 pos = make_float2(i, j);
	float2 displace = make_float2(i, j);
	switch (noise_type) {
		case(noise_t::PERLIN): {
			displace.x += FBM2d(pos, 1.0f, 2.0f, 0.50f, seed, roughness) * strength;
			displace.y += FBM2d(pos, 1.0f, 2.0f, 0.50f, seed, roughness) * strength;
		}
		case(noise_t::SIMPLEX): {
			displace.x += FBM2d_Simplex(pos, 1.0f, 2.0f, 0.50f, seed, roughness) * strength;
			displace.y += FBM2d_Simplex(pos, 1.0f, 2.0f, 0.50f, seed, roughness) * strength;
		}
	}

	// Get displace into proper range
	float2 new_pos;
	new_pos.x = (int)floorf(displace.x) % width;
	new_pos.y = (int)floorf(displace.y) % height;

	float offset_val;
	surf2Dread(&offset_val, input, new_pos.x * sizeof(float), new_pos.y);

	// Write new offset value.
	surf2Dwrite(offset_val, out, i * sizeof(float), j);
}

void TurbulenceLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t input, int width, int height, noise_t noise_type, int roughness, int seed, float strength){

#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	TurbulenceKernel<<<numBlocks, threadsPerBlock>>>(out, input, width, height, noise_type, roughness, seed, strength);
	// Check for succesfull kernel launch
	cudaAssert(cudaGetLastError());
	cudaAssert(cudaThreadSynchronize());
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
