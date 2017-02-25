#include "FBM.cuh"
#include "..\..\cpp\modules\generators\FBM.h"

__device__ float FBM2d_Simplex(float2 point, float freq, float lacun, float persist, int init_seed, float octaves) {
	// Will be incremented upon.
	float result = 0.0f;
	float amplitude = 1.0f;
	float val;
	// Scale point by freq
	point.x = point.x * freq;
	point.y = point.y * freq;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		val = simplex2d(point, freq);
		result += val * amplitude;
		// Modify vars for next octave.
		freq *= lacun;
		point.x *= freq;
		point.y *= freq;
		amplitude *= persist;
	}
	// float tmp = result / 100.0f;
	// * // 
	return result;
}

__device__ float FBM2d(float2 point, float freq, float lacun, float persist, int init_seed, float octaves) {
	// Will be incremented upon.
	float result = 0.0f;
	float amplitude = 1.0f;
	float val;
	// Scale point by freq
	point.x = point.x * freq;
	point.y = point.y * freq;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		val = perlin2d(point, freq, seed);
		result += val * amplitude;
		// Modify vars for next octave.
		freq *= lacun;
		point.x *= freq;
		point.y *= freq;
		amplitude *= persist;
	}
	// float tmp = result / 100.0f;
	// * // 
	return result;
}

__global__ void FBM2DKernel(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= width || j >= height) {
		return;
	}

	float x, y;
	x = i + origin.x;
	y = j + origin.y;
	float2 p = make_float2(x, y);
	// Call billow function
	float val;
	switch (noise_type) {
		case(noise_t::PERLIN): {
			val = FBM2d(p, freq, lacun, persist, seed, octaves);
		}
		case(noise_t::SIMPLEX): {
			val = FBM2d_Simplex(p, freq, lacun, persist, seed, octaves);
		}
	}

	// Write val to the surface
	surf2Dwrite(val, out, i * sizeof(float), j);
}

void FBM_Launcher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves){
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	FBM2DKernel<<<numBlocks, threadsPerBlock>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
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




