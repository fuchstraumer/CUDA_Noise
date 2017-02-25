#include "ridged_multi.cuh"

__device__ float Ridged2D_Simplex(float2 point, float freq, float lacun, float persist, int init_seed, int octaves) {
	float result = 0.0f;
	float amplitude = 1.0f;
	float val;
	// Scale starting point by frequency.
	point.x = point.x * freq;
	point.y = point.y * freq;
	// Use loop for fractal octave bit
	for (size_t i = 0; i < octaves; ++i) {
		val = simplex2d(point, freq);
		val = fabsf(val);
		result += val * amplitude;
		freq *= lacun;
		point.x *= freq;
		point.y *= freq;
		amplitude *= persist;
	}
	//result /= 100.0f;
	return result;
}

__device__ float Ridged2D(float2 point, float freq, float lacun, float persist, int init_seed, int octaves) {
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
		val = fabsf(val);
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

__global__ void Ridged2DKernel(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	// Get offset pos.
	float x, y;
	x = i + origin.x;
	y = j + origin.y;
	float2 p = make_float2(x, y);
	// Call ridged function
	float val;
	switch (noise_type) {
	case(noise_t::PERLIN): {
		val = Ridged2D(p, freq, lacun, persist, seed, octaves);
	}
	case(noise_t::SIMPLEX): {
		val = Ridged2D_Simplex(p, freq, lacun, persist, seed, octaves);
	}
	}
	// Write val to the surface
	surf2Dwrite(val, out, i * sizeof(float), j);
}

void RidgedMultiLauncher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	Ridged2DKernel<<<numBlocks, threadsPerBlock>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
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