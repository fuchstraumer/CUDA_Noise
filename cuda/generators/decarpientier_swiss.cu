#include "decarpientier_swiss.cuh"

__device__ float d_swiss_simplex(float2 point, const float freq, const float lacun, const float persist, const int octaves) {
	// Will be incremented upon.
	float result = 0.0f;
	float amplitude = 1.0f;
	// Scale point by freq
	point.x = point.x * freq;
	point.y = point.y * freq;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	float warp = 0.1f;
	float dx_sum = 0.0f, dy_sum = 0.0f;
	for (size_t i = 0; i < octaves; ++i) {
		float2 dx_dy; 
		volatile float n = simplex2d(point, &dx_dy);
		result += (1.0f - fabsf(n)) * amplitude;
		dx_sum += amplitude * dx_dy.x * -n;
		dy_sum += amplitude * dx_dy.y * -n;
		// Modify vars for next octave.
		point.x *= lacun;
		point.y *= lacun;
		point.x += (warp * dx_sum);
		point.y += (warp * dy_sum);
		amplitude *= persist * __saturatef(result);
	}

	return result;
}

__device__ float d_swiss_perlin(float px, float py, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float amplitude = 1.0f;
	// Scale point by freq
	float2 point = make_float2(px * freq, py * freq);
	float warp = 0.1f;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	float result = 0.0f;
	float dx_sum = 0.0f, dy_sum = 0.0f;
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		float n = perlin2d(point, seed);
		result += (1.0f - fabsf(n)) * amplitude;
		dx_sum += amplitude * perlin2d_dx(point, seed) * -n;
		dy_sum += amplitude * perlin2d_dy(point, seed) * -n;
		// Modify vars for next octave.
		point.x += (warp * dx_sum);
		point.y += (warp * dy_sum);
		point.x *= lacun;
		point.y *= lacun;
		amplitude *= persist * __saturatef(result);
	}
	return result;
}

__global__ void d_swiss_kernel(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < width && j < height) {
		// Call noise function
		float val;
		switch (noise_type) {
		case noise_t::PERLIN:
			val = d_swiss_perlin(origin.x + i, origin.y + j, freq, lacun, persist, seed, octaves);
			break;
		case noise_t::SIMPLEX:
			val = d_swiss_simplex(make_float2(origin.x + i, origin.y + j), freq, lacun, persist, octaves);
			break;
		}
		// Write val to the surface
		surf2Dwrite(val, out, i * sizeof(float), j);
	}
}

void DecarpientierSwissLauncher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, d_swiss_kernel);
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	d_swiss_kernel<<<numBlocks, threadsPerBlock>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
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