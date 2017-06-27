#include "decarpientier_swiss.cuh"


__device__ float d_swiss_simplex(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	// Will be incremented upon.
	float result = 0.0f;
	float amplitude = 1.0f;
	// Scale point by freq
	point.x = point.x * freq;
	point.y = point.y * freq;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	float warp = 0.02f;
	float dx_sum = 0.0f, dy_sum = 0.0f;
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		float2 dx_dy;
		float n = simplex2d(point.x, point.y, seed, &dx_dy);
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
	float warp = 0.01f;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	float result = 0.0f;
	float dx_sum = 0.0f, dy_sum = 0.0f;
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		float2 dx_dy;
		float n = perlin2d(point.x, point.y, seed, &dx_dy);
		result += (1.0f - fabsf(n)) * amplitude;
		dx_sum += amplitude * dx_dy.x * -n;
		dy_sum += amplitude * dx_dy.y * -n;
		point.x *= lacun;
		point.y *= lacun;
		point.x += (warp * dx_sum);
		point.y += (warp * dy_sum);
		// Modify vars for next octave.
		amplitude *= persist * __saturatef(result);
	}
	return result;
}

__device__ float d_swiss_3d(float3 position, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float amplitude = 1.0f;
	position *= freq;
	float warp = 0.01f;
	float result = 0.0f;
	float3 dsum = make_float3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		float3 d;
		float n = simplex3d(position.x, position.y, position.z, seed, &d);
		result += (1.0f - fabsf(n)) * amplitude;
		d.x += amplitude * d.x * -n;
		d.y += amplitude * d.y * -n;
		d.z += amplitude * d.z * -n;
		position *= lacun;
		position += (warp * dsum);
		amplitude *= persist * __saturatef(result);
	}
	return result;
}

__global__ void d_swiss_kernel(float* out, int width, int height, cnoise::noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < width && j < height) {
		// Call noise function
		float val;
		switch (noise_type) {
		case cnoise::noise_t::PERLIN:
			val = d_swiss_perlin(origin.x + i, origin.y + j, freq, lacun, persist, seed, octaves);
			break;
		case cnoise::noise_t::SIMPLEX:
			val = d_swiss_simplex(make_float2(origin.x + i, origin.y + j), freq, lacun, persist, seed, octaves);
			break;
		}
		// Write val to the surface
		out[(j * width) + i] = val;
	}
}

__global__ void d_swiss_kernel_3d(cnoise::Point* coords, int width, int height, const float freq, const float lacun, const float persist, const int seed, const int octaves) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	// Get value in 2D array at position of coordinate at position i,j in array
	coords[i + (j * width)].Value = d_swiss_3d(coords[i + (j * width)].Position, freq, lacun, persist, seed, octaves);
}

void DecarpientierSwissLauncher(float* out, int width, int height, cnoise::noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	d_swiss_kernel<<<numBlocks, threadsPerBlock>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
	cudaError_t err = cudaGetLastError();
	cudaAssert(err);
	// Synchronize device
	err = cudaDeviceSynchronize();
	cudaAssert(err);

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Decarpientier_swiss Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

	// If this completes, kernel is done and "output" contains correct data.
}

void DecarpientierSwissLauncher3D(cnoise::Point* coords, int width, int height, float freq, float lacun, float persist, int seed, int octaves){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	d_swiss_kernel_3d<<<numBlocks, threadsPerBlock>>>(coords, width, height, freq, lacun, persist, seed, octaves);
	cudaError_t err = cudaGetLastError();
	cudaAssert(err);
	// Synchronize device
	err = cudaDeviceSynchronize();
	cudaAssert(err);

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

}