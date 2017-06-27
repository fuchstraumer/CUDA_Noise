#include "ridged_multi.cuh"

__device__ float Ridged2D_Simplex(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float result = 0.0f;
	float amplitude = 1.0f;
	// Scale starting point by frequency.
	point.x = point.x * freq;
	point.y = point.y * freq;
	// Use loop for fractal octave bit
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		result += (1.0f - fabsf(simplex2d(point.x, point.y, seed, nullptr))) * amplitude;
		point.x *= lacun;
		point.y *= lacun;
		amplitude *= persist;
	}
	return result;
}

__device__ float Ridged2D(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	// Will be incremented upon.
	float result = 0.0f;
	float amplitude = 1.0f;
	// Scale point by freq
	point.x = point.x * freq;
	point.y = point.y * freq;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		result += (1.0f - fabsf(perlin2d(point.x, point.y, seed, nullptr)))* amplitude;
		// Modify vars for next octave.
		point.x *= lacun;
		point.y *= lacun;
		amplitude *= persist;
	}
	return result;
}

__device__ float Ridged3D(float3 p, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float amplitude = 1.0f;
	float result = 0.0f;
	p *= freq;
	for (int i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		result += (1.0f - fabsf(simplex3d(p.x, p.y, p.z, seed, nullptr))) * amplitude;
		p *= lacun;
		amplitude *= persist;
	}
	return result;
}

__global__ void Ridged2DKernel(float* out, int width, int height, cnoise::noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < width && j < height) {
		// Get offset pos.
		float2 p = make_float2(i + origin.x, j + origin.y);
		// Call ridged function
		float val;
		switch (noise_type) {
			case(cnoise::noise_t::PERLIN): {
				val = Ridged2D(p, freq, lacun, persist, seed, octaves);
				break;
			}
			case(cnoise::noise_t::SIMPLEX): {
				val = Ridged2D_Simplex(p, freq, lacun, persist, seed, octaves);
				break;
			}
		}
		// Write val to the surface
		out[(j * width) + i] = val;
	}
	
}

__global__ void Ridged3DKernel(cnoise::Point* coords, const int width, const int height, const float freq, const float lacun, const float persist, const int seed, const int octaves) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	coords[i + (j * width)].Value = Ridged3D(coords[i + (j * width)].Position, freq, lacun, persist, seed, octaves);
}

void RidgedMultiLauncher(float* out, int width, int height, cnoise::noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, Ridged2DKernel);
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	Ridged2DKernel<<<numBlocks, threadsPerBlock>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
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
	printf("Ridged-multi Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

	// If this completes, kernel is done and "output" contains correct data.
}

void RidgedMultiLauncher3D(cnoise::Point* coords, const int width, const int height, const float freq, const float lacun, const float persist, const int seed, const int octaves){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, Ridged2DKernel);
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	Ridged3DKernel<<<numBlocks, threadsPerBlock >>>(coords, width, height, freq, lacun, persist, seed, octaves);
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
