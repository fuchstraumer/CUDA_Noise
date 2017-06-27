#include "FBM.cuh"

__device__ float FBM2d_Simplex(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
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
		result += simplex2d(point.x, point.y, seed, nullptr) * amplitude;
		// Modify vars for next octave.
		point.x *= lacun;
		point.y *= lacun;
		amplitude *= persist;
	}

	return result;
}

__device__ float FBM2d(float2 point, const float freq, const float lacun, 
const float persist, const int init_seed, const int octaves) {
	float amplitude = 1.0f;
	// Scale point by freq
	point.x = point.x * freq;
	point.y = point.y * freq;

	// Accumulate result from each octave into this variable.
	float result = 0.0f;
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		result += perlin2d(point.x, point.y, seed, nullptr) * amplitude;
		// Increases frequency of successive octaves.
		point.x *= lacun;
		point.y *= lacun;
		// Decreases amplitude of successive octaves.
		amplitude *= persist;
	}

	return result;
}

__global__ void FBM2DKernel(float* out, int width, int height, cnoise::noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= width && j >= height) {
		return;
	}

	float2 p = make_float2(origin.x + i, origin.y + j);
	// Call billow function
	float val;
	switch (noise_type) {
		case(cnoise::noise_t::PERLIN): {
			val = FBM2d(p, freq, lacun, persist, seed, octaves);
			break;
		}
		case(cnoise::noise_t::SIMPLEX): {
			val = FBM2d_Simplex(p, freq, lacun, persist, seed, octaves);
			break;
		}
	}

	// Write val to the surface
	out[(j * width) + i] = val;
}

__device__ float fbm_3d(float3 p, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float amplitude = 1.0f;
	float result = 0.0f;
	p *= freq;
	for (int i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xfffffff;
		result += simplex3d(p.x, p.y, p.z, seed, nullptr) * amplitude;
		p *= lacun;
		amplitude *= persist;
	}
	return result;
}

__global__ void FBM3DKernel(cnoise::Point* coords, const int width, const int height, const float freq, const float lacun, const float persist, const int seed, const int octaves) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	coords[i + (j * width)].Value = fbm_3d(coords[i + (j * width)].Position, freq, lacun, persist, seed, octaves);
}

void FBM_Launcher(float* out, int width, int height, cnoise::noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves){
#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	FBM2DKernel<<<numBlocks, threadsPerBlock>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
	// Check for succesfull kernel launch
	cudaError_t err;
	err = cudaGetLastError();
	cudaAssert(err);
	// Synchronize device
	err = cudaDeviceSynchronize();
	cudaAssert(err);

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("FBM Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

	// If this completes, kernel is done and "output" contains correct data.
}

void FBM_Launcher3D(cnoise::Point* points, const int width, const int height, const float freq, const float lacun, const float persist, const int seed, const int octaves){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	dim3 threadsPerBlock(8, 8, 1);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);
	FBM3DKernel<<<numBlocks, threadsPerBlock >>>(points, width, height, freq, lacun, persist, seed, octaves);
	// Check for succesfull kernel launch
	cudaError_t err;
	err = cudaGetLastError();
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




