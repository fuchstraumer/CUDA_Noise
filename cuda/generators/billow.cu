#include "billow.cuh"
#include "..\..\cpp\modules\generators\Billow.h"


__device__ float billow2D(float2 point, cudaTextureObject_t perm, cudaTextureObject_t grad, float freq, float lacun, float persist, int init_seed, int octaves) {
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
		val = perlin2d(perm, grad, point, seed);
		val = fabsf(val);
		result += val * amplitude;
		// Modify vars for next octave.
		freq *= lacun;
		point.x *= freq;
		point.y *= freq;
		amplitude *= persist;
	}
	float tmp = result / 100.0f;
	// * // 
	return tmp;
}

__device__ float billow2D_S(float2 point, cudaTextureObject_t perm, cudaTextureObject_t grad, float freq, float lacun, float persist, int init_seed, int octaves) {
	float result = 0.0f;
	float amplitude = 1.0f;
	float val;
	// Scale starting point by frequency.
	point.x = point.x * freq;
	point.y = point.y * freq;
	// Use loop for fractal octave bit
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		val = simplex2d(perm, grad, point, seed);
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

__global__ void Billow2DKernel(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
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
	float val = billow2D(p, perm, grad, freq, lacun, persist, seed, octaves);

	// Write val to the surface
	surf2Dwrite(val, out, i * sizeof(float), j);
}

__global__ void AddKernel(cudaSurfaceObject_t out, cudaTextureObject_t in, float val) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	float previous_value;
	previous_value = tex2D<float>(in, i, j);

	float final_value;
	final_value = previous_value + val;

	surf2Dwrite(final_value, out, i * sizeof(float), j);

}

__global__ void Billow2DKernelSimplex(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves){
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
	float val = billow2D_S(p, perm, grad, freq, lacun, persist, seed, octaves);

	// Write val to the surface
	surf2Dwrite(val, out, i * sizeof(float), j);
}


void BillowLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif // CUDA_TIMING_TESTS

	// Setup dimensions of kernel launch. 
	
	// Use occupancy calc to find optimal sizes.
	int blockSize, minGridSize;
#ifdef CUDA_TIMING_TESTS
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)Billow2DKernel, 0, 0);
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	Billow2DKernel<<<block,grid>>>(out, perm, grad, width, height, origin, freq, lacun, persist, seed, octaves);
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

void BillowSimplexLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves){
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif // CUDA_TIMING_TESTS

	// Setup dimensions of kernel launch. 

	// Use occupancy calc to find optimal sizes.
	int blockSize, minGridSize;
#ifdef CUDA_TIMING_TESTS
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)Billow2DKernel, 0, 0);
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	Billow2DKernelSimplex<<<block, grid>>>(out, perm, grad, width, height, origin, freq, lacun, persist, seed, octaves);
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

