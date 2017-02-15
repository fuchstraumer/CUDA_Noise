#include "billow.cuh"


__device__ float billow2D(float2 point, cudaTextureObject_t perm, cudaTextureObject_t grad, float freq, float lacun, float persist, int init_seed, int octaves) {
	// Will be incremented upon.
	float result = 0.0f;
	float val = 0.0f;
	float curPersistence = 1.0f;
	// Scale point by freq
	point.x *= freq;
	point.y *= freq;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		val = perlin2d(perm, grad, point, seed);
		val = 2.0f * fabsf(val) - 1.0f;
		result += val * curPersistence;
		// Modify vars for next octave.
		point.x *= lacun;
		point.y *= lacun;
		curPersistence *= persist;
	}

	result += 0.50f;
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


void BillowLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	// Setup dimensions of kernel launch. 
	
	// Use occupancy calc to find optimal sizes.
	int blockSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)Billow2DKernel, 0, 0);
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	Billow2DKernel<<<block,grid>>>(out, perm, grad, width, height, origin, freq, lacun, persist, seed, octaves);

	// Check for succesfull kernel launch
	cudaAssert(cudaGetLastError());

	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());

	// If this completes, kernel is done and "output" contains correct data.
}