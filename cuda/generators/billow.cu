#include "billow.cuh"



__device__ float billow2D_Simplex(float2 point, float freq, float lacun, float persist, int octaves) {
	float result = 0.0f;
	float amplitude = 1.0f;
	// Scale starting point by frequency.
	point.x = point.x * freq;
	point.y = point.y * freq;
	// Use loop for fractal octave bit
	for (size_t i = 0; i < octaves; ++i) {
		result += fabsf(simplex2d(point, nullptr)) * amplitude;
		point.x *= lacun;
		point.y *= lacun;
		amplitude *= persist;
	}
	//result /= 100.0f;
	return result;
}

__device__ float billow2D(float2 point, float freq, float lacun, float persist, int init_seed, int octaves) {
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
		result += fabsf(perlin2d(point, seed)) * amplitude;
		// Modify vars for next octave.
		point.x *= lacun;
		point.y *= lacun;
		amplitude *= persist;
	}
	// float tmp = result / 100.0f;
	// * // 
	return result;
}



__global__ void Billow2DKernel(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < width && j < height) {
		float x, y;
		x = i + origin.x;
		y = j + origin.y;
		float2 p = make_float2(x, y);
		// Call billow function
		float val;
		switch (noise_type) {
			case(noise_t::PERLIN): {
				val = billow2D(p, freq, lacun, persist, seed, octaves);
				break;
			}
			case(noise_t::SIMPLEX): {
				val = billow2D_Simplex(p, freq, lacun, persist, octaves);
				break;
			}
		}
		// Write val to the surface
		surf2Dwrite(val, out, i * sizeof(float), j);
	}

	
}




void BillowLauncher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {

#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	Billow2DKernel<<<numBlocks,threadsPerBlock>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
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

