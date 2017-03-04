#include "Turbulence.cuh"

/*
	
	Turbulence process:

	1. Get current pixel position.
	2. Offset pixel position using turbulence device functions.
	3. Before reading with pixel position, make sure its in range of surfaceObject
	4. Read from input with offset position, and use this value to set the (i,j) position to this new value in output.

*/

__global__ void TurbulenceKernel(cudaSurfaceObject_t out, cudaSurfaceObject_t input, const int width, const int height, const noise_t noise_type, const int roughness, const int seed, const float strength) {
	// Get current pixel.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	// Return if out of bounds.
	if (i < width && j < height) {
		// Position that will be displaced
		float2 displace;
		switch (noise_type) {
			case(noise_t::PERLIN): {
				displace.x = i + perlin2d(i + (12414.0f / 65536.0f), j + (65124.0f / 65536.0f), seed, nullptr) * strength;
				displace.y = j + perlin2d(i + (26519.0f / 65536.0f), j + (18128.0f / 65536.0f), seed, nullptr) * strength;
				break;
			}
			case(noise_t::SIMPLEX): {
				displace.x = i + simplex2d(i, j, seed, nullptr) * strength;
				displace.y = j + simplex2d(i, j, seed, nullptr) * strength;
				break;
			}
		}

		// Get displace into proper range
		

		float offset_val = perlin2d(displace.x, displace.y, seed, nullptr);
		// Write new offset value.
		surf2Dwrite(offset_val, out, i * sizeof(float), j);
	}

}

void TurbulenceLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t input, const int width, const int height, const noise_t noise_type, const int roughness, const int seed, const float strength){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	TurbulenceKernel<<<numBlocks, threadsPerBlock>>>(out, input, width, height, noise_type, roughness, seed, strength);
	// Check for succesfull kernel launch
	cudaAssert(cudaGetLastError());
	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

	// If this completes, kernel is done and "output" contains correct data.
}
