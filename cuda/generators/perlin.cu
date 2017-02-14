#include "perlin.cuh"

#ifndef HALF_PRECISION_SUPPORT

__device__ float perlin2d(cudaTextureObject_t perm_tex, cudaTextureObject_t grad_tex, float2 point, int seed) {
	
	// Calculate 2D integer coordinates and fractional component 
	float2 i = make_float2(floorf(point.x), floorf(point.y));
	float2 f = make_float2(point.x - i.x, point.y - i.y);

	// Get weights.
	float2 w;
	w.x = f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f);
	w.y = f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f);
	float4 w4 = make_float4(1.0f, w.x, w.y, w.x * w.y);

	// Get four randomly permutated indices from the noise lattice nearest "point"
	// and offset them by the seed.
	float4 perm = tex2D<float4>(perm_tex, i.x / 256, i.y / 256);
	perm.x += seed;
	perm.y += seed;
	perm.z += seed;
	perm.w += seed;

	// Permute the fourst indices again and get the 2D gradient for each of
	// the four new coord-seed pairs.
	float4 g1, g2;
	g1 = tex2D<float4>(grad_tex, perm.x, perm.y);
	g1 *= make_float4(2.0f, 2.0f, 2.0f, 2.0f);
	g1.x *= 2.0f;
	g1.x -= 1.0f;
	g1.y *= 2.0f;
	g1.y -= 1.0f;
	g1.z *= 2.0f;
	g1.z -= 1.0f;
	g1.w *= 2.0f;
	g1.z -= 1.0f;
}

__global__ void perlin2d_Kernel(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int octaves) {

}


#else

// TODO: Removed these until its re-implemented. Need to figure out how it works with textures.

#endif // !HALF_PRECISION_SUPPORT

void PerlinLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	// Use occupancy calc to find optimal sizes.

	int blockSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)perlin2d_Kernel, 0, 0);
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	if (grid.x > static_cast<unsigned int>(minGridSize) || grid.y > static_cast<unsigned int>(minGridSize)) {
		throw("Grid sizing error.");
	}
	// 32-bit kernel.
	perlin2D_Kernel<<<block, grid>>>(out, perm, width, height, origin, freq, lacun, persist, octaves);

	// Check for kernel launch errors
	cudaAssert(cudaGetLastError());

	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());

	// If this completes, kernel is done and "output" contains correct data.
}