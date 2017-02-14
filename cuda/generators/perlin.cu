#include "perlin.cuh"

// Linear interpolation between given values.
__device__ float lerp(const float a, const float b, const float c) {
	float res;
	res = b - a;
	res = __fmaf_rn(res, c, a);
	return res;
}

// Applies ease-curve to "t", easing it towards integral values.
__device__ float ease(const float t) {
	float res;
	res = t * 6.0f;
	res = res - 15.0f;
	res = __fmaf_rn(res, t, 10.0f);

	// Three multiplications
	for (int i = 0; i < 3; ++i) {
		res *= t;
	}

	return res;
}

// Credits to original creator of this code: I only
// claim the (minor) credit of converting it to CUDA
// Author: Stefan Gustavson, 2003-2005
// Contact: stegu@itn.liu.se
// These methods get the gradient vectors in various dimensions.

__device__ float grad2(uchar hash, float2 p) {
	int h = hash & 7;
	float u = h < 4 ? p.x : p.y;
	float v = h < 4 ? p.y : p.x;
	float res = ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
	return res;
}

__device__ float grad3(uchar hash, float3 p) {
	int h = hash & 15;
	float u = h < 8 ? p.x : p.y;
	float v = h < 8 ? p.y : (h == 12 || h == 14 ? p.x : p.z);
	float res = ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
	return res;
}

__device__ float grad4(uchar hash, float4 p) {
	int h = hash & 31;
	float u = h < 24 ? p.x : p.y;
	float v = h < 15 ? p.y : p.z;
	float w = h < 8 ? p.z : p.w;
	float res = ((h & 1) ? -u : u) + ((h & 2) ? -v : v) + ((h & 4) ? -w : w);
	return res;
}

__device__ float perlin2d(float2 point, cudaTextureObject_t perm) {
	int ix0, iy0, ix1, iy1;
	float fx0, fy0, fx1, fy1;
	float s, t, nx0, nx1, n0, n1;

	// Integer part of point
	ix0 = static_cast<int>(floorf(point.x));
	iy0 = static_cast<int>(floorf(point.y));

	// Fractional part of point
	fx0 = frexpf(point.x, nullptr);
	fy0 = frexpf(point.y, nullptr);

	// Continue finding various components.
	fx1 = fx0 - 1.0f;
	fy1 = fx1 - 1.0f;
	
	// Wrap these components into 0-255 range of permutation table
	ix1 = (ix0 + 1) & 0xff;
	iy1 = (iy0 + 1) & 0xff;
	ix0 &= 0xff;
	iy0 &= 0xff;
	
	// Set t/s, used again later for lerp'ing to final value.
	t = ease(fy0);
	s = ease(fx0);

	// We feed uchar's into the gradient functions, 
	// and fetch them by reading from our given texture.
	uchar hash0;
	hash0 = tex1D<uchar>(perm, iy0);
	hash0 += tex1D<uchar>(perm, ix0);

	// Get first gradient point.
	nx0 = grad2(hash0, make_float2(fx0, fy0));

	// Second hash 
	uchar hash1;
	hash1 = tex1D<uchar>(perm, iy1);
	hash1 += tex1D<uchar>(perm, ix0);
	
	// Second gradient point.
	nx1 = grad2(hash1, make_float2(fx0, fy1));

	n0 = lerp(t, nx0, nx1);

	// Third hash
	uchar hash2;
	hash2 = tex1D<uchar>(perm, iy0);
	hash2 += tex1D<uchar>(perm, ix1);

	// Third gradient point.
	nx0 = grad2(hash2, make_float2(fx1, fy0));

	// Fourth hash
	uchar hash3;
	hash3 = tex1D<uchar>(perm, iy1);
	hash3 += tex1D<uchar>(perm, ix1);

	// Final gradient point
	nx1 = grad2(hash3, make_float2(fx1, fy1));

	n1 = lerp(t, nx0, nx1);

	// Get result: scale by the magic number, and lerp between
	// two lerped gradient points and "s".
	float result = 0.507f * (lerp(s, n0, n1));

	// Return final result.
	return result;
}


__global__ void perlin2D_Kernel(cudaSurfaceObject_t dest, cudaTextureObject_t perm_table, int width, int height, float2 origin) {
	// Get indices
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Don't do anything if out of bounds
	if (i > width || j > height) {
		return;
	}

	// Offset position by origin.
	float x, y;
	x = i + origin.x;
	y = j + origin.y;

	// Call perlin function
	float val = perlin2d(make_float2(x, y), perm_table);

	// Write val to surface.
	surf2Dwrite(val, dest, i * sizeof(float), j);
}

void PerlinLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
	// Setup dimensions of kernel launch. 
	// threads_per_block can vary
	dim3 block(threads_per_block, threads_per_block, 1);
	dim3 grid((width - 1) / block.x + 1, (height - 1) / block.y + 1, 1);
	perlin2D_Kernel<<<block, grid>>>(out, perm, width, height, origin);

	// Check for succesfull kernel launch
	cudaAssert(cudaGetLastError());

	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());

	// If this completes, kernel is done and "output" contains correct data.
}