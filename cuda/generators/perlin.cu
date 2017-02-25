#include "perlin.cuh"

// Hash function for seed.
__device__ unsigned int hash(unsigned int seed){
	seed = (seed + 0x7ed55d16) + (seed << 12);
	seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
	seed = (seed + 0x165667b1) + (seed << 5);
	seed = (seed + 0xd3a2646c) ^ (seed << 9);
	seed = (seed + 0xfd7046c5) + (seed << 3);
	seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

	return seed;
}

// Random unsigned int for a grid coordinate [0, MAXUINT]
__device__ unsigned int randomIntGrid(int x, int y, int z, int seed = 0){
	return hash((unsigned int)(x * 1723 + y * 93241 + z * 149812 + 3824 + seed));
}

// Helper functions for perlin noise

__device__ float lerp(float a, float b, float ratio){
	return a * (1.0f - ratio) + b * ratio;
}

__device__ float cubic(float p0, float p1, float p2, float p3, float x){
	return p1 + 0.5 * x * (p2 - p0 + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + x * (3.0 * (p1 - p2) + p3 - p0)));
}

__device__ float grad(int hash, float x, float y, float z){
	switch (hash & 0xF)
	{
	case 0x0: return  x + y;
	case 0x1: return -x + y;
	case 0x2: return  x - y;
	case 0x3: return -x - y;
	case 0x4: return  x + z;
	case 0x5: return -x + z;
	case 0x6: return  x - z;
	case 0x7: return -x - z;
	case 0x8: return  y + z;
	case 0x9: return -y + z;
	case 0xA: return  y - z;
	case 0xB: return -y - z;
	case 0xC: return  y + x;
	case 0xD: return -y + z;
	case 0xE: return  y - x;
	case 0xF: return -y - z;
	default: return 0; // never happens
	}
}

__device__ float fade(float t)
{
	// Fade function as defined by Ken Perlin.  This eases coordinate values
	// so that they will ease towards integral values.  This ends up smoothing
	// the final output.
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);         // 6t^5 - 15t^4 + 10t^3
}

__device__ float perlin2d_tex(cudaTextureObject_t perm_tex, cudaTextureObject_t grad_tex, float2 point, int seed) {

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
	uchar4 tmp = tex2D<uchar4>(perm_tex, i.x + 0.50f, i.y + 0.50f);
	float4 perm = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
	perm = perm + seed;

	// Permute the fourst indices again and get the 2D gradient for each of
	// the four new coord-seed pairs.
	float4 gLeft, gRight;
	uchar4 tmp0 = tex2D<uchar4>(grad_tex, perm.x + 0.50f, perm.y + 0.50f);
	gLeft = make_float4(tmp0.x, tmp0.y, tmp0.z, tmp0.w);
	gLeft = gLeft * 2.0f;
	gLeft = gLeft - 1.0f;
	uchar4 tmp1 = tex2D<uchar4>(grad_tex, perm.z + 0.50f, perm.w + 0.50f);
	gRight = make_float4(tmp1.x, tmp1.y, tmp1.z, tmp1.w);
	gRight = gRight * 2.0f;
	gRight = gRight - 1.0f;

	// Evaluate gradients at four lattice points.
	float nLeftTop = dot(make_float2(gLeft.x, gLeft.y), f);
	float nRightTop = dot(make_float2(gRight.x, gRight.y), f + make_float2(-1.0f, 0.0f));
	float nLeftBottom = dot(make_float2(gLeft.z, gLeft.w), f + make_float2(0.0f, -1.0f));
	float nRightBottom = dot(make_float2(gRight.z, gRight.w), f + make_float2(-1.0f, -1.0f));

	// Blend gradients.
	float4 gradientBlend = make_float4(nLeftTop, nRightTop - nLeftTop, nLeftBottom - nLeftTop,
		nLeftTop - nRightTop - nLeftBottom + nRightBottom);
	float n = dot(gradientBlend, w4);

	// Return value.
	return (n * 1.5f) / (2.5f);
	//return n * 1.530734f;
}

__device__ float perlin2d(float2 position, float scale, int seed){
	return perlin3d(make_float3(position.x, position.y, hash(seed)), scale, seed);
}

__device__ float perlin3d(float3 pos, float scale, int seed){
	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

	// zero corner integer position
	int ix = (int)floorf(pos.x);
	int iy = (int)floorf(pos.y);
	int iz = (int)floorf(pos.z);

	// current position within unit cube
	pos.x -= floorf(pos.x);
	pos.y -= floorf(pos.y);
	pos.z -= floorf(pos.z);

	// adjust for fade
	float u = fade(pos.x);
	float v = fade(pos.y);
	float w = fade(pos.z);

	// influence values
	float i000 = grad(randomIntGrid(ix, iy, iz, seed), pos.x, pos.y, pos.z);
	float i100 = grad(randomIntGrid(ix + 1, iy, iz, seed), pos.x - 1.0f, pos.y, pos.z);
	float i010 = grad(randomIntGrid(ix, iy + 1, iz, seed), pos.x, pos.y - 1.0f, pos.z);
	float i110 = grad(randomIntGrid(ix + 1, iy + 1, iz, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z);
	float i001 = grad(randomIntGrid(ix, iy, iz + 1, seed), pos.x, pos.y, pos.z - 1.0f);
	float i101 = grad(randomIntGrid(ix + 1, iy, iz + 1, seed), pos.x - 1.0f, pos.y, pos.z - 1.0f);
	float i011 = grad(randomIntGrid(ix, iy + 1, iz + 1, seed), pos.x, pos.y - 1.0f, pos.z - 1.0f);
	float i111 = grad(randomIntGrid(ix + 1, iy + 1, iz + 1, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);

	// interpolation
	float x00 = lerp(i000, i100, u);
	float x10 = lerp(i010, i110, u);
	float x01 = lerp(i001, i101, u);
	float x11 = lerp(i011, i111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	float avg = lerp(y0, y1, w);

	return avg;
}

