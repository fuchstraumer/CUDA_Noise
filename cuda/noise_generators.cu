#include "noise_generators.cuh"

/*

Hashing methods from accidental noise

These have the tremendous benefit of letting us avoid
LUTs!

*/

// Hashing constants.
__device__ __constant__ uint FNV_32_PRIME = 0x01000193;
__device__ __constant__ uint FNV_32_INIT = 2166136261;
__device__ __constant__ uint FNV_MASK_8 = (1 << 8) - 1;

__device__ __constant__ int grad_2d_lut[256][2] =
{
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 },
	{ 0,1 },
	{ 0,-1 },
	{ 1,0 },
	{ -1,0 }
};


inline __device__ uint fnv_32_a_buf(const void* buf, const uint len) {
	uint hval = FNV_32_INIT;
	uint *bp = (uint*)buf;
	uint *be = bp + len;
	while (bp < be) {
		hval ^= (*bp++);
		hval *= FNV_32_PRIME;
	}
	return hval;
}

inline __device__ uchar xor_fold_hash(const uint hash) {
	return (uchar)((hash >> 8) ^ (hash & FNV_MASK_8));
}

inline __device__ uint hash_2d(const int x, const int y, const int seed) {
	uint d[3] = { (uint)x, (uint)y, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, 3));
}

inline __device__ uint hash_3d(const int x, const int y, const int z, const int seed) {
	uint d[4] = { (uint)x, (uint)y, (uint)z, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, 4));
}

inline __device__ uint hash_float_2d(const float x, const float y, const int seed) {
	uint d[3] = { (uint)x, (uint)y, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, sizeof(float) * 3 / sizeof(uint)));
}

inline __device__ uint hash_float_3d(const float x, const float y, const float z, const int seed) {
	uint d[4] = { (uint)x, (uint)y, (uint)z, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, sizeof(float) * 4 / sizeof(uint)));
}

// 5th degree easing/interp curve from libnoise.
__device__ float sCurve5(const float a) {
	return (6.0f * a * a * a * a * a) - (15.0f * a * a * a * a) + (10.0f * a * a * a);
}

__device__ float perlin2d(const float px, const float py, const int seed, float2 * deriv){
	volatile int ix0, iy0;
	ix0 = floorf(px);
	iy0 = floorf(py);

	float fx0, fy0, x0, y0;
	x0 = px - ix0;
	y0 = py - iy0;
	fx0 = sCurve5(x0);
	fy0 = sCurve5(y0);

	// Get four hashes
	volatile uint h0, h1, h2, h3;
	h0 = hash_2d(ix0, iy0, seed);
	h1 = hash_2d(ix0, iy0 + 1, seed);
	h2 = hash_2d(ix0 + 1, iy0, seed);
	h3 = hash_2d(ix0 + 1, iy0 + 1, seed);

	// Get four gradient sets.
	float4 g1, g2;
	g1 = make_float4(grad_2d_lut[h0][0], grad_2d_lut[h0][1], grad_2d_lut[h1][0], grad_2d_lut[h1][1]);
	g2 = make_float4(grad_2d_lut[h2][0], grad_2d_lut[h2][1], grad_2d_lut[h3][0], grad_2d_lut[h3][1]);

	// Get dot products of gradients and positions.
	float a, b, c, d;
	a = g1.x*x0 + g1.y*y0;
	b = g2.x*(x0 - 1.0f) + g2.y*y0;
	c = g1.z*x0 + g1.w*(y0 - 1.0f);
	d = g2.z*(x0 - 1.0f) + g2.w*(y0 - 1.0f);

	// Get gradients
	float4 gradients = make_float4(a, b - a, c - a, a - b - c + d);
	float n = dot(make_float4(1.0f, fx0, fy0, fx0 * fy0), gradients);

	// Now get derivative
	if (deriv != nullptr) {
		float dx = fx0 * fx0 * (fx0 * (30.0f * fx0 - 60.0f) + 30.0f);
		float dy = fy0 * fy0 * (fy0 * (30.0f * fy0 - 60.0f) + 30.0f);
		float dwx = fx0 * fx0 * fx0 * (fx0 * (fx0 * 36.0f - 75.0f) + 40.0f);
		float dwy = fy0 * fy0 * fy0 * (fy0 * (fy0 * 36.0f - 75.0f) + 40.0f);

		deriv->x =
			(g1.x + (g1.z - g1.x)*fy0) + ((g2.y - g1.y)*y0 - g2.x +
			((g1.y - g2.y - g1.w + g2.w)*y0 + g2.x + g1.w - g2.z - g2.w)*fy0)*
			dx + ((g2.x - g1.x) + (g1.x - g2.x - g1.z + g2.z)*fy0)*dwx;
		deriv->y = 
			(g1.y + (g2.y - g1.y)*fx0) + ((g1.z - g1.x)*x0 - g1.w + ((g1.x -
				g2.x - g1.z + g2.z)*x0 + g2.x + g1.w - g2.z - g2.w)*fx0)*dy +
				((g1.w - g1.y) + (g1.y - g2.y - g1.w + g2.w)*fx0)*dwy;
	}

	return (n * 1.50f);
	
}

__device__ float simplex2d(const float px, const float py, const int seed, float2 * deriv){
	// Contributions from the three corners of the simplex.
	float n0, n1, n2;
	static float F2 = 0.366035403f;
	static float G2 = 0.211324865f;

	// Using volatile to stop CUDA from dumping these in registers: we use them
	// frequently.
	int ix = floorf(px + ((px + py) * F2));
	int iy = floorf(py + ((px + py) * F2));

	float x0 = px - (ix - ((ix + iy) * G2));
	float y0 = py - (iy - ((ix + iy) * G2));

	// Find which simplex we're in, get offsets for middle corner in ij/simplex spcae
	short i1, j1;
	x0 > y0 ? i1 = 1 : i1 = 0;
	x0 > y0 ? j1 = 0 : j1 = 1;

	float x1, y1, x2, y2;
	x1 = x0 - i1 + G2;
	y1 = y0 - j1 + G2;
	x2 = x0 - 1.0f + 2.0f * G2;
	y2 = y0 - 1.0f + 2.0f * G2;

	// Get triangle coordinate hash to index into gradient table.
	uint h0 = hash_2d(ix, iy, seed);
	uint h1 = hash_2d(ix + i1, iy + j1, seed);
	uint h2 = hash_2d(ix + 1, iy + 1, seed);

	// Get values from table.
	short g0x = grad_2d_lut[h0][0];
	short g0y = grad_2d_lut[h0][1];
	short g1x = grad_2d_lut[h1][0];
	short g1y = grad_2d_lut[h1][1];
	short g2x = grad_2d_lut[h2][0];
	short g2y = grad_2d_lut[h2][1];

	// Now calculate contributions from 3 corners of the simplex
	float t0 = 0.50f - x0*x0 - y0*y0;
	// Squared/fourth-ed(?) t0.
	float t0_2, t0_4;
	if (t0 < 0.0f) {
		n0 = t0 = t0_2 = t0_4 = 0.0f;
	}
	else {
		t0_2 = t0 * t0;
		t0_4 = t0_2 * t0_2;
		n0 = t0_4 * (g0x * x0 + g0y * y0);
	}

	float t1 = 0.50f - x1*x1 - y1*y1;
	float t1_2, t1_4;
	if (t1 < 0.0f) {
		n1 = t1 = t1_2 = t1_4 = 0.0f;
	}
	else {
		t1_2 = t1 * t1;
		t1_4 = t1_2 * t1_2;
		n1 = t1_4 * (g1x*x1 + g1y*y1);
	}

	float t2 = 0.50f - x2*x2 - y2*y2;
	float t2_2, t2_4;
	if (t2 < 0.0f) {
		n2 = t2 = t2_2 = t2_4 = 0.0f;
	}
	else {
		t2_2 = t2 * t2;
		t2_4 = t2_2 * t2_2;
		n2 = t2_4 * (g2x*x2 + g2y*y2);
	}

	if (deriv != nullptr) {
		deriv->x = (t0_2 * t0 * (g0x*x0 + g0y*y0)) * x0;
		deriv->y = (t0_2 * t0 * (g0x*x0 + g0y*y0)) * y0;
		deriv->x += (t1_2 * t1 * (g1x*x1 + g1y*y1)) * x1;
		deriv->y += (t1_2 * t1 * (g1x*x1 + g1y*y1)) * y1;
		deriv->x += (t2_2 * t2 * (g2x*x2 + g2y*y2)) * x2;
		deriv->y += (t2_2 * t2 * (g2x*x2 + g2y*y2)) * y2;
		deriv->x *= -8.0f;
		deriv->y *= -8.0f;
		deriv->x += (t0_4 * g0x + t1_4 * g1x + t2_4 * g2x);
		deriv->y += (t0_4 * g0y + t1_4 * g1y + t2_4 * g2y);
		deriv->x *= 40.0f;
		deriv->y *= 40.0f;
	}

	return 40.0f * (n0 + n1 + n2);
}

// This LUT is from Stefan Gustavson's source code (along with the following simplex 3D method)
__device__ __constant__ short grad_3d_lut[256][3] =
{
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 },
	{ 1,0,0 },
	{ -1,0,0 },
	{ 0,0,1 },
	{ 0,0,-1 },
	{ 0,1,0 },
	{ 0,-1,0 }
};

__device__ float simplex3d(const float px, const float py, const float pz, const int seed, float3 * deriv){
	static float F3 = 0.333333333f;
	static float G3 = 0.166666667f;

	// Skew input space about and find our simplex cell and simplex coordinates in ijk space
	float3 s = make_float3(px, py, pz) + ((px + py + pz) * F3);
	int3 i_s = make_int3(floorf(s.x), floorf(s.y), floorf(s.z));

	// First positional coordinate
	float3 p0;
	p0.x = px - (i_s.x - ((i_s.x + i_s.y + i_s.z) * G3));
	p0.y = py - (i_s.y - ((i_s.x + i_s.y + i_s.z) * G3));
	p0.z = pz - (i_s.z - ((i_s.x + i_s.y + i_s.z) * G3));

	int3 i1, i2;

	if (p0.x >= p0.y) {
		if (p0.y >= p0.z) {
			i1.x = 1;
			i1.y = 0;
			i1.z = 0;
			i2.x = 1;
			i2.y = 1;
			i2.z = 0;
		}
		else if (p0.x >= p0.z) {
			i1.x = 1;
			i1.y = 0;
			i1.z = 0;
			i2.x = 1;
			i2.y = 0;
			i2.z = 1;
		}
		else {
			i1.x = 0;
			i1.y = 0;
			i1.z = 1;
			i2.x = 1;
			i2.y = 0;
			i2.z = 1;
		}
	}
	// If p0.x < p0.y
	else {
		if (p0.y < p0.z) {
			i1.x = 0;
			i1.y = 0;
			i1.z = 1;
			i2.x = 0;
			i2.y = 1;
			i2.z = 1;
		}
		else if (p0.x < p0.z) {
			i1.x = 0;
			i1.y = 1;
			i1.z = 0;
			i2.x = 0;
			i2.y = 1;
			i2.z = 1;
		}
		else {
			i1.x = 0;
			i1.y = 1;
			i1.z = 0;
			i2.x = 1;
			i2.y = 1;
			i2.z = 0;
		}
	}

	// Get simplex coords in xyz coords
	float3 p1, p2, p3;
	p1 = make_float3(p0.x - i1.x + G3, p0.y - i1.y + G3, p0.z - i1.z + G3);
	p2 = make_float3(p0.x - i2.x + (2.0f * G3), p0.y - i2.y + (2.0f * G3), p0.z - i2.z + (2.0f * G3));
	p3 = make_float3(p0.x - 1.0f + (3.0f * G3), p0.y - 1.0f + (3.0f * G3), p0.z - 1.0f + (3.0f * G3));

	// Hash coordinates
	uint h0, h1, h2, h3;
	h0 = hash_3d(i_s.x, i_s.y, i_s.z, seed);
	h1 = hash_3d(i_s.x + i1.x, i_s.y + i1.y, i_s.z + i1.z, seed);
	h2 = hash_3d(i_s.x + i2.x, i_s.y + i2.y, i_s.z + i2.z, seed);
	h3 = hash_3d(i_s.x + 1, i_s.y + 1, i_s.z + 1, seed);

	// Get gradient vectors using hash coordinates and get contributions from each of four corners.
	float3 g0, g1, g2, g3;
	g0 = make_float3(grad_3d_lut[h0][0], grad_3d_lut[h0][1], grad_3d_lut[h0][2]);
	g1 = make_float3(grad_3d_lut[h1][0], grad_3d_lut[h1][1], grad_3d_lut[h1][2]);
	g2 = make_float3(grad_3d_lut[h2][0], grad_3d_lut[h2][1], grad_3d_lut[h2][2]);
	g3 = make_float3(grad_3d_lut[h3][0], grad_3d_lut[h3][1], grad_3d_lut[h3][2]);


	float t0, t1, t2, t3;
	// Squares of t0-t3.
	volatile float t0_2, t1_2, t2_2, t3_2;

	// Actual contribution to final result
	float n0, n1, n2, n3;

	// First corner
	t0 = 0.60f - p0.x*p0.x - p0.y*p0.y - p0.z*p0.z;
	if (t0 < 0.0f) {
		t0 = t0_2 = n0 = 0.0f;
		g0 = make_float3(0, 0, 0);
	}
	else {
		t0_2 = t0 * t0;
		n0 = (t0_2 * t0_2) * (g0.x*p0.x + g0.y*p0.y + g0.z*p0.z);
	}

	// Second corner
	t1 = 0.60f - p1.x*p1.x - p1.y*p1.y - p1.z*p1.z;
	if (t1 < 0.0f) {
		t1 = t1_2 = n1 = 0.0f;
		g1 = make_float3(0, 0, 0);
	}
	else {
		t1_2 = t1 * t1;
		n1 = (t1_2 * t1_2) * (g1.x*p1.x + g1.y*p1.y + g1.z*p1.z);
	}

	// Third corner.
	t2 = 0.60f - p2.x*p2.x - p2.y*p2.y - p2.z*p2.z;
	if (t2 < 0.0f) {
		t2 = t2_2 = n2 = 0.0f;
		g2 = make_float3(0, 0, 0);
	}
	else {
		t2_2 = t2 * t2;
		n2 = (t2_2 * t2_2) * (g2.x*p2.x + g2.y*p2.y + g2.z*p2.z);
	}

	// Fourth and final corner.
	t3 = 0.60f - p3.x*p3.x - p3.y*p3.y - p3.z*p3.z;
	if (t3 < 0.0f) {
		t3 = t3_2 = n3 = 0.0f;
		g3 = make_float3(0, 0, 0);
	}
	else {
		t3_2 = t3 * t3;
		n3 = (t3_2 * t3_2) * (g3.x*p3.x + g3.y*p3.y + g3.z*p3.z);
	}

	// Add all contributions and scale.
	float result = 28.0f * (n0 + n1 + n2 + n3);

	// Calculate derivative, if desired (i.e deriv != nullptr)
	if (deriv != nullptr) {
		volatile float tmp0, tmp1, tmp2, tmp3;
		tmp0 = t0_2 * t0 * (g0.x*p0.x + g0.y*p0.y + g0.z*p0.z);
		deriv->x = tmp0 * p0.x;
		deriv->y = tmp0 * p0.y;
		deriv->z = tmp0 * p0.z;
		tmp1 = t1_2 * t1 * (g1.x*p1.x + g1.y*p1.y + g1.z*p1.z);
		deriv->x += tmp1 * p1.x;
		deriv->y += tmp1 * p1.y;
		deriv->z += tmp1 * p1.z;
		tmp2 = t2_2 * t2 * (g2.x*p2.x + g2.y*p2.y + g2.z*p2.z);
		deriv->x += tmp2 * p2.x;
		deriv->y += tmp2 * p2.y;
		deriv->z += tmp2 * p2.z;
		tmp3 = t3_2 * t3 * (g3.x*p3.x + g3.y*p3.y + g3.z*p3.z);
		deriv->x += tmp3 * p3.x;
		deriv->y += tmp3 * p3.y;
		deriv->z += tmp3 * p3.z;
		*deriv *= -8.0f;
		g0 *= (t0_2 * t0_2);
		g1 *= (t1_2 * t1_2);
		g2 *= (t2_2 * t2_2);
		g3 *= (t3_2 * t3_2);
		*deriv += (g0 + g1 + g2 + g3);
		*deriv *= 28.0f;
	}

	return result;
}