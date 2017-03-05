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

	return 0.0f;
}