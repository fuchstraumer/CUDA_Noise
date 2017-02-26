#include "simplex.cuh"
#include "../vector_operators.cuh"


/*

	2D Simplex noise, from: https://github.com/Auburns/FastNoiseSIMD/blob/master/FastNoiseSIMD/FastNoiseSIMD_internal.cpp
	Ported to CUDA, but credit goes to original creator. Used this SIMD code as the basis for mine, as it also doesn't
	require LUTs (like GLSL noise) but performs significantly better and still allows seeding (like LUT noise)

*/

__device__ int hash(const int& seed, const int3& pos) {
	int result;
	result = seed;
	static const int x_prime = 1619;
	static const int y_prime = 31337;
	static const int z_prime = 6971;
	result = (pos.x * x_prime) ^ result;
	result = (pos.y * y_prime) ^ result;
	result = (pos.z * z_prime) ^ result;

	result = ((result * result) * 60493) * result;
	result = (result >> 13) ^ result;

	return result;
}

__device__ int blend(int a, int b, int mask) {
	return ((~mask & a) | (mask & b));
}

__device__ float gradient(const int& seed, const int3& i_pos, float3& f_pos) {
	int h = hash(seed, i_pos);

	/*
	
		*reinterpret_cast<int*>( is used to just trick compiler into thinking I'm passing
		int data into the blend function, so I can use bitwise ops on it.

	*/
	int u = (h < 8);
	u = blend(*reinterpret_cast<int*>(&f_pos.y), *reinterpret_cast<int*>(&f_pos.x), u);
	int v = (h < 4);
	int h12o14 = ((h == 12) | (h == 14));
	h12o14 = blend(*reinterpret_cast<int*>(&f_pos.z), *reinterpret_cast<int*>(&f_pos.x), u);
	v = blend(h12o14, *reinterpret_cast<int*>(&f_pos.y), v);

	// Shuffle int data about, convert it to a float.
	float tmp0, tmp1;
	tmp0 = (u ^ (h << 31));
	tmp1 = (u ^ ((h & 2) << 30));

	return tmp0 + tmp1;
}

// Seed_offset is calculated in the fractal loop. Automatically wrapped to 0-512 range.
__device__ float simplex3d(float3 pos, const int seed) {
	// Constant values.
	static const float F3 = 1.0f / 3.0f;
	static const float G3 = 1.0f / 6.0f;

	// Skew point 
	float f = F3 * (pos.x + pos.y + pos.z);
	int3 p0;
	float3 pf0;

	// Integral component.
	pf0 = floorf_f3(f + pos);

	// Use cheaper round vs floor to get int3 of previous
	p0 = round_f3(pf0);

	// Get fractional component.
	float g = G3 * (pf0.x + pf0.y + pf0.z);

	// pf0 = fractional float part of pos, p0 = starting point, position0 so to speak.
	pf0 = pos - (p0 - g);

	// Going to use these masks shortly.
	int pf0x_ge_pf0y, pf0y_ge_pf0z, pf0x_ge_pf0z;

	if (pf0.x >= pf0.y) {
		pf0x_ge_pf0y = 0xffffffff;
	}
	else {
		pf0x_ge_pf0y = 0;
	}

	if (pf0.y >= pf0.z) {
		pf0y_ge_pf0z = 0xffffffff;
	}
	else {
		pf0y_ge_pf0z = 0;
	}

	if (pf0.x >= pf0.z) {
		pf0x_ge_pf0z = 0xffffffff;
	}
	else {
		pf0x_ge_pf0z = 0;
	}

	// Use masks found above
	int3 ijk0;
	ijk0.x = (1 & (pf0x_ge_pf0y & pf0y_ge_pf0z));
	ijk0.y = (1 & (~pf0x_ge_pf0y & pf0y_ge_pf0z));
	ijk0.z = (1 & (~pf0x_ge_pf0z & ~pf0y_ge_pf0z));

	int3 ijk1;
	ijk1.x = (1 & (pf0x_ge_pf0y | pf0x_ge_pf0z));
	ijk1.y = (1 & (~pf0x_ge_pf0y | pf0y_ge_pf0z));
	ijk1.z = (1 & ~(pf0x_ge_pf0z & pf0y_ge_pf0z));

	float3 f0, f1, f2;
	f0 = (pf0 - ijk0) + G3;
	f1 = (pf0 - ijk1) + (2.0f / 6.0f);
	f2 = (pf0 - 1.0f) + (3.0f / 6.0f);

	float4 t;
	t.x = ((0.6f - (pf0.x * pf0.x)) - (pf0.y * pf0.y)) - (pf0.z * pf0.z);
	t.y = ((0.6f - (f0.x * f0.x)) - (f0.y * f0.y)) - (f0.z * f0.z);
	t.z = ((0.6f - (f1.x * f1.x)) - (f1.y * f1.y)) - (f1.z * f1.z);
	t.w = ((0.6f - (f2.x * f2.x)) - (f2.y * f2.y)) - (f2.z * f2.z);

	// Test for correct magnitude of t (above 0)
	bool m0, m1, m2, m3;
	m0 = (t.x > 0.0f);
	m1 = (t.y > 0.0f);
	m2 = (t.z > 0.0f);
	m3 = (t.w > 0.0f);

	// Square components of t
	t = t * t;
	// Square them again.
	t = t * t;

	// Gradient components.
	float4 n;

	// If gradient t is out of bounds, no contribution from that corner of the simplex.
	if (m0) {
		n.x = gradient(seed, p0, pf0);
	}
	else {
		n.x = 0.0f;
	}

	if (m1) {
		n.y = gradient(seed, p0 + ijk0, f0);
	}
	else {
		n.y = 0.0f;
	}

	if (m2) {
		n.z = gradient(seed, p0 + ijk1, f1);
	}
	else {
		n.z = 0.0f;
	}

	if (m3) {
		n.w = gradient(seed, p0 + 1, f2);
	}

	// Last step.
	n = t * n;

	// And return final value.
	return 32.0f * (n.x + n.y + n.z + n.w);
}

__device__ float simplex2d(float2 pos, const int seed) {
	return simplex3d(make_float3(0.0f, pos.x, pos.y), seed);
}

/*
Ported to CUDA from the "WebGL" noise repo.
https://github.com/ashima/webgl-noise/tree/master/src
*/

__device__ float fract(float x) {
	return x - floorf(x);
}

__device__ float2 mod289(float2 x) {
	return x - make_float2(floorf(x.x * (1.0f / 289.0f)) * 289.0f, floorf(x.y * (1.0f / 289.0f)) * 289.0f);
}

__device__ float3 mod289(float3 x) {
	return x - make_float3(floorf(x.x * (1.0f / 289.0f)) * 289.0f, floorf(x.y * (1.0f / 289.0f)) * 289.0f, floorf(x.z * (1.0f / 289.0f)) * 289.0f);
}

__device__ float3 permute(float3 x) {
	float3 _x = ((x * 34.0f) + 1.0f) * x;
	return mod289(_x);
}

__device__ float glsl_simplex2d(float2 pos, float freq) {
	float4 C = make_float4(0.211324865405187f,  // (3.0-sqrt(3.0))/6.0
						   0.366025403784439f,  // 0.5*(sqrt(3.0)-1.0)
						  -0.577350269189626f,  // -1.0 + 2.0 * C.x
						   0.024390243902439f); // 1.0 / 41.0)
	pos.x *= freq;
	pos.y *= freq;

	float dp = dot(pos, make_float2(C.y, C.y));
	int2 i = make_int2(floorf(pos.x + dp), floorf(pos.y + dp));

	// First corner.
	float2 x0 = pos - make_float2(i.x, i.y) + dot(make_float2(i.x, i.y), make_float2(C.x, C.x));

	// Other corners.
	int2 i1;
	if (x0.x > x0.y) {
		i1 = make_int2(1, 0);
	}
	else {
		i1 = make_int2(0, 1);
	}

	float4 x12 = make_float4(x0.x + C.x, x0.y + C.x, x0.x + C.z, x0.y + C.z);
	x12.x = x12.x - i1.x;
	x12.y = x12.y - i1.y;

	// Permutations
	float2 tmp = mod289(make_float2(i.x, i.y));
	i = make_int2(tmp.x, tmp.y);
	float3 p = permute(permute(i.y + make_float3(0.0f, i1.y, 1.0f)) + i.x + make_float3(0.0f, i1.x, 1.0f));
	float3 m;
	float3 ret = 0.50f - make_float3(dot(x0, x0), dot(make_float2(x12.x, x12.y), make_float2(x12.x, x12.y)), dot(make_float2(x12.z, x12.w), make_float2(x12.z, x12.w)));
	if (ret.x > 0.0f || ret.y > 0.0f || ret.z > 0.0f) {
		m = ret;
	}
	else {
		m = make_float3(0.0f, 0.0f, 0.0f);
	}
	m = m * m;
	m = m * m;

	// Getting final gradient points.
	float3 x = 2.0f * make_float3(fract(p.x * C.w), fract(p.y * C.w), fract(p.z * C.w)) - 1.0f;
	float3 h = make_float3(fabsf(x.x), fabsf(x.y), fabsf(x.z)) - 0.50f;
	float3 ox = make_float3(floorf(x.x + 0.50f), floorf(x.y + 0.50f), floorf(x.z + 0.50f));
	float3 a0 = x - ox;

	m = m * (1.79284291400159f - 0.85373472095314f * (a0*a0 + h*h));

	// FInal value
	float3 g;
	g.x = a0.x * x0.x + h.x * x0.y;
	g.y = a0.y * x12.x + h.y * x12.y;
	g.z = a0.z * x12.z + h.z * x12.w;

	return 130.0f * dot(m, g);
}