#include "simplex.cuh"
#include "../vector_operators.cuh"
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

__device__ float simplex2d(float2 pos, float freq) {
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