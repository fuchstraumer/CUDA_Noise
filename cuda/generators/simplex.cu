#include "simplex.cuh"

/*
	Ported from: https://github.com/ashima/webgl-noise/blob/master/src/noise3D.glsl
*/

__device__ int fastfloor(float n) {
	return static_cast<int>(n) > 0 ? static_cast<int>(n) : static_cast<int>(n) - 1;
}

__device__ float2 mod289(float2 n) {
	float x, y, z;
	x = n.x - floor(n.x * (1.0f / 289.0f)) * 289.0f;
	y = n.y - floor(n.y * (1.0f / 289.0f)) * 289.0f;
	return make_float2(x, y);
}

__device__ float3 mod289(float3 n) {
	float x, y, z;
	x = n.x - floor(n.x * (1.0f / 289.0f)) * 289.0f;
	y = n.y - floor(n.y * (1.0f / 289.0f)) * 289.0f;
	z = n.z - floor(n.z * (1.0f / 289.0f)) * 289.0f;
	return make_float3(x, y, z);
}

__device__ float3 permute(float3 n) {
	n.x = ((n.x * 34.0f) + 1.0f) * n.x;
	n.y = ((n.y * 34.0f) + 1.0f) * n.y;
	n.z = ((n.z * 34.0f) + 1.0f) * n.z;
	return mod289(n);
}

__global__ void simplex(float* result, float* x, float *y) {
	// Various constants
	const float4 C = make_float4(0.211324865405187f, 0.366025403784439f, -0.577350269189626f, 0.024390243902439f);

}