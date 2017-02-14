#include "vector_operators.cuh"

__device__ float4 operator*(const float4& v0, const float4& v1) {
	return make_float4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}

__device__ float4 operator*(const float& f, const float4& v) {
	return make_float4(f * v.x, f * v.y, f * v.z, f * v.w);
}

__device__ float4 operator/(const float4 & v0, const float4 & v1){
	return make_float4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}

__device__ float4 operator/(const float & f, const float4 & v){
	return make_float4(f / v.x, f / v.y, f / v.z, f / v.w);
}

__device__ float4 operator/(const float4& v, const float& f) {
	return make_float4(v.x / f, v.y / f, v.z / f, v.w / f);
}

__device__ float4 operator+(const float4 & v0, const float4 & v1){
	return make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}

__device__ float4 operator+(const float & f, const float4 & v){
	return make_float4(f + v.x, f + v.y, f + v.z, f + v.w);
}

__device__ float4 operator-(const float4 & v0, const float4 & v1) {
	return make_float4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}

__device__ float4 operator-(const float & f, const float4 & v){
	return make_float4(f - v.x, f - v.y, f - v.z, f - v.w);
}
