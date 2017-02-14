#include "vector_operators.cuh"

__device__ float2 operator*(const float2& v0, const float2& v1) {
	return make_float2(v0.x * v1.x, v0.y * v1.y);
}

__device__ float2 operator*(const int2 & v0, const float2 & v1){
	return make_float2(__int2float_rn(v0.x) * v1.x, __int2float_rn(v0.y) * v1.y);
}

__device__ float2 operator/(const float2 & v0, const float2 & v1){
	return make_float2(v0.x / v1.x, v0.y / v1.y);
}

__device__ float2 operator+(const float2 & v0, const float2 & v1){
	return make_float2(v0.x + v1.x, v0.y + v1.y);
}

__device__ float2 operator+(const int2 & v0, const float2 & v1){
	return make_float2(__int2float_rn(v0.x) + v1.x, __int2float_rn(v0.y) + v1.y);
}

__device__ float2 operator-(const float2 & v0, const float2 & v1){
	return make_float2(v0.x - v1.x, v0.y - v1.y);
}

__device__ float2 operator-(const int2 & v0, const float2 & v1){
	return make_float2(__int2float_rn(v0.x) - v1.x, __int2float_rn(v0.y) - v1.y);
}

__device__ float2 operator*(const float & f, const float2 & v){
	return make_float2(f * v.x, f * v.y);
}

__device__ float2 operator*(const float2 & v, const float & f){
	return make_float2(v.x * f, v.y * f);
}

__device__ float2 operator*(const int & i, const float2 & v){
	float f = __int2float_rn(i);
	return f * v;
}

__device__ float2 operator*(const float2 & v, const int & i){
	float f = __int2float_rn(i);
	return v * f;
}

__device__ float2 operator/(const float & f, const float2 & v){
	return make_float2(f / v.x, f / v.y);
}

__device__ float2 operator/(const float2 & v, const float & f){
	return make_float2(v.x / f, v.y / f);
}

__device__ float2 operator/(const int & i, const float2 & v){
	float f = __int2float_rn(i);
	return f / v;
}

__device__ float2 operator/(const float2 & v, const int & i){
	float f = __int2float_rn(i);
	return v / f;
}

__device__ float2 operator+(const float & f, const float2 & v){
	return make_float2(f + v.x, f + v.y);
}

__device__ float2 operator+(const float2 & v, const float & f){
	return make_float2(v.x + f, v.y + f);
}

__device__ float2 operator+(const int & i, const float2 & v){
	float f = __int2float_rn(i);
	return f + v;
}

__device__ float2 operator+(const float2 & v, const int & i){
	float f = __int2float_rn(i);
	return v + f;
}

__device__ float2 operator-(const float & f, const float2 & v){
	return make_float2(f - v.x, f - v.y);
}

__device__ float2 operator-(const float2 & v, const float & f){
	return make_float2(v.x - f, v.y - f);
}

__device__ float2 operator-(const int & i, const float2 & v){
	float f = __int2float_rn(i);
	return f - v;
}

__device__ float2 operator-(const float2 & v, const int & i){
	float f = __int2float_rn(i);
	return v - f;
}

__device__ float4 operator*(const float4& v0, const float4& v1) {
	return make_float4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}

__device__ float4 operator*(const int4 & v0, const float4 & v1){
	return make_float4(__int2float_rn(v0.x) + v1.x, __int2float_rn(v0.y) + v1.y,
					   __int2float_rn(v0.z) + v1.z, __int2float_rn(v0.w) + v1.w);
}

__device__ float4 operator*(const float& f, const float4& v) {
	return make_float4(f * v.x, f * v.y, f * v.z, f * v.w);
}

__device__ float4 operator*(const float4& v, const float& f) {
	return make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
}

__device__ float4 operator*(const int & i, const float4 & v){
	return __int2float_rn(i) * v;
}

__device__ float4 operator*(const float4 & v, const int & i){
	return v * __int2float_rn(i);
}

__device__ float4 operator/(const float4 & v0, const float4 & v1){
	return make_float4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}

__device__ float4 operator/(const int4 & v0, const float4 & v1){
	return make_float4(__int2float_rn(v0.x) / v1.x, __int2float_rn(v0.y) / v1.y,
		__int2float_rn(v0.z) / v1.z, __int2float_rn(v0.w) / v1.w);
}

__device__ float4 operator/(const float & f, const float4 & v){
	return make_float4(f / v.x, f / v.y, f / v.z, f / v.w);
}

__device__ float4 operator/(const float4& v, const float& f) {
	return make_float4(v.x / f, v.y / f, v.z / f, v.w / f);
}

__device__ float4 operator/(const int & i, const float4 & v){
	return __int2float_rn(i) / v;
}

__device__ float4 operator/(const float4 & v, const int & i){
	return v / __int2float_rn(i);
}

__device__ float4 operator+(const float4 & v0, const float4 & v1){
	return make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}

__device__ float4 operator+(const int4 & v0, const float4 & v1){
	return make_float4(__int2float_rn(v0.x) + v1.x, __int2float_rn(v0.y) + v1.y,
		__int2float_rn(v0.z) + v1.z, __int2float_rn(v0.w) + v1.w);
}

__device__ float4 operator+(const float & f, const float4 & v){
	return make_float4(f + v.x, f + v.y, f + v.z, f + v.w);
}

__device__ float4 operator+(const float4& v, const float& f) {
	return make_float4(f + v.x, f + v.y, f + v.z, f + v.w);
}

__device__ float4 operator+(const int & i, const float4 & v){
	return __int2float_rn(i) + v;
}

__device__ float4 operator+(const float4 & v, const int & i){
	return v + __int2float_rn(i);
}

__device__ float4 operator-(const float4 & v0, const float4 & v1) {
	return make_float4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}

__device__ float4 operator-(const int4 & v0, const float4 & v1){
	return make_float4(__int2float_rn(v0.x) - v1.x, __int2float_rn(v0.y) - v1.y,
		__int2float_rn(v0.z) - v1.z, __int2float_rn(v0.w) - v1.w);
}

__device__ float4 operator-(const float & f, const float4 & v){
	return make_float4(f - v.x, f - v.y, f - v.z, f - v.w);
}

__device__ float4 operator-(const float4& v, const float& f) {
	return make_float4(v.x - f, v.y - f, v.z - f, v.w - f);
}

__device__ float4 operator-(const int & i, const float4 & v){
	return __int2float_rn(i) - v;
}

__device__ float4 operator-(const float4 & v, const int & i){
	return v - __int2float_rn(i);
}

__device__ float dot(const float2 & v0, const float2 & v1){
	return v0.x*v1.x + v0.y*v1.y;
}

__device__ float dot(const float3& v0, const float3& v1) {
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

__device__ float dot(const float4& v0, const float4& v1) {
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z + v0.w*v1.w;
}