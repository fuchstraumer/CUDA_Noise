#include "vector_operators.cuh"

/*

	UTILITY FUNCTIONS

*/

__device__ int2 floor_f2(const float2& v) {
	return make_int2(floorf(v.x), floorf(v.y));
}

__device__ int2 round_f2(const float2& v) {
	return make_int2(llrintf(v.x), llrintf(v.y));
}

__device__ int3 floor_f3(const float3& v) {
	return make_int3(floorf(v.x), floorf(v.y), floorf(v.z));
}

__device__ int3 round_f3(const float3& v) {
	return make_int3(llrintf(v.x), llrintf(v.y), llrintf(v.z));
}

__device__ float3 int3_to_f3(const int3& v) {
	return make_float3(__int2float_rn(v.x), __int2float_rn(v.y), __int2float_rn(v.z));
}

/*
	
	INT3 OPERATORS

*/


__device__ int3 operator*(const int3 & v0, const int3 & v1){
	return make_int3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}

__device__ int3 operator/(const int3 & v0, const int3 & v1){
	return make_int3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}

__device__ int3 operator+(const int3 & v0, const int3 & v1){
	return make_int3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}

__device__ int3 operator-(const int3 & v0, const int3 & v1)
{
	return make_int3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

__device__ int3 operator*(const float3 & v0, const int3 & v1){
	int3 iv0 = floor_f3(v0);
	return iv0 * v1;
}

__device__ int3 operator*(const float & f, const int3 & v){
	int i = llrintf(f);
	return  make_int3(i * v.x, i * v.y, i * v.z);
}

__device__ int3 operator*(const int3 & v, const float & f){
	int i = llrintf(f);
	return make_int3(v.x * i, v.y * i, v.z * i);
}

__device__ int3 operator*(const int & i, const int3 & v){
	return make_int3(i * v.x, i * v.y, i * v.z);
}

__device__ int3 operator*(const int3 & v, const int & i){
	return make_int3(v.x * i, v.y * i, v.z * i);
}

__device__ int3 operator/(const int & i, const int3 & v){
	return make_int3(i / v.x, i / v.y, i / v.z);
}

__device__ int3 operator/(const int3 & v, const int & i){
	return make_int3(v.x / i, v.y / i, v.z / i);
}

__device__ int3 operator/(const float & f, const int3 & v){
	int i = llrintf(f);
	return i / v;
}

__device__ int3 operator/(const int3 & v, const float & f){
	int i = llrintf(f);
	return v / i;
}

__device__ int3 operator+(const int & i, const int3 & v){
	return make_int3(i + v.x, i + v.y, i + v.z);
}

__device__ int3 operator+(const int3 & v, const int & i){
	return make_int3(v.x + i, v.y + i, v.z + i);
}

__device__ int3 operator+(const float & f, const int3 & v){
	int i = llrintf(f);
	return i + v;
}

__device__ int3 operator+(const int3 & v, const float & f){
	int i = llrintf(f);
	return v + i;
}

__device__ int3 operator-(const int & i, const int3 & v){
	return make_int3(i - v.x, i - v.y, i - v.z);
}

__device__ int3 operator-(const int3 & v, const int & i){
	return make_int3(v.x - i, v.y - i, v.z - i);
}

__device__ int3 operator-(const float & f, const int3 & v){
	int i = llrintf(f);
	return i - v;
}

__device__ int3 operator-(const int3 & v, const float & f){
	int i = llrintf(f);
	return v - 1;
}

__device__ int3 operator&(const int3 & v0, const int3 & v1){
	return make_int3(v0.x & v1.x, v0.y & v1.y, v0.y & v1.y);
}

__device__ int3 operator&(const int3 & v, const int & i){
	return make_int3(v.x & i, v.y & i, v.z & i);
}

__device__ int3 operator&(const int & i, const int3 & v){
	return make_int3(i & v.x, i & v.y, i & v.z);
}

__device__ int3 operator|(const int3 & v0, const int3 & v1){
	return make_int3(v0.x | v1.x, v0.y | v1.y, v0.z | v1.z);
}

__device__ int3 operator|(const int3 & v, const int & i){
	return make_int3(v.x | i, v.y | i, v.z | i);
}

__device__ int3 operator|(const int & i, const int3 & v){
	return make_int3(i | v.x, i | v.y, i | v.z);
}

__device__ int3 operator^(const int3 & v0, const int3 & v1){
	return make_int3(v0.x ^ v1.x, v0.y ^ v1.y, v0.z ^ v1.z);
}

__device__ int3 operator^(const int3 & v, const int & i){
	return make_int3(v.x ^ i, v.y ^ i, v.z ^ i);
}

__device__ int3 operator^(const int & i, const int3 & v){
	return make_int3(i ^ v.x, i ^ v.y, i ^ v.z);
}

__device__ int3 operator~(const int3 & v0){
	return make_int3(~v0.x, ~v0.y, ~v0.z);
}

__device__ int3 and_not(const int3 & v0, const int3 & v1){
	int3 vN = ~v0;
	return vN & v1;
}

__device__ int3 operator>(const int3 & v0, const int3 & v1){
	return make_int3(
		v0.x > v1.x ? v0.x : v1.x ,
		v0.y > v1.y ? v0.y : v1.y ,
		v0.z > v1.z ? v0.z : v1.z );
}

__device__ int3 operator<(const int3 & v0, const int3 & v1){
	return make_int3(
		v0.x < v1.x ? v0.x : v1.x ,
		v0.y < v1.y ? v0.y : v1.y ,
		v0.z < v1.z ? v0.z : v1.z );
}

__device__ int3 operator>=(const int3 & v0, const int3 & v1){
	return make_int3(
		v0.x >= v1.x ? v0.x : v1.x ,
		v0.y >= v1.y ? v0.y : v1.y ,
		v0.z >= v1.z ? v0.z : v1.z );
}

__device__ int3 operator<=(const int3 & v0, const int3 & v1){
	return make_int3(
		v0.x <= v1.x ? v0.x : v1.x ,
		v0.y <= v1.y ? v0.y : v1.y ,
		v0.z <= v1.z ? v0.z : v1.z );
}

__device__ int3 operator >> (const int3 & v, const size_t & amt){
	return make_int3(v.x >> amt, v.y >> amt, v.z >> amt);
}

__device__ int3 operator >> (const int3 & v, const int3 & mask){
	return make_int3(v.x >> mask.x, v.y >> mask.y, v.z >> mask.z);
}

__device__ int3 operator<<(const int3 & v, size_t & amt){
	return make_int3(v.x << amt, v.y << amt, v.z << amt);
}

__device__ int3 operator<<(const int3 & v, const int3 & mask){
	return make_int3(v.x << mask.x, v.y << mask.y, v.z << mask.z);
}

/*
	
	FLOAT2 OPERATORS

*/

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

/*

	FLOAT3 OPERATORS

*/

__device__ float3 floorf_f3(const float3 & v) {
	return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

__device__ float3 operator*(const float3& v0, const float3& v1) {
	return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}

__device__ float3 operator*(const float& f, const float3& v) {
	return make_float3(f * v.x, f * v.y, f * v.z);
}

__device__ float3 operator*(const float3& v, const float& f) {
	return make_float3(v.x * f, v.y * f, v.z * f);
}

__device__ float3 operator*(const int & i, const float3 & v) {
	return __int2float_rn(i) * v;
}

__device__ float3 operator*(const float3 & v, const int & i) {
	return v * __int2float_rn(i);
}

__device__ float3 operator/(const float3 & v0, const float3 & v1) {
	return make_float3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}

__device__ float3 operator/(const int3 & v0, const float3 & v1) {
	return make_float3(__int2float_rn(v0.x) / v1.x, __int2float_rn(v0.y) / v1.y,
		__int2float_rn(v0.z) / v1.z);
}

__device__ float3 operator/(const float & f, const float3 & v) {
	return make_float3(f / v.x, f / v.y, f / v.z);
}

__device__ float3 operator/(const float3& v, const float& f) {
	return make_float3(v.x / f, v.y / f, v.z / f);
}

__device__ float3 operator/(const int & i, const float3 & v) {
	return __int2float_rn(i) / v;
}

__device__ float3 operator/(const float3 & v, const int & i) {
	return v / __int2float_rn(i);
}

__device__ float3 operator+(const float3 & v0, const float3 & v1) {
	return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}

__device__ float3 operator+(const int3 & v0, const float3 & v1) {
	return make_float3(__int2float_rn(v0.x) + v1.x, __int2float_rn(v0.y) + v1.y,
		__int2float_rn(v0.z) + v1.z);
}

__device__ float3 operator+(const float & f, const float3 & v) {
	return make_float3(f + v.x, f + v.y, f + v.z);
}

__device__ float3 operator+(const float3& v, const float& f) {
	return make_float3(f + v.x, f + v.y, f + v.z);
}

__device__ float3 operator+(const int & i, const float3 & v) {
	return __int2float_rn(i) + v;
}

__device__ float3 operator+(const float3 & v, const int & i) {
	return v + __int2float_rn(i);
}

__device__ float3 operator-(const float3 & v0, const float3 & v1) {
	return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

__device__ float3 operator/(const float3 & v0, const int3 & v1){
	float3 vf1 = int3_to_f3(v1);
	return v0 / vf1;
}

__device__ float3 operator+(const float3 & v0, const int3 & v1){
	float3 vf1 = int3_to_f3(v1);
	return v0 + vf1;
}

__device__ float3 operator-(const float3 & v0, const int3 & v1){
	float3 vf1 = int3_to_f3(v1);
	return v0 - vf1;
}

__device__ float3 operator-(const int3 & v0, const float3 & v1) {
	return make_float3(__int2float_rn(v0.x) - v1.x, __int2float_rn(v0.y) - v1.y,
		__int2float_rn(v0.z) - v1.z);
}

__device__ float3 operator-(const float & f, const float3 & v) {
	return make_float3(f - v.x, f - v.y, f - v.z);
}

__device__ float3 operator-(const float3& v, const float& f) {
	return make_float3(v.x - f, v.y - f, v.z - f);
}

__device__ float3 operator-(const int & i, const float3 & v) {
	return __int2float_rn(i) - v;
}

__device__ float3 operator-(const float3 & v, const int & i) {
	return v - __int2float_rn(i);
}

/*

	FLOAT4 OPERATORS

*/

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

/*

	GENERAL MATHEMATICAL FUNCTIONS

*/

__device__ float dot(const float2 & v0, const float2 & v1){
	return v0.x*v1.x + v0.y*v1.y;
}

__device__ float dot(const float3& v0, const float3& v1) {
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

__device__ float dot(const float4& v0, const float4& v1) {
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z + v0.w*v1.w;
}

__device__ int dot(const int2 & v0, const int2 & v1) {
	return v0.x*v1.x + v0.y*v1.y;
}

__device__ int dot(const int3& v0, const int3& v1) {
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

__device__ int dot(const int4& v0, const int4& v1) {
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z + v0.w*v1.w;
}
