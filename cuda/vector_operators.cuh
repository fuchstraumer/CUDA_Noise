#ifndef VECTOR_OPERATORS_CUH
#define VECTOR_OPERATORS_CUH
#include "common\CUDA_Include.h"

/*

	INT2 OPERATORS

*/

__device__ int2 floor_f2(const float2& v);

__device__ int2 round_f2(const float2& v);


/*

	INT3 OPERATORS

*/

// Floating/rounding for int3

__device__ int3 floor_f3(const float3& v);

__device__ int3 round_f3(const float3& v);

// Binary operators w/ other int3's

__device__ int3 operator*(const int3& v0, const int3& v1);

__device__ int3 operator/(const int3& v0, const int3& v1);

__device__ int3 operator+(const int3& v0, const int3& v1);

__device__ int3 operator-(const int3& v0, const int3& v1);

// Binary operators w/ other vector types of similar dimensions

__device__ int3 operator*(const int3& v0, const float3& v1);

__device__ int3 operator*(const float3& v0, const int3& v1);

// Binary operators with other scalar types

__device__ int3 operator*(const float& f, const int3& v);

__device__ int3 operator*(const int3& v, const float& f);

__device__ int3 operator*(const int& i, const int3& v);

__device__ int3 operator*(const int3& v, const int& i);

__device__ int3 operator/(const float& f, const int3& v);

__device__ int3 operator/(const int3& v, const float& f);

__device__ int3 operator/(const int& i, const int3& v);

__device__ int3 operator/(const int3& v, const int& i);

__device__ int3 operator+(const float& f, const int3& v);

__device__ int3 operator+(const int3& v, const float& f);

__device__ int3 operator+(const int& i, const int3& v);

__device__ int3 operator+(const int3& v, const int& i);

__device__ int3 operator-(const float& f, const int3& v);

__device__ int3 operator-(const int3& v, const float& f);

__device__ int3 operator-(const int& i, const int3 &v);

__device__ int3 operator-(const int3& v, const int& i);

// FOR INT VECTOR TYPES ONLY: LOGICAL OPERATORS w/ INTS AND INT VECTORS

__device__ int3 operator&(const int3& v0, const int3& v1);

__device__ int3 operator&(const int3& v, const int& i);

__device__ int3 operator&(const int& i, const int3& v);

__device__ int3 operator|(const int3& v0, const int3& v1);

__device__ int3 operator|(const int3& v, const int& i);

__device__ int3 operator|(const int& i, const int3& v);

__device__ int3 operator^(const int3& v0, const int3& v1);

__device__ int3 operator^(const int3& v, const int& i);

__device__ int3 operator^(const int& i, const int3& v);

__device__ int3 operator~(const int3& v0);

__device__ int3 and_not(const int3& v0, const int3& v1);

// LOGICAL COMPARATORS - RETURNS MOST SIGNIFICANT ELEMENT IN EACH LOCATION

__device__ int3 operator>(const int3& v0, const int3& v1);

__device__ int3 operator<(const int3& v0, const int3& v1);

__device__ int3 operator>=(const int3& v0, const int3& v1);

__device__ int3 operator<=(const int3& v0, const int3& v1);

// BIT-SHIFTING

__device__ int3 operator>>(const int3& v, const size_t& amt);

__device__ int3 operator>>(const int3& v, const int3& mask);

__device__ int3 operator<<(const int3& v, size_t& amt);

__device__ int3 operator<<(const int3& v, const int3& mask);

/*

	FLOAT2 OPERATORS

*/

// Binary operators w/ other float2's

__device__ float2 operator*(const float2& v0, const float2& v1);

__device__ float2 operator*(const int2& v0, const float2& v1);

__device__ float2 operator/(const float2& v0, const float2& v1);

__device__ float2 operator+(const float2& v0, const float2& v1);

__device__ float2 operator+(const int2& v0, const float2& v1);

__device__ float2 operator-(const float2& v0, const float2& v1);

__device__ float2 operator-(const int2& v0, const float2& v1);

// Binary operators with other scalar types

__device__ float2 operator*(const float& f, const float2& v);

__device__ float2 operator*(const float2& v, const float& f);

__device__ float2 operator*(const int& i, const float2& v);

__device__ float2 operator*(const float2& v, const int& i);

__device__ float2 operator/(const float& f, const float2& v);

__device__ float2 operator/(const float2& v, const float& f);

__device__ float2 operator/(const int& i, const float2& v);

__device__ float2 operator/(const float2& v, const int& i);

__device__ float2 operator+(const float& f, const float2& v);

__device__ float2 operator+(const float2& v, const float& f);

__device__ float2 operator+(const int& i, const float2& v);

__device__ float2 operator+(const float2& v, const int& i);

__device__ float2 operator-(const float& f, const float2& v);

__device__ float2 operator-(const float2& v, const float& f);

__device__ float2 operator-(const int& i, const float2 &v);

__device__ float2 operator-(const float2& v, const int& i);

/*

	FLOAT3 OPERATORS

*/

// Floor float3

__device__ float3 floorf_f3(const float3& v);

// Binary operators w/ other float3's

__device__ float3 operator*(const float3& v0, const float3& v1);

__device__ float3 operator/(const float3& v0, const float3& v1);

__device__ float3 operator+(const float3& v0, const float3& v1);

__device__ float3 operator-(const float3& v0, const float3& v1);

// Binary operators w/ int3's

__device__ float3 operator/(const float3& v0, const int3& v1);

__device__ float3 operator+(const float3& v0, const int3& v1);

__device__ float3 operator-(const float3& v0, const int3& v1);

__device__ float3 operator/(const int3& v0, const float3& v1);

__device__ float3 operator+(const int3& v0, const float3& v1);

__device__ float3 operator-(const int3& v0, const float3& v1);

// Binary operators with other scalar types

__device__ float3 operator*(const float& f, const float3& v);

__device__ float3 operator*(const float3& v, const float& f);

__device__ float3 operator*(const int& i, const float3& v);

__device__ float3 operator*(const float3& v, const int& i);

__device__ float3 operator/(const float& f, const float3& v);

__device__ float3 operator/(const float3& v, const float& f);

__device__ float3 operator/(const int& i, const float3& v);

__device__ float3 operator/(const float3& v, const int& i);

__device__ float3 operator+(const float& f, const float3& v);

__device__ float3 operator+(const float3& v, const float& f);

__device__ float3 operator+(const int& i, const float3& v);

__device__ float3 operator+(const float3& v, const int& i);

__device__ float3 operator-(const float& f, const float3& v);

__device__ float3 operator-(const float3& v, const float& f);

__device__ float3 operator-(const int& i, const float3 &v);

__device__ float3 operator-(const float3& v, const int& i);

/*

	FLOAT4 OPERATORS

*/

// Binary operators w/ other float4's

__device__ float4 operator*(const float4& v0, const float4& v1);

__device__ float4 operator*(const int4& v0, const float4& v1);

__device__ float4 operator/(const float4& v0, const float4& v1);

__device__ float4 operator/(const int4& v0, const float4& v1);

__device__ float4 operator+(const float4& v0, const float4& v1);

__device__ float4 operator+(const int4& v0, const float4& v1);

__device__ float4 operator-(const float4& v0, const float4& v1);

__device__ float4 operator-(const int4& v0, const float4& v1);

// Binary operators with other scalar types

__device__ float4 operator*(const float& f, const float4& v);

__device__ float4 operator*(const float4& v, const float& f);

__device__ float4 operator*(const int& i, const float4& v);

__device__ float4 operator*(const float4& v, const int& i);

__device__ float4 operator/(const float& f, const float4& v);

__device__ float4 operator/(const float4& v, const float& f);

__device__ float4 operator/(const int& i, const float4& v);

__device__ float4 operator/(const float4& v, const int& i);

__device__ float4 operator+(const float& f, const float4& v);

__device__ float4 operator+(const float4& v, const float& f);

__device__ float4 operator+(const int& i, const float4& v);

__device__ float4 operator+(const float4& v, const int& i);

__device__ float4 operator-(const float& f, const float4& v);

__device__ float4 operator-(const float4& v, const float& f);

__device__ float4 operator-(const int& i, const float4 &v);

__device__ float4 operator-(const float4& v, const int& i);

/*

	MISC MATHEMATICAL OPERATORS AND FUNCTIONS, ALL TYPES

*/

// Math operators, like dot etc

__device__ float dot(const float2& v0, const float2& v1);

__device__ float dot(const float3& v0, const float3& v1);

__device__ float dot(const float4& v0, const float4& v1);

__device__ int dot(const int2& v0, const int2& v1);

__device__ int dot(const int3& v0, const int3& v1);

__device__ int dot(const int4& v0, const int4& v1);

#endif // !VECTOR_OPERATORS_CUH
