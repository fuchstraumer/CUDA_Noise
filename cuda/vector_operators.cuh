#ifndef VECTOR_OPERATORS_CUH
#define VECTOR_OPERATORS_CUH
#include "common\CUDA_Include.h"

// Float2 operators


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

// Float3 operators.

// Binary operators w/ other float3's

__device__ float3 operator*(const float3& v0, const float3& v1);

__device__ float3 operator/(const float3& v0, const float3& v1);

__device__ float3 operator+(const float3& v0, const float3& v1);

__device__ float3 operator-(const float3& v0, const float3& v1);

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

// Float4 operators

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

// Math operators, like dot etc
__device__ float dot(const float2& v0, const float2& v1);

__device__ float dot(const float3& v0, const float3& v1);

__device__ float dot(const float4& v0, const float4& v1);

#endif // !VECTOR_OPERATORS_CUH
