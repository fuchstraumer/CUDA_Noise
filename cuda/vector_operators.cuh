#ifndef VECTOR_OPERATORS_CUH
#define VECTOR_OPERATORS_CUH
#include "cuda_stdafx.cuh"

__device__ float4 operator*(const float4& v0, const float4& v1);

__device__ float4 operator*(const float& f, const float4& v);

__device__ float4 operator/(const float4& v0, const float4& v1);

__device__ float4 operator/(const float& f, const float4& v);

__device__ float4 operator+(const float4& v0, const float4& v1);

__device__ float4 operator+(const float& f, const float4& v);

__device__ float4 operator-(const float4& v0, const float4& v1);

__device__ float4 operator-(const float& f, const float4& v);

#endif // !VECTOR_OPERATORS_CUH
