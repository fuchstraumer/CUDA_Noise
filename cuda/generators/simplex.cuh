#ifndef SIMPLEX_CUH
#define SIMPLEX_CUH
#include "cuda_stdafx.cuh"

__device__ float simplex2d(float2 point);

__device__ float simplex3d(float3 point);

__device__ float simplex4d(float4 point);

#endif // !SIMPLEX_3D_CUH
