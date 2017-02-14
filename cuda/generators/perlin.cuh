#ifndef PERLIN_CUH
#define PERLIN_CUH
#include "cuda_stdafx.cuh"
#include "cuda_assert.h"

typedef unsigned char uchar;

__device__ float lerp(const float a, const float b, const float c);
__device__ float ease(const float t);
__device__ float grad2(uchar hash, float2 p);
__device__ float grad3(uchar hash, float3 p);
__device__ float grad4(uchar hash, float4 p);
__device__ float perlin2d(float2 point, cudaTextureObject_t perm);

__global__ void perlin2D_Kernel(cudaSurfaceObject_t dest, cudaTextureObject_t perm_table, int width, int height, float2 origin);

void PerlinLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !PERLIN_2D_CUH
