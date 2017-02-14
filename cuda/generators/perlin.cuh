#ifndef PERLIN_CUH
#define PERLIN_CUH
#include "cuda_stdafx.cuh"
#include "cuda_assert.h"

typedef unsigned char uchar;

#ifndef HALF_PRECISION_SUPPORT

// 32-bit device functions for generating perlin noise.

__device__ float lerp(const float a, const float b, const float c);

__device__ float ease(const float t);

__device__ float grad2(uchar hash, float2 p);

__device__ float grad3(uchar hash, float3 p);

__device__ float grad4(uchar hash, float4 p);

__device__ float perlin2d(float2 point, cudaTextureObject_t perm);

__global__ void perlin2D_Kernel(cudaSurfaceObject_t dest, cudaTextureObject_t perm_table, int width, int height, float2 origin);


#endif // !HALF_PRECISION_SUPPORT

#ifdef HALF_PRECISION_SUPPORT

// 16-bit device functions for generating perlin noise.

// TODO: Half-precision version of the noise functions.

__device__ half lerp(const half a, const half b, const half c);

__device__ half ease(const half t);

__device__ half grad2(uchar hash, half2 p);

__device__ half perlin2d(half2 p, cudaTextureObject_t perm);

__global__ void perlin2D_KernelHalf(cudaSurfaceObject_t dest, cudaTextureObject_t perm, int width, int height, float2 origin);

// Can perform two operations at a time per thread! Could vastly increase speed,
// and should still work for noise generation!

#endif // HALF_PRECISION_SUPPORT

// Kernel launching function. Uses configurator utility to set optimal block and grid size based on our kernel.
void PerlinLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !PERLIN_2D_CUH
