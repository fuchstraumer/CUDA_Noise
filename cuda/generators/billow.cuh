#ifndef BILLOW_CUH
#define BILLOW_CUH
#include "..\cuda_stdafx.cuh"
#include "..\cuda_assert.h"
#include "perlin.cuh"

#ifndef HALF_PRECISION_SUPPORT
__global__ void Billow2DKernel(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

void BillowLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, cudaTextureObject_t grad, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);
#endif // !HALF_PRECISION_SUPPORT

#endif // !BILLOW_CUH
