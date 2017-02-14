#ifndef BILLOW_CUH
#define BILLOW_CUH
#include "..\cuda_stdafx.cuh"
#include "..\cuda_assert.h"
#include "perlin.cuh"

__global__ void Billow2DKernel(cudaSurfaceObject_t out, cudaTextureObject_t perm, int width, int height, float2 origin, float freq, float lacun, float persist, int init_seed, int octaves);

void BillowLauncher(cudaSurfaceObject_t out, cudaTextureObject_t perm, int width, int height, float2 origin, float freq, float lacun, float persist, int init_seed, int octaves);

#endif // !BILLOW_CUH
