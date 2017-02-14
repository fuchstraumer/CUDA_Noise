#ifndef PERLIN_CUH
#define PERLIN_CUH
#include "cuda_stdafx.cuh"

__global__ void perlin2D_Kernel(cudaSurfaceObject_t dest, cudaTextureObject_t perm_table, int width, int height, float2 origin);

#endif // !PERLIN_2D_CUH
