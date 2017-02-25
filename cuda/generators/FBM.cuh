#ifndef FBM_CUH
#define FBM_CUH
#include "common\CUDA_Include.h"
#include "perlin.cuh"
#include "simplex.cuh"
#include "..\cuda_assert.h"

__global__ void FBM2DKernel(cudaSurfaceObject_t out, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

void FBM_Launcher(cudaSurfaceObject_t out, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

__global__ void FBM2DKernel_Simplex(cudaSurfaceObject_t out, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

void FBM_Launcher_Simplex(cudaSurfaceObject_t out, int width, int height, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !FBM_CUH
