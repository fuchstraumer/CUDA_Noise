#ifndef FBM_CUH
#define FBM_CUH
#include "common\CUDA_Include.h"
#include "perlin.cuh"
#include "simplex.cuh"
#include "..\cuda_assert.h"

__device__ float FBM2d_Simplex(float2 point, float freq, float lacun, float persist, int init_seed, float octaves);

__device__ float FBM2d(float2 point, float freq, float lacun, float persist, int init_seed, float octaves);

__global__ void FBM2DKernel(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

void FBM_Launcher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !FBM_CUH
