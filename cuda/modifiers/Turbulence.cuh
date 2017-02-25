#ifndef TURBULENCE_CUH
#define TURBULENCE_CUH
#include "common\CUDA_Include.h"
#include "../cuda_assert.h"
#include "../generators/perlin.cuh"
#include "../generators/simplex.cuh"

__device__ float Turbluence(cudaSurfaceObject_t out, cudaSurfaceObject_t input, int width, int height, float2 pos, float freq, int seed, float strength, noise_t noise_type);

__global__ void TurbulenceKernel(cudaSurfaceObject_t out, cudaSurfaceObject_t input, int width, int height, float2 pos, float freq, int seed, float strength, noise_t noise_type);

void TurbulenceLauncher(cudaSurfaceObject_t out, int width, int height, float freq, int seed, float strength, noise_t noise_type);

#endif // !TURBULENCE_CUH
