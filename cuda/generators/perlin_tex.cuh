#ifndef PERLIN_TEX_CUH
#define PERLIN_TEX_CUH
#include "common\CUDA_Include.h"

__device__ float perlin2d_tex(cudaTextureObject_t permutation, cudaTextureObject_t gradient, const float px, const float py, const int seed);

__device__ float FBM2d_tex(cudaTextureObject_t permutation, cudaTextureObject_t gradient, float px, float py, const float freq, const float lacun, const float persist, const int init_seed, const int octaves);

__global__ void texFBMKernel(cudaSurfaceObject_t output, cudaTextureObject_t permutation, cudaTextureObject_t gradient, const int width, const int height, const float2 origin, const float freq, const float lacun, const float persist, const int seed, const int octaves);

void texFBMLauncher(cudaSurfaceObject_t output, cudaTextureObject_t permutation, cudaTextureObject_t gradient, const int width, const int height, const float2 origin, const float freq, const float lacun, const float persist, const int seed, const int octaves);

#endif