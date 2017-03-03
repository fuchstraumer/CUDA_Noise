#ifndef BILLOW_CUH
#define BILLOW_CUH
#include "common\CUDA_Include.h"
#include "..\cuda_assert.h"
#include "perlin.cuh"
#include "simplex.cuh"


__global__ void Billow2DKernel(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

void BillowLauncher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);



#endif // !BILLOW_CUH
