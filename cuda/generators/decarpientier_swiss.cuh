#ifndef DECARPIENTIER_SWISS_CUH
#define DECARPIENTIER_SWISS_CUH
#include "common\CUDA_Include.h"
#include "perlin.cuh"
#include "simplex.cuh"

void DecarpientierSwissLauncher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !DECARPIENTIER_SWISS_CUH
