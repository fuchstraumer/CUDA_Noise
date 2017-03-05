#ifndef RIDGED_MULTI_CUH
#define RIDGED_MULTI_CUH

#include "../common/CUDA_Include.h"
#include "../noise_generators.cuh"

void RidgedMultiLauncher(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !RIDGED_MULTI_CUH
