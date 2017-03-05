#ifndef BILLOW_CUH
#define BILLOW_CUH
#include "../common/CUDA_Include.h"
#include "../noise_generators.cuh"

void BillowLauncher(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !BILLOW_CUH
