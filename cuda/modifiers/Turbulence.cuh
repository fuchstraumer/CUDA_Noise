#ifndef TURBULENCE_CUH
#define TURBULENCE_CUH
#include "../common/CUDA_Include.h"
#include "../generators/FBM.cuh"

void TurbulenceLauncher(float* out, const float* input, const int width, const int height, const noise_t noise_type, const int roughness, const int seed, const float strength, const float freq);

#endif // !TURBULENCE_CUH
