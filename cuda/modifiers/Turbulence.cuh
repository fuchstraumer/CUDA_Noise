#ifndef TURBULENCE_CUH
#define TURBULENCE_CUH
#include "common\CUDA_Include.h"
#include "../cuda_assert.h"
#include "../generators/FBM.cuh"

void TurbulenceLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t input, const int width, const int height, const noise_t noise_type, const int roughness, const int seed, const float strength);

#endif // !TURBULENCE_CUH
