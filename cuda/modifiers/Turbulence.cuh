#ifndef TURBULENCE_CUH
#define TURBULENCE_CUH
#include "common\CUDA_Include.h"
#include "../cuda_assert.h"
#include "../generators/FBM.cuh"

void TurbulenceLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t input, int width, int height, noise_t noise_type, int roughness, int seed, float strength);

#endif // !TURBULENCE_CUH
