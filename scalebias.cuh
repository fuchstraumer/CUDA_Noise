#ifndef SCALEBIAS_CUH
#define SCALEBIAS_CUH
#include "common\CUDA_Include.h"
#include "../cuda_assert.h"

void scalebiasLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input, const int width, const int height, float scale, float bias);

#endif // 
