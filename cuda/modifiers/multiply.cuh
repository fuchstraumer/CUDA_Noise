#ifndef MULTIPLY_CUH
#define MULTIPLY_CUH
#include "common\CUDA_Include.h"
#include "..\cuda_assert.h"




void multiplyLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t in, const int width, const int height, float factor);


#endif 
