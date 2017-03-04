#ifndef POWER_CUH
#define POWER_CUH
#include "../common/CUDA_Include.h"

void powerLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input0, cudaSurfaceObject_t input1, const int width, const int height);


#endif 
