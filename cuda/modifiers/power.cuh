#ifndef POWER_CUH
#define POWER_CUH
#include "../common/CUDA_Include.h"

void powerLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input, const int width, const int height, int p);


#endif 
