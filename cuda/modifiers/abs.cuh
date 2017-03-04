#ifndef ABS_CUH
#define ABS_CUH
#include "../common/CUDA_Include.h"

void absLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t in, const int width, const int height);

#endif 
