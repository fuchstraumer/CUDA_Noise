#ifndef ABS_CUH
#define ABS_CUH
#include "../common/CUDA_Include.h"

void absLauncher(float* out, float* in, const int width, const int height);

void absLauncher3D(cnoise::Point* data, const int width, const int height);

#endif 
