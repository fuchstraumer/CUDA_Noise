#ifndef POWER_CUH
#define POWER_CUH
#include "../common/CUDA_Include.h"

void powerLauncher(float* output, float* input0, float* input1, const int width, const int height);

void PowerLauncher3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height);

#endif 
