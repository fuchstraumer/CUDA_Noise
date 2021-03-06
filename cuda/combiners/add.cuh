#ifndef ADD_CUH
#define ADD_CUH
#include "../common/CUDA_Include.h"

void AddLauncher(float* output, float* input0, float* input1, const int width, const int height);

void AddLauncher3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height);

#endif // !ADD_CUH
