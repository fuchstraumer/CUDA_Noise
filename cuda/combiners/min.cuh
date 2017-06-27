#ifndef MIN_CUH
#define MIN_CUH
#include "../common/CUDA_Include.h"

void MinLauncher(float *output, const float* in0, const float* in1, const int width, const int height);

void MinLauncher3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height);

#endif // !MIN_CUH
