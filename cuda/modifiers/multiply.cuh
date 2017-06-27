#ifndef MULTIPLY_CUH
#define MULTIPLY_CUH
#include "../common/CUDA_Include.h"

void multiplyLauncher(float* out, float* in, const int width, const int height, float factor);

// Left will be where we store the output values, right will be read-only.
void MultiplyLauncher3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height);

#endif 
