#ifndef CLAMP_CUH
#define CLAMP_CUH
#include "../common/CUDA_Include.h"

void ClampLauncher(float* output, float* input, const int width, const int height, const float low_val, const float high_val);

#endif // !CLAMP_CUH
