#ifndef MAX_CUH
#define MAX_CUH
#include "../common/CUDA_Include.h"

void MaxLauncher(float* output, const float* in0, const float* in1, const int width, const int height);

#endif // !MAX_CUH
