#ifndef ADD_CUH
#define ADD_CUH
#include "common\CUDA_Include.h"
#include "../cuda_assert.h"

void AddLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input, const int width, const int height, float add_value);

#endif // !ADD_CUH
