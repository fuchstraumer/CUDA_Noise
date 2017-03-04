#ifndef ADD_CUH
#define ADD_CUH
#include "../common/CUDA_Include.h"

void AddLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input0, cudaSurfaceObject_t input1, const int width, const int height);

#endif // !ADD_CUH
