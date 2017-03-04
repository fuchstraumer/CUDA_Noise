#ifndef SELECT_CUH
#define SELECT_CUH
#include "../common/CUDA_Include.h"

__global__ void SelectKernel(cudaSurfaceObject_t out, cudaSurfaceObject_t select_item, cudaSurfaceObject_t subject0, cudaSurfaceObject_t subject1, int width, int height, float upper_bound, float lower_bound, float falloff);

void SelectLauncher(cudaSurfaceObject_t out, cudaSurfaceObject_t select_item, cudaSurfaceObject_t subject0, cudaSurfaceObject_t subject1, int width, int height, float upper_bound, float lower_bound, float falloff);

#endif // !SELECT_CUH